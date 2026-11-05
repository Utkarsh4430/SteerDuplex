"""Reference SteerBench inference for Moshi-family full-duplex models.

Covers three checkpoints, all through the same code path:

* public Moshi: ``hf_repo="kyutai/moshiko-pytorch-bf16"``, no ``checkpoint``;
* NVIDIA PersonaPlex: ``hf_repo="nvidia/personaplex-7b-v1"`` (or ``family="personaplex"``
  with a PersonaPlex ``.safetensors`` passed as ``checkpoint``);
* SteerDuplex: ``checkpoint=<.../consolidated>`` (``consolidated.safetensors`` +
  ``config.json``) on top of ``kyutai/moshiko-pytorch-bf16``, which supplies the
  Mimi codec and the text tokenizer.

Decoding protocol, batch size 1 (unchanged from the published runs):

1. Reset the streams. Prefill the voice prompt (normalised to -24 LUFS),
   0.5 s of silence, the system prompt wrapped as ``<system> ... <system>``,
   and another 0.5 s of silence (PersonaPlex ``step_system_prompts``).
2. Stream the user turn frame by frame (12.5 Hz), padded with silence up to
   ``max_duration_sec`` of timeline, sampling with temp 0.8 / top-k 250
   for audio and temp 0.7 / top-k 25 for text.
3. Stop early after 5 s of text silence once the model has produced text,
   after 5 s of audio silence (frame RMS < 0.002) once it has spoken, or after
   5 s past the end of the user turn with no answer at all.
4. If generation used the whole timeline without stopping (a safety-cap hit),
   regenerate the item once without reseeding, with the timeline raised to
   ``min(lossless_retry_max_duration, context_ceiling_sec, model window)``
   (300 s / 300 s / ``context`` frames). This is the "lossless retry".

The returned text is the model's own text stream with control tokens removed:
``"".join(t for t in pieces if not t.startswith("["))``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

# Same as the reference runs: keep torch.compile / dynamo out of the decode loop.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sphn  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from ..interface import ModelResponse, SteerBenchModel, seed_global_rng  # noqa: E402
from .vendor.moshi.models import loaders  # noqa: E402
from .vendor.moshi.models.lm import LMGen, _iterate_audio, encode_from_sphn  # noqa: E402

logger = logging.getLogger(__name__)

N_MIMI_CODEBOOKS = 8
DEFAULT_HF_REPO = "kyutai/moshiko-pytorch-bf16"

#: Mimi codec frame rate (Hz).
FRAME_RATE: float = 12.5
#: Native moshiko attention window (``loaders._lm_kwargs["context"]``), used only as a fallback.
BASE_MOSHI_CONTEXT_FRAMES: int = 3000


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def wrap_with_system_tags(text: str) -> str:
    """Wrap the system prompt the way SteerDuplex/PersonaPlex were trained: ``<system> ... <system>``."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ---------------------------------------------------------------------------
# Model-family routing
# ---------------------------------------------------------------------------
# PersonaPlex checkpoints cannot load through get_moshi_lm: their depformer
# self_attn in_projs are [6144, 1024] (dep_q=16) vs [3072, 1024] for moshiko.

FAMILIES = ("auto", "moshi", "personaplex")


def is_personaplex_family(family: str, hf_repo_id: str, model_type: str | None = None) -> bool:
    family = (family or "auto").strip().lower()
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if family != "auto":
        return family == "personaplex"
    if model_type == "personaplex":
        return True
    return "personaplex" in hf_repo_id.lower()


def resolve_checkpoint_safetensors(checkpoint_path: str) -> Path:
    """Resolve a checkpoint dir (or a direct file path) to its first ``*.safetensors``."""
    p = Path(checkpoint_path)
    if p.is_file():
        return p
    candidates = sorted(p.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors file found in {p}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Context window + lossless retry
# ---------------------------------------------------------------------------

def frames_to_sec(frames: int) -> float:
    return float(frames) / FRAME_RATE


def model_window_frames(model: object, *, fallback_frames: int = BASE_MOSHI_CONTEXT_FRAMES) -> int:
    """Native rolling attention window of a constructed model (``LMModel.context``)."""
    lm = getattr(model, "lm_model", None)
    frames = getattr(lm, "context", None) if lm is not None else None
    if frames is None:
        frames = getattr(model, "context", None)
    if frames is None:
        return int(fallback_frames)
    frames = int(frames)
    if frames <= 0:
        raise ValueError(f"resolved LM attention window is {frames} frames")
    return frames


def resolved_context_ceiling_sec(model: Any, context_ceiling_sec: float | None = None) -> float:
    """Retry ceiling in seconds: the model window, or ``context_ceiling_sec`` clamped to it."""
    frames = model_window_frames(model)
    window_sec = frames_to_sec(frames)
    if context_ceiling_sec is None:
        return window_sec
    requested = float(context_ceiling_sec)
    if requested > window_sec + 1e-9:
        logger.info(
            "context_ceiling_sec=%.1fs exceeds the model window %.1fs (%d frames); clamping.",
            requested, window_sec, frames,
        )
        return window_sec
    return requested


def context_bounded_timeline_sec(
    model: Any, *, requested_sec: float, context_ceiling_sec: float | None = None,
) -> float:
    frame_rate = float(model.mimi.frame_rate)
    if abs(frame_rate - FRAME_RATE) > 1e-9:
        raise ValueError(f"model.mimi.frame_rate={frame_rate} != {FRAME_RATE}")
    ceiling_sec = resolved_context_ceiling_sec(model, context_ceiling_sec)
    window_sec = frames_to_sec(model_window_frames(model))
    ceiling_sec = window_sec if ceiling_sec is None else min(float(ceiling_sec), window_sec)
    return min(float(requested_sec), ceiling_sec)


def recover_simple_generation_cap(
    *,
    model: "MoshiInference",
    user_audio_path: str,
    system_prompt: str,
    voice_prompt_path: str | None,
    first_output: Any,
    first_metadata: dict[str, Any],
    initial_max_duration_sec: float,
    retry_max_duration_sec: float,
    context_ceiling_sec: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Regenerate a safety-cap row once with a longer timeline (no reseed)."""
    metadata = dict(first_metadata)
    window_frames = model_window_frames(model)
    metadata.setdefault("resolved_window_frames", window_frames)
    metadata.setdefault("resolved_window_sec", frames_to_sec(window_frames))
    metadata.setdefault("context_window_policy", "rolling_last_window")
    if metadata.get("hit_safety_cap") is not True:
        return first_output, metadata
    if float(retry_max_duration_sec) <= float(initial_max_duration_sec) + 1e-6:
        metadata["lossless_retry_skipped"] = "retry_not_longer"
        return first_output, metadata

    effective_ceiling_sec = resolved_context_ceiling_sec(model, context_ceiling_sec)
    retry_sec = context_bounded_timeline_sec(
        model, requested_sec=retry_max_duration_sec, context_ceiling_sec=effective_ceiling_sec,
    )
    if retry_sec <= float(initial_max_duration_sec) + 1e-6:
        metadata["lossless_retry_skipped"] = "no_context_headroom"
        return first_output, metadata

    retry_output = model.generate(
        user_audio_path=user_audio_path,
        system_prompt=system_prompt,
        voice_prompt_path=voice_prompt_path,
        max_duration_sec=retry_sec,
    )
    retry_metadata = dict(model.last_generation_metadata or {})
    retry_metadata.update({
        "lossless_retry": True,
        "initial_stop_reason": metadata.get("stop_reason"),
        "initial_timeline_sec": metadata.get("timeline_sec"),
        "initial_max_duration_sec": float(initial_max_duration_sec),
        "retry_max_duration_sec": retry_sec,
        "context_ceiling_sec": float(effective_ceiling_sec),
        "context_ceiling_source": (
            "derived_from_model_window" if context_ceiling_sec is None else "explicit_freeze"
        ),
        "resolved_window_frames": window_frames,
        "resolved_window_sec": frames_to_sec(window_frames),
        "context_window_policy": "rolling_last_window",
    })
    return retry_output, retry_metadata


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class MoshiInference:
    """Offline batch-1 Moshi/PersonaPlex/SteerDuplex generator (PersonaPlex LMGen)."""

    def __init__(
        self,
        hf_repo_id: str = DEFAULT_HF_REPO,
        checkpoint_path: str | None = None,
        family: str = "auto",
        device: str = "cuda",
        seed: int | None = None,
        greedy: bool = False,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", 0)  # CUDA graphs need an explicit device index
        device = str(self.device)
        self.hf_repo_id = hf_repo_id
        self.family = family
        self.last_generation_metadata: dict[str, object] = {}

        # CUDA graphs are captured on the current device.
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            import random
            random.seed(seed)
            np.random.seed(seed)

        with torch.cuda.device(self.device):
            logger.info("Loading Moshi from %s", hf_repo_id)
            self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo=hf_repo_id)

            # Two Mimi instances keep codec streaming state isolated.
            self.mimi = self._load_mimi(str(device))
            self.mimi.eval()
            self.other_mimi = self._load_mimi(str(device))
            self.other_mimi.eval()
            self.sample_rate = self.mimi.sample_rate

            self.spm = self.checkpoint_info.get_text_tokenizer()

            self.lm_model = self._load_lm(checkpoint_path, str(device))
            self.lm_model.eval()
            logger.info("Model loaded (dep_q=%d, n_q=%d).", self.lm_model.dep_q, self.lm_model.n_q)

            self.lm_gen = LMGen(
                self.lm_model,
                device=device,
                use_sampling=not greedy,
                temp=temp,
                temp_text=temp_text,
                top_k=top_k,
                top_k_text=top_k_text,
                audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                sample_rate=self.sample_rate,
                frame_rate=self.mimi.frame_rate,
                save_voice_prompt_embeddings=False,
            )

            # Persistent streaming + warmup (CUDA graphs are captured here).
            self.mimi.streaming_forever(1)
            self.other_mimi.streaming_forever(1)
            self.lm_gen.streaming_forever(1)
            self._warmup()

    # -- loading ------------------------------------------------------------
    def _load_mimi(self, device: str):
        repo_lm_config = self.checkpoint_info.lm_config or {}
        if "dep_q" in repo_lm_config and "n_q" in repo_lm_config:
            num_codebooks = max(
                repo_lm_config["dep_q"],
                repo_lm_config["n_q"] - repo_lm_config["dep_q"],
            )
        else:
            num_codebooks = 8
        if self.checkpoint_info.tts_config.get("multistream"):
            num_codebooks //= 2
        return loaders.get_mimi(
            self.checkpoint_info.mimi_weights,
            self.checkpoint_info.mimi_config,
            num_codebooks=num_codebooks,
            device=device,
        )

    def _load_lm(self, checkpoint_path: str | None, device: str):
        repo_lm_config = self.checkpoint_info.lm_config or {}
        if "dim" in repo_lm_config and "n_q" in repo_lm_config:
            lm_config = dict(repo_lm_config)
        else:
            # Repos without a full LM config (moshiko, PersonaPlex) use the Moshi 7B shape.
            lm_config = dict(loaders._lm_kwargs)

        if checkpoint_path:
            config_path = Path(checkpoint_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    ckpt_config = json.load(f)
                if "dim" in ckpt_config and "n_q" in ckpt_config:
                    lm_config = dict(ckpt_config)
                elif ckpt_config.get("lora"):
                    lm_config["lora"] = True
                    lm_config["lora_rank"] = ckpt_config.get("lora_rank", 128)
                    lm_config["lora_scaling"] = ckpt_config.get("lora_scaling", 2.0)

        if is_personaplex_family(self.family, self.hf_repo_id, self.checkpoint_info.model_type):
            weights = self.checkpoint_info.moshi_weights
            if checkpoint_path:
                weights = resolve_checkpoint_safetensors(checkpoint_path)
            logger.info("Loading PersonaPlex LM (dep_q=16)")
            return loaders.get_moshi_lm_personaplex(weights, device=device)

        model = loaders.get_moshi_lm(
            self.checkpoint_info.moshi_weights,
            lm_kwargs=lm_config,
            device=device,
            lora_weights=self.checkpoint_info.lora_weights,
        )
        if self.checkpoint_info.model_type == "hibiki":
            model.text_emb.weight.data[2] = model.text_emb.weight.data[3]
        if checkpoint_path:
            self._load_checkpoint_weights(model, checkpoint_path)
        return model

    def _load_checkpoint_weights(self, model, checkpoint_path: str):
        ckpt_dir = Path(checkpoint_path)
        safetensor_files = list(ckpt_dir.glob("*.safetensors"))
        if safetensor_files:
            logger.info("Loading checkpoint weights %s", safetensor_files[0].name)
            # Load on CPU, then copy into the already-initialised GPU params.
            state_dict = load_file(str(safetensor_files[0]), device="cpu")
            # LoRA-wrapped layers keep their base weight under `.frozen_W.weight`.
            model_keys = set(model.state_dict().keys())
            renamed = 0
            for src_key in list(state_dict.keys()):
                if src_key in model_keys:
                    continue
                if src_key.endswith(".weight"):
                    candidate = src_key[: -len(".weight")] + ".frozen_W.weight"
                    if candidate in model_keys:
                        state_dict[candidate] = state_dict.pop(src_key)
                        renamed += 1
            if renamed:
                logger.info("Renamed %d keys -> .frozen_W.weight for LoRA-wrapped layers", renamed)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info("Missing keys: %d (samples: %s)", len(missing), missing[:3])
            if unexpected:
                logger.info("Unexpected keys: %d (samples: %s)", len(unexpected), unexpected[:3])
            del state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            pt_files = list(ckpt_dir.glob("*.pt"))
            if pt_files:
                state_dict = torch.load(str(pt_files[0]), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                del state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise FileNotFoundError(f"No *.safetensors or *.pt checkpoint found in {ckpt_dir}")

    def _warmup(self):
        frame_size = int(self.sample_rate / self.mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is None:
                    continue
                _ = self._decode_agent_codes(tokens[:, 1:1 + N_MIMI_CODEBOOKS])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    # -- decoding helpers ---------------------------------------------------
    def _decode_text_tokens(self, text_tokens: list[int]) -> list[str]:
        text_strs = []
        text_token_map = {0: "[EPAD]", 1: "[BOS]", 2: "[EOS]", 3: "[PAD]"}
        for tok in text_tokens:
            if tok in text_token_map:
                text_strs.append(text_token_map[tok])
            else:
                try:
                    text_strs.append(self.spm.id_to_piece(tok).replace("▁", " "))
                except Exception:
                    text_strs.append(f"[{tok}]")
        return text_strs

    def _safe_audio_codes_for_decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Replace non-codec special tokens with legal silence codes before Mimi decode."""
        card = int(self.lm_model.card)
        valid = (audio_codes >= 0) & (audio_codes < card)
        silence = self.lm_gen._encode_zero_frame().expand_as(audio_codes)
        return torch.where(valid, audio_codes, silence)

    def _decode_agent_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return self.mimi.decode(self._safe_audio_codes_for_decode(audio_codes))

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        user_audio_path: str,
        system_prompt: str = "",
        voice_prompt_path: str | None = None,
        max_duration_sec: float = 30.0,
    ) -> tuple[np.ndarray, int, list[str]]:
        """Return ``(audio, sample_rate, text_pieces)`` for one user turn."""
        if max_duration_sec is None:
            max_duration_sec = 30.0
        lm_gen = self.lm_gen
        self.last_generation_metadata = {}
        frame_size = int(self.sample_rate / self.mimi.frame_rate)

        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        lm_gen.reset_streaming()

        if system_prompt:
            lm_gen.text_prompt_tokens = self.spm.encode(wrap_with_system_tags(system_prompt))
        else:
            lm_gen.text_prompt_tokens = []

        if voice_prompt_path:
            p = Path(voice_prompt_path)
            if p.suffix == ".pt" and p.exists():
                lm_gen.load_voice_prompt_embeddings(str(p))
            elif p.exists():
                lm_gen.load_voice_prompt(str(p))
            else:
                raise FileNotFoundError(f"voice prompt not found: {voice_prompt_path}")
        else:
            lm_gen.voice_prompt_audio = None

        # Voice prompt -> silence -> system text -> silence.
        lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()

        user_audio, sr = sphn.read(user_audio_path)
        user_audio = sphn.resample(user_audio, src_sample_rate=sr, dst_sample_rate=self.sample_rate)
        if user_audio.ndim == 1:
            user_audio = user_audio[np.newaxis, :]
        if user_audio.shape[0] > 1:
            user_audio = user_audio[0:1]

        max_samples = int(max_duration_sec * self.sample_rate)
        original_user_samples = int(user_audio.shape[-1])
        input_truncated = original_user_samples > max_samples
        if user_audio.shape[-1] > max_samples:
            user_audio = user_audio[:, :max_samples]
            original_user_samples = max_samples
        elif user_audio.shape[-1] < max_samples:
            # Trailing silence gives the model the rest of the timeline to answer.
            silence = np.zeros(
                (user_audio.shape[0], max_samples - user_audio.shape[-1]),
                dtype=user_audio.dtype,
            )
            user_audio = np.concatenate([user_audio, silence], axis=-1)

        output_frames = []
        text_tokens = []
        SILENCE_TEXT_IDS = {0, 2, 3}  # [EPAD], [EOS], [PAD]
        silence_stop_frames = int(self.mimi.frame_rate * 5.0)
        user_audio_frames = int(math.ceil(original_user_samples / frame_size))
        audio_silence_rms_threshold = 0.002
        seen_text = False
        seen_audio_speech = False
        consecutive_text_silence = 0
        consecutive_audio_silence = 0
        frames_after_user_without_answer = 0
        early_stopped = False
        stop_reason = "max_duration"
        frame_idx = 0

        for user_encoded in encode_from_sphn(
            self.mimi,
            _iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
        ):
            for c in range(user_encoded.shape[-1]):
                frame_idx += 1
                tokens = lm_gen.step(user_encoded[:, :, c:c + 1])
                if tokens is None:
                    continue
                agent_codes = tokens[:, 1:1 + N_MIMI_CODEBOOKS]
                pcm = self._decode_agent_codes(agent_codes)
                pcm_np = pcm.detach().cpu().numpy()[0, 0]
                output_frames.append(pcm_np)
                rms = float(np.sqrt(np.mean(np.square(pcm_np)))) if pcm_np.size else 0.0
                if rms >= audio_silence_rms_threshold:
                    seen_audio_speech = True
                    consecutive_audio_silence = 0
                else:
                    consecutive_audio_silence += 1
                text_tok = tokens[0, 0, 0].item()
                text_tokens.append(text_tok)
                if text_tok in SILENCE_TEXT_IDS:
                    consecutive_text_silence += 1
                else:
                    seen_text = True
                    consecutive_text_silence = 0

                if frame_idx > user_audio_frames and not seen_text and not seen_audio_speech:
                    frames_after_user_without_answer += 1
                else:
                    frames_after_user_without_answer = 0

                if seen_text and consecutive_text_silence >= silence_stop_frames:
                    stop_reason = "text_silence"
                    early_stopped = True
                    break
                if seen_audio_speech and consecutive_audio_silence >= silence_stop_frames:
                    stop_reason = "audio_silence"
                    early_stopped = True
                    break
                if frames_after_user_without_answer >= silence_stop_frames:
                    stop_reason = "no_answer_silence"
                    early_stopped = True
                    break
            if early_stopped:
                break

        self.last_generation_metadata = {
            "mode": "simple",
            "stop_reason": stop_reason,
            "hit_safety_cap": stop_reason == "max_duration",
            "input_truncated": input_truncated,
            "input_frames": user_audio_frames,
            "generated_timeline_frames": len(output_frames),
            "max_duration_frames": int(math.ceil(max_duration_sec * self.mimi.frame_rate)),
            "generated_timeline_sec": len(output_frames) / float(self.mimi.frame_rate),
            "max_duration_sec": float(max_duration_sec),
        }

        if not output_frames:
            self.last_generation_metadata["stop_reason"] = "no_output"
            self.last_generation_metadata["hit_safety_cap"] = False
            logger.warning("No output frames generated")
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate, []

        output_audio = np.concatenate(output_frames, axis=-1)
        return output_audio.astype(np.float32), self.sample_rate, self._decode_text_tokens(text_tokens)


# ---------------------------------------------------------------------------
# SteerBenchModel adapter
# ---------------------------------------------------------------------------

def _public_checkpoint_name(checkpoint: str | None) -> str | None:
    """Last two path components only (e.g. ``checkpoint_001050/consolidated``); never an absolute path."""
    if not checkpoint:
        return None
    parts = Path(checkpoint).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


class MoshiFamilyModel(SteerBenchModel):
    """``--model moshi_family``: Moshi / PersonaPlex / SteerDuplex reference adapter.

    Keyword arguments (``--model_kwargs`` JSON):
        hf_repo: HF repo for Mimi, tokenizer and base LM weights
            (default ``kyutai/moshiko-pytorch-bf16``).
        checkpoint: Optional SteerDuplex ``consolidated/`` dir
            (``consolidated.safetensors`` + ``config.json``), or a PersonaPlex
            ``.safetensors`` file/dir when ``family="personaplex"``.
        family: ``auto`` (default; PersonaPlex if the repo says so), ``moshi``
            or ``personaplex``.
        device: torch device (default ``cuda``).
        greedy: Disable sampling (default ``False``; the reference runs sample).
        lossless_retry_max_duration: Timeline for the single cap-hit retry (300).
        context_ceiling_sec: Upper bound on that retry, clamped to the model's
            attention window (300; moshiko's 3000-frame window gives 240 s).
        label: Optional free-text name stored in ``run_config.json``.
    """

    name = "moshi_family"

    def __init__(
        self,
        hf_repo: str = DEFAULT_HF_REPO,
        checkpoint: str | None = None,
        family: str = "auto",
        device: str = "cuda",
        greedy: bool = False,
        lossless_retry_max_duration: float = 300.0,
        context_ceiling_sec: float | None = 300.0,
        label: str | None = None,
        init_seed: int | None = None,
    ):
        if lossless_retry_max_duration <= 0:
            raise ValueError("lossless_retry_max_duration must be positive")
        if checkpoint and not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.hf_repo = hf_repo
        self.checkpoint = checkpoint or None
        self.family = family
        self.greedy = bool(greedy)
        self.lossless_retry_max_duration = float(lossless_retry_max_duration)
        self.context_ceiling_sec = None if context_ceiling_sec is None else float(context_ceiling_sec)
        self.label = label
        self.engine = MoshiInference(
            hf_repo_id=hf_repo,
            checkpoint_path=self.checkpoint,
            family=family,
            device=device,
            seed=init_seed,
            greedy=self.greedy,
        )

    def respond(
        self,
        user_audio_path: str,
        system_prompt: str,
        voice_prompt_path: str | None,
        seed: int,
        max_duration_sec: float,
    ) -> ModelResponse:
        seed_global_rng(seed)
        engine = self.engine
        first = engine.generate(
            user_audio_path=user_audio_path,
            system_prompt=system_prompt,
            voice_prompt_path=voice_prompt_path,
            max_duration_sec=max_duration_sec,
        )
        generated, generation_metadata = recover_simple_generation_cap(
            model=engine,
            user_audio_path=user_audio_path,
            system_prompt=system_prompt,
            voice_prompt_path=voice_prompt_path,
            first_output=first,
            first_metadata=dict(engine.last_generation_metadata or {}),
            initial_max_duration_sec=max_duration_sec,
            retry_max_duration_sec=self.lossless_retry_max_duration,
            context_ceiling_sec=self.context_ceiling_sec,
        )
        out_audio, out_sr, out_text_tokens = generated
        text = "".join(t for t in out_text_tokens if not t.startswith("["))
        return ModelResponse(
            audio=out_audio, sample_rate=int(out_sr), text=text, metadata=generation_metadata,
        )

    def describe(self) -> dict[str, Any]:
        engine = self.engine
        return {
            "adapter": "moshi_family",
            "label": self.label,
            "hf_repo": self.hf_repo,
            "checkpoint": _public_checkpoint_name(self.checkpoint),
            "family": "personaplex" if is_personaplex_family(
                self.family, self.hf_repo, engine.checkpoint_info.model_type) else "moshi",
            "context_frames": model_window_frames(engine),
            "greedy": self.greedy,
            "sampling": {"temp": 0.8, "top_k": 250, "temp_text": 0.7, "top_k_text": 25},
            "lossless_retry_max_duration": self.lossless_retry_max_duration,
            "context_ceiling_sec": self.context_ceiling_sec,
            "batch_size": 1,
        }
