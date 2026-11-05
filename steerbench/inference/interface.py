"""Model-agnostic interface for generating SteerBench responses.

To benchmark a model, implement :class:`SteerBenchModel` (one method,
:meth:`SteerBenchModel.respond`) and pass it to the driver::

    python -m inference.run_inference --model my_pkg.my_module:MyModel --out_dir runs/my_model

SteerBench is **single-turn**. Each item is one spoken user turn (a 24 kHz
mono WAV) that asks for a particular speaking style, persona, tone, pace or
length. Your model listens to the user turn and produces one assistant turn.
The judge scores that turn's audio and its text.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ModelResponse",
    "SteerBenchModel",
    "ITEM_SEED_DERIVATION",
    "derive_item_seed",
    "seed_global_rng",
]

#: Human-readable name of the per-item seed rule (stored in every sidecar / run_config).
ITEM_SEED_DERIVATION = "sha256:steerbench-item-seed-v1/base_seed/sample_id:63bit"


@dataclass
class ModelResponse:
    """One assistant turn.

    Attributes:
        audio: The assistant turn ONLY, as a 1-D ``float32`` mono waveform in
            [-1, 1]. Do not include the user turn, a system/voice prompt or a
            stereo mix. Leading silence before the model starts speaking may
            be kept; the judge listens to the whole clip.
        sample_rate: Sample rate of ``audio`` in Hz (any rate; WAVs are
            written as-is, 16-bit PCM).
        text: The model's OWN text stream for this turn (e.g. Moshi's inner
            monologue, or the text a cascaded system sent to its TTS),
            without control tokens. Use ``None`` if the model has no text
            stream. With ``--asr auto`` (the default), ``None`` makes the
            driver transcribe ``audio`` with Whisper; see ``inference/asr.py``
            for why those scores are not directly comparable.
        metadata: Free-form JSON-serialisable diagnostics (stop reason,
            number of frames, ...). Stored as ``generation_metadata`` in the
            per-item sidecar. Do not put absolute paths or secrets here.
    """

    audio: np.ndarray
    sample_rate: int
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio)
        if audio.ndim == 2 and 1 in audio.shape:
            audio = audio.reshape(-1)
        if audio.ndim != 1:
            raise ValueError(f"ModelResponse.audio must be mono 1-D, got shape {audio.shape}")
        self.audio = audio.astype(np.float32, copy=False)
        self.sample_rate = int(self.sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("ModelResponse.sample_rate must be positive")


class SteerBenchModel(abc.ABC):
    """Base class for a model under test.

    The driver builds one instance and calls :meth:`respond` once per item,
    one item at a time (batch size 1), in benchmark order. Constructor
    keyword arguments come from ``--model_kwargs`` (a JSON object).
    """

    #: Short identifier stored in ``run_config.json``.
    name: str = "custom"

    @abc.abstractmethod
    def respond(
        self,
        user_audio_path: str,
        system_prompt: str,
        voice_prompt_path: str | None,
        seed: int,
        max_duration_sec: float,
    ) -> ModelResponse:
        """Generate the assistant's reply to one spoken user turn.

        Args:
            user_audio_path: Path to the user turn (24 kHz mono 16-bit WAV).
                Feed the whole turn to the model as the user's speech, then
                let the model respond as it would in a live conversation.
            system_prompt: The examinee system prompt (``data/system_prompt.txt``
                by default). Give it to the model the way it normally takes
                a system or persona prompt. Models without one may ignore it,
                but should say so in their write-up.
            voice_prompt_path: Optional WAV giving the assistant's voice
                (``data/voice_prompt/libri_hq24k_1226.wav`` by default), or
                ``None``. Models without voice conditioning may ignore it.
            seed: Item seed derived from ``--seed`` and the item id (see
                :func:`derive_item_seed`). The driver has already called
                :func:`seed_global_rng` with it; seed any other RNG you
                use, e.g. a sampling ``generator`` or a remote API ``seed``.
            max_duration_sec: Generation budget in seconds. Duplex models
                count it as the conversation timeline: the user turn plus
                the silence that follows it. Turn-based models can treat it
                as the maximum reply length.

        Returns:
            A :class:`ModelResponse` holding only the assistant turn.
        """

    def describe(self) -> dict[str, Any]:
        """Optional JSON-serialisable description stored in ``run_config.json``.

        Report model id, checkpoint name, decoding settings and so on. Do not
        report absolute local paths or credentials.
        """
        return {}


def derive_item_seed(base_seed: int, sample_id: str) -> int:
    """Return a stable per-item seed independent of shard and worker layout.

    ``sha256(b"steerbench-item-seed-v1\\0" + str(base_seed) + b"\\0" + sample_id)``,
    first 8 bytes big-endian, masked to 63 bits. This matches the rule used for
    the published reference runs (base seeds 292500 and 292501).
    """
    if base_seed < 0:
        raise ValueError("item seed mode requires a non-negative base seed")
    if not sample_id:
        raise ValueError("item seed mode requires a non-empty sample_id")
    payload = (
        b"steerbench-item-seed-v1\0"
        + str(base_seed).encode("ascii")
        + b"\0"
        + sample_id.encode("utf-8")
    )
    # Keep the value in torch.manual_seed's portable non-negative int64 range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def seed_global_rng(seed: int) -> None:
    """Reset Python's, NumPy's and (if installed) torch's global RNGs to ``seed``."""
    import random

    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    try:
        import torch
    except ImportError:  # torch is optional for custom (e.g. API-based) models
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
