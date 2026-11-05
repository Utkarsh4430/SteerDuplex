"""Whisper ASR for models that have no text stream of their own.

The text rubrics score a response transcript. Moshi, PersonaPlex and
SteerDuplex all produce text alongside their audio (their own text stream),
and the published reference numbers score that text. A model that only outputs
audio needs a transcript, so the driver transcribes the audio with Whisper
(``--asr auto``, the default, when ``respond()`` returns ``text=None``).

Caveat: text-rubric scores from ASR transcripts are NOT directly comparable
to scores from a model's own text stream. ASR adds recognition errors and
normalises disfluencies, fillers and casing. It also drops anything that was
not clearly spoken, and keeps speech the model produced without a matching
text token. Report which source you used: every sidecar records it in
``text_source`` (``"model_text_stream"`` or ``"asr:<model id>"``). For a
like-for-like comparison with an audio-only system, rerun the reference
models with ``--asr always`` so both sides are scored on ASR transcripts.
Audio-rubric scores are unaffected because the judge listens to the WAV.

Requires ``transformers`` (and ``torch``); the default model is
``openai/whisper-large-v3``. Decoding is greedy with English forced, and audio
longer than 30 s is transcribed with Whisper's sequential long-form mode.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


class WhisperASR:
    """Thin wrapper around the Hugging Face ``automatic-speech-recognition`` pipeline."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str | None = None,
        language: str | None = "en",
        torch_dtype: str | None = None,
    ):
        import torch
        from transformers import pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)
        self.model_id = model_id
        self.language = language
        logger.info("Loading ASR model %s on %s", model_id, device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=device,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float waveform; returns stripped text ("" for silence)."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-6:
            return ""
        if sample_rate != WHISPER_SAMPLE_RATE:
            import sphn
            audio = sphn.resample(audio, src_sample_rate=sample_rate, dst_sample_rate=WHISPER_SAMPLE_RATE)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        generate_kwargs = {"task": "transcribe"}
        if self.language:
            generate_kwargs["language"] = self.language
        long_form = audio.shape[-1] > 30 * WHISPER_SAMPLE_RATE
        result = self.pipe(
            {"raw": audio, "sampling_rate": WHISPER_SAMPLE_RATE},
            generate_kwargs=generate_kwargs,
            return_timestamps=True if long_form else False,
        )
        return str(result.get("text", "")).strip()
