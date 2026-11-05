"""Template: plug your own model into SteerBench.

Copy this file, replace ``my_generate`` with a call to your model, and run (from the repo root)::

    python -m inference.run_inference --model inference/examples/custom_model.py:CallableModel \\
        --out_dir runs/my_model

The two runnable baselines below are useful for smoke-testing the pipeline end to end::

    # echoes the user turn back (no text stream, so Whisper ASR fills in the transcript)
    python -m inference.run_inference --model inference/examples/custom_model.py:EchoModel \\
        --out_dir runs/echo --limit 3

    # one second of silence with an empty transcript (a floor for the judge)
    python -m inference.run_inference --model inference/examples/custom_model.py:SilenceModel \\
        --out_dir runs/silence --limit 3 --asr never
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from inference.interface import ModelResponse, SteerBenchModel
except ImportError:  # loaded by file path outside the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from inference.interface import ModelResponse, SteerBenchModel


def my_generate(user_audio: np.ndarray, sample_rate: int, system_prompt: str,
                voice_prompt_path: str | None, seed: int, max_duration_sec: float):
    """Replace me. Return ``(audio_float32_mono, sample_rate, text_or_None)``.

    Default behaviour: echo the user turn (a trivial non-steerable baseline).
    """
    return user_audio, sample_rate, None


class CallableModel(SteerBenchModel):
    """Adapter around a plain function with ``my_generate``'s signature."""

    name = "callable"

    def __init__(self, fn=my_generate, label: str = "my_generate"):
        self.fn = fn
        self.label = label

    def respond(self, user_audio_path, system_prompt, voice_prompt_path, seed, max_duration_sec):
        audio, sr = sf.read(user_audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out_audio, out_sr, text = self.fn(audio, sr, system_prompt, voice_prompt_path, seed, max_duration_sec)
        # Respect the budget: trim the reply to max_duration_sec.
        out_audio = np.asarray(out_audio, dtype=np.float32)[: int(max_duration_sec * out_sr)]
        return ModelResponse(audio=out_audio, sample_rate=out_sr, text=text,
                             metadata={"stop_reason": "callable"})

    def describe(self):
        return {"adapter": "callable", "label": self.label}


class EchoModel(CallableModel):
    """Repeats the user turn. Has no text stream, so ``--asr auto`` transcribes it."""

    name = "echo"

    def __init__(self):
        super().__init__(fn=my_generate, label="echo")


class SilenceModel(SteerBenchModel):
    """One second of silence and an empty transcript."""

    name = "silence"

    def respond(self, user_audio_path, system_prompt, voice_prompt_path, seed, max_duration_sec):
        return ModelResponse(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000, text="",
                             metadata={"stop_reason": "silence_baseline"})
