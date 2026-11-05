"""SteerBench judge configuration.

SteerBench scores depend on the judge configuration more than on most model differences of interest (turning a
reasoning model's thinking off, or judging on different days, moves rubric pass rates by several points). So the
judge config is pinned here, recorded with every output, and all systems in a comparison must be judged together
in ONE interleaved session (see the README "Protocol").

The judge is any OpenAI or OpenAI-compatible chat model. The client uses the official ``openai`` SDK, which reads
``OPENAI_API_KEY`` (and, optionally, ``OPENAI_BASE_URL``) from the environment -- so you add your own key and run.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class JudgeConfig:
    model: str                         # e.g. "gpt-4o"; any OpenAI-compatible chat model
    reasoning_effort: str | None = None  # for reasoning models ("low"/"medium"/"high"); None to omit the field
    temperature: float = 0.0
    max_tokens: int = 4096
    response_format_json: bool = True
    votes: int = 3                     # per-rubric majority over J independent calls (protocol: 3)

    def as_record(self) -> dict:
        """Judge settings to store in the session receipt (no credentials are ever included)."""
        return dataclasses.asdict(self)
