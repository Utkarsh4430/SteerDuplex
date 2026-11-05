"""Judge API client: one chat.completions call per vote, with retries and a tolerant JSON reader.

Uses the official ``openai`` SDK. The key comes from ``OPENAI_API_KEY`` and, for any OpenAI-compatible endpoint,
``OPENAI_BASE_URL`` -- both read from the environment by the SDK. Nothing else is sent besides the request.
"""
from __future__ import annotations

import json
import os
import re
import time

from openai import (AuthenticationError, BadRequestError, NotFoundError, OpenAI, PermissionDeniedError,
                    RateLimitError)

from .config import JudgeConfig

PERMANENT_ERRORS = (NotFoundError, BadRequestError, AuthenticationError, PermissionDeniedError)


class PermanentJudgeError(RuntimeError):
    """Non-retryable API error (unknown model, malformed request, bad key)."""


def make_client(cfg: JudgeConfig, timeout: float = 180.0) -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY (and OPENAI_BASE_URL for a non-OpenAI endpoint) before judging")
    return OpenAI(timeout=timeout, max_retries=2)


def _is_permanent(e: Exception) -> bool:
    if isinstance(e, RateLimitError):
        return False
    if isinstance(e, PERMANENT_ERRORS):
        return True
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429)


def _close_unterminated_json(s: str) -> str | None:
    """Append missing closing brackets to a reply that ends mid-structure outside a string literal.

    Some gateways return replies whose final closing brace is missing (finish_reason=stop). Returns None when the
    text ends inside a string or nothing is open, so no content is ever invented.
    """
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}" and stack and stack[-1] == ("[" if ch == "]" else "{"):
            stack.pop()
    if in_str or not stack:
        return None
    return s + "".join("]" if b == "[" else "}" for b in reversed(stack))


def loads_judge_json(content: str) -> dict:
    """Parse the FIRST JSON object in a reply (tolerates markdown fences and trailing text)."""
    s = (content or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        if start == -1:
            raise ValueError(f"judge reply contains no JSON object: {s[:200]!r}")
        try:
            obj, _ = json.JSONDecoder().raw_decode(s, start)
        except json.JSONDecodeError:
            closed = _close_unterminated_json(s[start:])
            if closed is None:
                raise
            obj, _ = json.JSONDecoder().raw_decode(closed, 0)
    if not isinstance(obj, dict):
        raise ValueError(f"judge reply's top-level JSON value is {type(obj).__name__}, expected object")
    return obj


def judge_call(client: OpenAI, cfg: JudgeConfig, messages: list[dict], *, max_retries: int = 8,
               label: str = "", validate=None) -> tuple[dict, dict]:
    """One vote. Returns (parsed_reply, call_record). Raises after ``max_retries`` failed attempts.

    Retries: transport errors, 429 and 5xx back off exponentially; malformed / truncated replies, and replies that
    fail ``validate`` (e.g. fewer verdicts than rubrics), are resampled immediately at the same settings. Permanent 4xx errors raise PermanentJudgeError at once.
    """
    kwargs = dict(model=cfg.model, messages=messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    if cfg.response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    if cfg.reasoning_effort is not None:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    last_err: Exception | None = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content
            finish = getattr(choice, "finish_reason", None)
            if not content:
                raise ValueError("empty reply")
            if finish == "length":
                raise ValueError(f"reply truncated (finish_reason=length, {len(content)} chars)")
            parsed = loads_judge_json(content)
            if validate is not None:
                validate(parsed)
            usage = getattr(resp, "usage", None)
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            rec = {
                "response_model": getattr(resp, "model", None),
                "finish_reason": finish,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
                "latency_s": round(time.time() - t0, 3),
                "attempts": attempt + 1,
                "reply_chars": len(content),
                "reply_hash": hash_text(content),
            }
            return parsed, rec
        except Exception as e:  # noqa: BLE001
            if _is_permanent(e):
                raise PermanentJudgeError(f"{label} permanent API error (model={cfg.model}): {e!r}") from e
            last_err = e
            if attempt == max_retries - 1:
                break
            if isinstance(e, (json.JSONDecodeError, ValueError)):
                continue
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"{label} judge failed after {max_retries} attempts: {last_err!r}")


def hash_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]
