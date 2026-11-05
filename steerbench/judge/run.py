"""Judge one or more SteerBench runs in ONE interleaved session.

    python -m judge.run --session_dir judgments/my_session --model gpt-4o \
        --run mymodel=runs/mymodel_s292500 --run mymodel=runs/mymodel_s292501

Every (run, item, rubric-kind, vote) call is shuffled into one job queue and executed by one worker pool, so all
compared systems see the same judge backend state (the "session"). Judging systems in separate sessions and
comparing the numbers is not decision-grade: between-session drift (~3-4 pp SD per read) exceeds most model gaps.

A run dir holds ``<item_id>.wav`` (assistant turn, any sample rate, mono) and ``<item_id>.json`` with ``text`` (the
model's own text stream, or an ASR transcript) -- the format ``inference/run_inference.py`` writes.

Outputs (resumable; re-running the same command continues where it stopped):
  <session_dir>/calls.jsonl                       one line per judge call (verdicts + usage, no credentials)
  <session_dir>/<label>/evaluation_results.json   per-item verdicts after the J-vote majority, one per run
  <session_dir>/session.json                      pinned judge config, dates, call statistics, run list
Then aggregate with ``python score.py --session_dir <session_dir>``.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import random
import statistics
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import JudgeConfig
from .client import PermanentJudgeError, judge_call, make_client
from .prompts import (AUDIO_JUDGE_PROMPT, EMPTY_TRANSCRIPT_PLACEHOLDER, REFERENCE_CLIP_LABEL, RESPONSE_CLIP_LABEL,
                      TEXT_JUDGE_PROMPT, format_rubrics)

ROOT = Path(__file__).resolve().parents[1]
JUDGE_PROTOCOL_VERSION = "steerbench-judge/2"
RESULTS_JSON = "evaluation_results.json"
INVALID_ERROR_RATE = 0.05


def response_text(sidecar: dict) -> str:
    return (sidecar.get("text") if sidecar.get("text") is not None else sidecar.get("moshi_text") or "").strip()


def build_messages(kind: str, item: dict, text: str, ref_wav: Path, resp_wav: Path) -> list[dict]:
    if kind == "text":
        prompt = TEXT_JUDGE_PROMPT.format(user_utterance=item["user_utterance"],
                                          moshi_text=text or EMPTY_TRANSCRIPT_PLACEHOLDER,
                                          text_rubrics_block=format_rubrics(item["text_rubrics"]))
        return [{"role": "user", "content": prompt}]
    prompt = AUDIO_JUDGE_PROMPT.format(audio_rubrics_block=format_rubrics(item["audio_rubrics"]))
    b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "text", "text": REFERENCE_CLIP_LABEL},
        {"type": "input_audio", "input_audio": {"data": b64(ref_wav), "format": "wav"}},
        {"type": "text", "text": RESPONSE_CLIP_LABEL},
        {"type": "input_audio", "input_audio": {"data": b64(resp_wav), "format": "wav"}},
    ]}]


def _validator(n_rubrics: int, key: str):
    def check(parsed: dict) -> None:
        res = parsed.get(key)
        if not isinstance(res, list) or len(res) < n_rubrics:
            raise ValueError(f"reply has {len(res) if isinstance(res, list) else 0} verdicts for {n_rubrics} rubrics")
    return check


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_runs(specs: list[str]) -> list[tuple[str, str, Path]]:
    """--run LABEL=DIR (repeat LABEL for K decode seeds). Returns [(system, run_label, dir)]."""
    out, seen = [], Counter()
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--run expects SYSTEM=DIR, got {spec!r}")
        system, d = spec.split("=", 1)
        seen[system] += 1
        out.append((system, f"{system}__r{seen[system]}", Path(d)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="SYSTEM=RUN_DIR; repeat the system name for K seeds")
    ap.add_argument("--session_dir", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--model", required=True, help="judge model, e.g. gpt-4o (any OpenAI-compatible chat model)")
    ap.add_argument("--reasoning_effort", default=None, help="for reasoning models: low/medium/high (omit otherwise)")
    ap.add_argument("--votes", type=int, default=None, help="J votes per rubric list (protocol: 3)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke tests)")
    ap.add_argument("--ids", nargs="+", default=None)
    args = ap.parse_args(argv)

    cfg = JudgeConfig(model=args.model, reasoning_effort=args.reasoning_effort, votes=args.votes or 3)
    data = Path(args.data_dir)
    bench_path = data / "steerbench.json"
    items = json.loads(bench_path.read_text())["items"]
    if args.ids:
        items = [it for it in items if it["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]
    by_id = {it["id"]: it for it in items}
    ref_dir = data / "reference_audio"
    missing_refs = [sid for sid in by_id if not (ref_dir / f"{sid}.wav").is_file()]
    if missing_refs:
        raise SystemExit(f"{len(missing_refs)} reference WAVs missing under {ref_dir}; run: python scripts/fetch_dataset.py")

    sess = Path(args.session_dir)
    sess.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(args.run)

    # Build the job list: (run_label, sid, kind, vote).
    sidecars: dict[tuple[str, str], dict] = {}
    jobs, missing = [], defaultdict(list)
    for system, label, d in runs:
        for sid, it in by_id.items():
            wav, js = d / f"{sid}.wav", d / f"{sid}.json"
            if not (wav.is_file() and js.is_file()):
                missing[label].append(sid)
                continue
            sidecars[(label, sid)] = json.loads(js.read_text())
            for kind in ("text", "audio"):
                if it.get(f"{kind}_rubrics"):
                    jobs += [(label, sid, kind, v) for v in range(cfg.votes)]
    calls_path = sess / "calls.jsonl"
    done = set()
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                done.add((r["run"], r["sample_id"], r["kind"], r["vote"]))
    pending = [j for j in jobs if j not in done]
    random.Random(args.shuffle_seed).shuffle(pending)     # interleave systems, items and votes
    run_dirs = {label: d for _, label, d in runs}
    print(f"[judge] judge={cfg.model} reasoning_effort={cfg.reasoning_effort} votes={cfg.votes} "
          f"| runs={len(runs)} items={len(by_id)} calls={len(jobs)} pending={len(pending)}", flush=True)
    for label, sids in missing.items():
        print(f"[judge] WARNING {label}: {len(sids)} items have no wav+json (counted as missing)", flush=True)

    receipt_path = sess / "session.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt.setdefault("started_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    receipt.update({
        "protocol": JUDGE_PROTOCOL_VERSION,
        "judge": cfg.as_record(),
        "bench_sha256": sha256_file(bench_path),
        "n_items": len(by_id),
        "runs": [{"system": s, "label": l, "dir_name": d.name} for s, l, d in runs],
        "shuffle_seed": args.shuffle_seed,
    })
    receipt_path.write_text(json.dumps(receipt, indent=1))

    lock = threading.Lock()
    if pending:
        client = make_client(cfg, timeout=args.timeout)

        def work(job):
            label, sid, kind, vote = job
            it = by_id[sid]
            key = f"{kind}_results"
            msgs = build_messages(kind, it, response_text(sidecars[(label, sid)]), ref_dir / f"{sid}.wav",
                                  run_dirs[label] / f"{sid}.wav")
            parsed, rec = judge_call(client, cfg, msgs, max_retries=args.max_retries, label=f"{label}/{sid}/{kind}",
                                     validate=_validator(len(it[f"{kind}_rubrics"]), key))
            return {"run": label, "sample_id": sid, "kind": kind, "vote": vote, "ok": True,
                    "verdicts": [bool(r.get("pass", False)) for r in parsed[key]][: len(it[f"{kind}_rubrics"])],
                    "explanations": [str(r.get("explanation", "")) for r in parsed[key]][: len(it[f"{kind}_rubrics"])],
                    "summary": parsed.get("style_summary" if kind == "audio" else "response_summary", ""),
                    "call": rec, "t_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

        n_ok = n_err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool, calls_path.open("a") as out:
            futs = {pool.submit(work, j): j for j in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                label, sid, kind, vote = futs[fut]
                try:
                    rec = fut.result()
                    n_ok += 1
                except PermanentJudgeError as e:
                    print(f"[judge] FATAL {e}", flush=True)
                    pool.shutdown(cancel_futures=True)
                    return 2
                except Exception as e:  # noqa: BLE001
                    rec = {"run": label, "sample_id": sid, "kind": kind, "vote": vote, "ok": False, "error": repr(e)[:500]}
                    n_err += 1
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                if i % 200 == 0 or i == len(pending):
                    print(f"[judge] {i}/{len(pending)} ok={n_ok} failed={n_err}", flush=True)

    # Assemble per-run results from the call log (latest ok record per key wins).
    calls = {}
    failures = defaultdict(list)
    all_recs = []
    for line in calls_path.read_text().splitlines():
        r = json.loads(line)
        k = (r["run"], r["sample_id"], r["kind"], r["vote"])
        if r.get("ok"):
            calls[k] = r
            all_recs.append(r)
        else:
            failures[k].append(r.get("error"))
    stats = _call_stats(all_recs, cfg.votes)
    for system, label, d in runs:
        results = []
        for sid, it in by_id.items():
            if (label, sid) not in sidecars:
                continue
            sc = sidecars[(label, sid)]
            row = {"sample_id": sid, "split": it["split"], "sub_axis": it["sub_axis"],
                   "sub_axis_group": it.get("sub_axis_group", ""), "user_utterance": it["user_utterance"],
                   "response_text": response_text(sc), "text_source": sc.get("text_source", "model_text_stream"),
                   "audio_duration_sec": sc.get("audio_duration_sec"),
                   "response_summary": "", "style_summary": "", "audio_results": [], "text_results": [], "errors": {}}
            for kind in ("text", "audio"):
                rubrics = it.get(f"{kind}_rubrics") or []
                if not rubrics:
                    continue
                votes = [calls.get((label, sid, kind, v)) for v in range(cfg.votes)]
                if any(v is None for v in votes):
                    errs = [e for v in range(cfg.votes) for e in failures.get((label, sid, kind, v), [])]
                    row["errors"][kind] = f"{sum(v is None for v in votes)}/{cfg.votes} votes missing" + (
                        f"; last error: {errs[-1]}" if errs else "")
                    row[f"{kind}_results"] = [{"item": r["item"], "axis": r["axis"], "pass": False,
                                               "explanation": "JUDGE ERROR"} for r in rubrics]
                    continue
                merged = []
                for i, r in enumerate(rubrics):
                    vs = [v["verdicts"][i] for v in votes]
                    maj = sum(vs) > len(vs) / 2.0             # strict majority; ties -> fail
                    expl = next((v["explanations"][i] for v in votes if v["verdicts"][i] == maj), "")
                    merged.append({"item": r["item"], "axis": r["axis"], "pass": maj, "votes": vs, "explanation": expl})
                row[f"{kind}_results"] = merged
                row["style_summary" if kind == "audio" else "response_summary"] = votes[0]["summary"]
            results.append(row)
        out_dir = sess / label
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / RESULTS_JSON).write_text(json.dumps(results, indent=1, ensure_ascii=False))
        n_err_items = sum(1 for r in results if r["errors"])
        rate = n_err_items / len(results) if results else 0.0
        flag = "  INVALID (>5% judge errors)" if rate > INVALID_ERROR_RATE else ""
        print(f"[judge] {label}: {len(results)} items, {n_err_items} with judge errors ({100 * rate:.1f}%), "
              f"{len(missing.get(label, []))} missing{flag}", flush=True)

    receipt.update({"finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "call_stats": stats})
    if stats.get("repeat_reply_identity", 0) > 0.5:
        receipt["warning"] = ("Most J-vote replies are byte-identical: the judge looks deterministic (thinking likely off). "
                              "This is NOT the reference regime; do not compare with reference numbers.")
        print("[judge] WARNING " + receipt["warning"], flush=True)
    receipt_path.write_text(json.dumps(receipt, indent=1))
    return 0


def _call_stats(recs: list[dict], votes: int) -> dict:
    if not recs:
        return {}
    med = lambda xs: (statistics.median(xs) if xs else None)
    calls = [r["call"] for r in recs]
    rt = [c["reasoning_tokens"] for c in calls if c.get("reasoning_tokens") is not None]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["run"], r["sample_id"], r["kind"])].append(r["call"].get("reply_hash"))
    multi = [g for g in groups.values() if len(g) >= 2]
    first = min(r["t_utc"] for r in recs)
    last = max(r["t_utc"] for r in recs)
    return {
        "n_ok_calls": len(recs),
        "first_call_utc": first, "last_call_utc": last,
        "response_models": dict(Counter(c.get("response_model") for c in calls)),
        "finish_reasons": dict(Counter(c.get("finish_reason") for c in calls)),
        "median_completion_tokens": med([c["completion_tokens"] for c in calls if c.get("completion_tokens") is not None]),
        "median_reasoning_tokens": med(rt) if rt else "not reported by endpoint",
        "median_latency_s": med([c["latency_s"] for c in calls]),
        "prompt_tokens_total": sum(c.get("prompt_tokens") or 0 for c in calls),
        "completion_tokens_total": sum(c.get("completion_tokens") or 0 for c in calls),
        "mean_attempts": round(sum(c.get("attempts", 1) for c in calls) / len(calls), 3),
        # Share of (run, item, kind) groups whose J replies are byte-identical. A thinking judge at temperature 0 is
        # still sampled, so this should be near 0; values near 1 indicate a deterministic (non-thinking) backend.
        "repeat_reply_identity": round(sum(1 for g in multi if len(set(g)) == 1) / len(multi), 4) if multi else None,
    }


if __name__ == "__main__":
    sys.exit(main())
