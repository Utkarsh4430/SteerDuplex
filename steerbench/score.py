#!/usr/bin/env python3
"""Aggregate a SteerBench judging session into per-system scores with item-bootstrap CIs.

    python score.py --session_dir judgments/my_session --compare mymodel other_system

Metrics (all over items without judge errors; K decode reads of a system are pooled):
  audio_rubric_rate  share of individual audio_steering rubrics passed            (per-rubric "headline")
  audio_apr          share of items whose audio rubrics ALL pass ("audio-steering APR", the paper's metric)
  sample_apr         share of items whose audio AND text rubrics all pass
  text_content       share of text_content rubrics passed
  text_constraint    share of text_constraint rubrics passed
  empty_responses    items whose response text has fewer than 3 words (mean over reads, and %)
  judge_errors       items with a judge error (excluded from every denominator; >5% of a read => INVALID)

CIs are 95% item-bootstrap intervals (items resampled, the K reads of an item kept together). They treat the K decodes
as fixed, so they under-cover decode variance; report K >= 2 and look at the per-read values as well.
Per-sub-axis numbers have n <= 4 items and are not reported.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ["A3_tone_controlled", "A4_persona_controlled", "A5_style_accent", "A6_speed_length"]
SPLIT_SHORT = {"A3_tone_controlled": "A3 tone", "A4_persona_controlled": "A4 persona",
               "A5_style_accent": "A5 style/accent", "A6_speed_length": "A6 speed/length"}
INVALID_ERROR_RATE = 0.05
EMPTY_WORDS = 3
# numerator/denominator columns accumulated per item (summed over the K reads of a system)
COLS = ["a_pass", "a_tot", "apr_pass", "apr_n", "all_pass", "all_n", "tc_pass", "tc_tot", "tk_pass", "tk_tot"]
METRICS = {"audio_rubric_rate": ("a_pass", "a_tot"), "audio_apr": ("apr_pass", "apr_n"),
           "sample_apr": ("all_pass", "all_n"), "text_content": ("tc_pass", "tc_tot"),
           "text_constraint": ("tk_pass", "tk_tot")}




def item_row(r: dict, excluded: set) -> dict | None:
    """Per-item numerators/denominators for one read; None when the item had a judge error."""
    if r.get("errors"):
        return None
    sid = r["sample_id"]
    a = [x["pass"] for i, x in enumerate(r.get("audio_results", [])) if (sid, "audio", i) not in excluded]
    t = [(x["axis"], x["pass"]) for i, x in enumerate(r.get("text_results", [])) if (sid, "text", i) not in excluded]
    row = dict.fromkeys(COLS, 0)
    row["a_pass"], row["a_tot"] = sum(a), len(a)
    if a:
        row["apr_pass"], row["apr_n"] = int(all(a)), 1
    if a or t:
        row["all_pass"], row["all_n"] = int(all(a) and all(p for _, p in t)), 1
    for axis, p in t:
        k = "tc" if axis == "text_content" else "tk"
        row[f"{k}_pass"] += int(p)
        row[f"{k}_tot"] += 1
    return row


def ratio(m: np.ndarray, num: int, den: int) -> float | None:
    d = m[:, den].sum()
    return float(m[:, num].sum() / d) if d else None


def boot_ci(m: np.ndarray, num: int, den: int, B: int, rng: np.random.Generator) -> tuple[float, float] | None:
    if m.shape[0] == 0 or m[:, den].sum() == 0:
        return None
    idx = rng.integers(0, m.shape[0], size=(B, m.shape[0]))
    nums, dens = m[idx, num].sum(1), m[idx, den].sum(1)
    vals = nums[dens > 0] / dens[dens > 0]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_session(session_dir: Path) -> tuple[dict, dict[str, list[tuple[str, list[dict]]]]]:
    receipt = json.loads((session_dir / "session.json").read_text())
    systems: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for r in receipt["runs"]:
        p = session_dir / r["label"] / "evaluation_results.json"
        if p.is_file():
            systems[r["system"]].append((r["label"], json.loads(p.read_text())))
        elif p.with_suffix(".json.gz").is_file():
            with gzip.open(p.with_suffix(".json.gz"), "rt", encoding="utf-8") as f:
                systems[r["system"]].append((r["label"], json.load(f)))
    return receipt, systems


def score_system(reads: list[tuple[str, list[dict]]], excluded: set, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    per_item: dict[str, np.ndarray] = {}
    split_of: dict[str, str] = {}
    per_read = []
    for label, results in reads:
        n_err = sum(1 for r in results if r.get("errors"))
        n_empty = sum(1 for r in results if len((r.get("response_text") or "").split()) < EMPTY_WORDS)
        acc = np.zeros(len(COLS))
        for r in results:
            split_of[r["sample_id"]] = r["split"]
            row = item_row(r, excluded)
            if row is None:
                continue
            v = np.array([row[c] for c in COLS], dtype=float)
            acc += v
            per_item[r["sample_id"]] = per_item.get(r["sample_id"], np.zeros(len(COLS))) + v
        one = {k: (acc[COLS.index(n)] / acc[COLS.index(d)] if acc[COLS.index(d)] else None) for k, (n, d) in METRICS.items()}
        rate = n_err / len(results) if results else 0.0
        per_read.append({"label": label, "n_items": len(results), "judge_errors": n_err, "judge_error_rate": rate,
                         "invalid": rate > INVALID_ERROR_RATE, "empty_responses": n_empty, **one})
    sids = sorted(per_item)
    m = np.stack([per_item[s] for s in sids]) if sids else np.zeros((0, len(COLS)))
    out = {"k_reads": len(reads), "n_items_scored": len(sids), "per_read": per_read,
           "judge_errors": sum(p["judge_errors"] for p in per_read),
           "any_invalid_read": any(p["invalid"] for p in per_read),
           "empty_responses_mean": float(np.mean([p["empty_responses"] for p in per_read])) if per_read else None,
           "empty_response_rate": (float(np.mean([p["empty_responses"] / p["n_items"] for p in per_read]))
                                   if per_read else None)}
    for k, (n, d) in METRICS.items():
        out[k] = {"value": ratio(m, COLS.index(n), COLS.index(d)), "ci95": boot_ci(m, COLS.index(n), COLS.index(d), B, rng),
                  "denominator": float(m[:, COLS.index(d)].sum())}
    out["per_split"] = {}
    for sp in SPLITS:
        ms = m[[i for i, s in enumerate(sids) if split_of[s] == sp]] if sids else m
        out["per_split"][sp] = {k: {"value": ratio(ms, COLS.index(METRICS[k][0]), COLS.index(METRICS[k][1])),
                                    "ci95": boot_ci(ms, COLS.index(METRICS[k][0]), COLS.index(METRICS[k][1]), B, rng)}
                                for k in ("audio_rubric_rate", "audio_apr")}
    out["_matrix"] = (sids, m)
    return out


def paired_diff(sa: dict, sb: dict, metric: str, B: int, seed: int) -> dict:
    ids_a, ma = sa["_matrix"]
    ids_b, mb = sb["_matrix"]
    common = sorted(set(ids_a) & set(ids_b))
    ia = {s: i for i, s in enumerate(ids_a)}
    ib = {s: i for i, s in enumerate(ids_b)}
    A, Bm = ma[[ia[s] for s in common]], mb[[ib[s] for s in common]]
    n, d = COLS.index(METRICS[metric][0]), COLS.index(METRICS[metric][1])
    point = A[:, n].sum() / A[:, d].sum() - Bm[:, n].sum() / Bm[:, d].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    diffs = A[idx, n].sum(1) / A[idx, d].sum(1) - Bm[idx, n].sum(1) / Bm[idx, d].sum(1)
    return {"metric": metric, "n_common_items": len(common), "diff": float(point),
            "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def ci(c) -> str:
    return "" if not c else f" [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def render(receipt: dict, variant: str, scores: dict, comparisons: list[dict]) -> str:
    j = receipt.get("judge", {})
    lines = [f"## SteerBench ({variant})", "",
             f"Judge: `{j.get('model')}`, reasoning_effort=`{j.get('reasoning_effort')}`, temperature={j.get('temperature')}, "
             f"J={j.get('votes')} votes; "
             f"session {receipt.get('call_stats', {}).get('first_call_utc', receipt.get('started_utc'))} .. "
             f"{receipt.get('call_stats', {}).get('last_call_utc', receipt.get('finished_utc'))}.", "",
             "| system | K | audio APR % [95% CI] | audio rubric % [95% CI] | sample APR % | text_content % | text_constraint % "
             "| empty responses (mean / %) | judge errors |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for name, s in scores.items():
        inval = " INVALID" if s["any_invalid_read"] else ""
        lines.append(f"| {name} | {s['k_reads']} | {pct(s['audio_apr']['value'])}{ci(s['audio_apr']['ci95'])} | "
                     f"{pct(s['audio_rubric_rate']['value'])}{ci(s['audio_rubric_rate']['ci95'])} | "
                     f"{pct(s['sample_apr']['value'])} | {pct(s['text_content']['value'])} | "
                     f"{pct(s['text_constraint']['value'])} | {s['empty_responses_mean']:.1f} / "
                     f"{pct(s['empty_response_rate'])} | {s['judge_errors']}{inval} |")
    lines += ["", "Per category, audio APR % (audio rubric %):", "",
              "| system | " + " | ".join(SPLIT_SHORT[s] for s in SPLITS) + " |", "|---|" + "---:|" * len(SPLITS)]
    for name, s in scores.items():
        cells = [f"{pct(s['per_split'][sp]['audio_apr']['value'])} ({pct(s['per_split'][sp]['audio_rubric_rate']['value'])})"
                 for sp in SPLITS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Per read (audio APR % / audio rubric % / empty / judge errors):", ""]
    for name, s in scores.items():
        reads = "; ".join(f"{p['label']}: {pct(p['audio_apr'])} / {pct(p['audio_rubric_rate'])} / {p['empty_responses']} / "
                          f"{p['judge_errors']}" for p in s["per_read"])
        lines.append(f"- {name}: {reads}")
    if comparisons:
        lines += ["", "Paired item-bootstrap differences (pp, 95% CI):", ""]
        for c in comparisons:
            lines.append(f"- {c['a']} - {c['b']} ({c['metric']}): {100 * c['diff']:+.1f} "
                         f"[{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}] over {c['n_common_items']} items")
    return "\n".join(lines) + "\n"


def run_variant(receipt, systems, excluded, args, variant):
    scores = {name: score_system(reads, excluded, args.bootstrap, args.seed) for name, reads in systems.items()}
    comps = []
    pairs = [tuple(args.compare)] if args.compare else []
    for a, b in pairs:
        for metric in ("audio_apr", "audio_rubric_rate"):
            comps.append({"a": a, "b": b, **paired_diff(scores[a], scores[b], metric, args.bootstrap, args.seed)})
    md = render(receipt, variant, scores, comps)
    clean = {n: {k: v for k, v in s.items() if not k.startswith("_")} for n, s in scores.items()}
    return md, {"variant": variant, "systems": clean, "comparisons": comps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session_dir", required=True)
    ap.add_argument("--compare", nargs=2, metavar=("SYS_A", "SYS_B"), default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args(argv)

    receipt, systems = load_session(Path(args.session_dir))
    if not systems:
        print("no evaluation_results.json found in the session", file=sys.stderr)
        return 1
    if receipt.get("warning"):
        print(f"WARNING: {receipt['warning']}\n")
    md, j = run_variant(receipt, systems, set(), args, "all rubrics")
    js = [j]
    text = md
    print(text)
    if args.out_md:
        Path(args.out_md).write_text(text)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"judge": receipt.get("judge"),
                                                   "session": {k: receipt.get(k) for k in ("started_utc", "finished_utc", "call_stats")},
                                                   "results": js}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
