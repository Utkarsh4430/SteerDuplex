#!/usr/bin/env python3
"""Download the SteerBench benchmark from the Hugging Face dataset into ``data/``.

Run this once before inference / judging / scoring. It writes:

    data/steerbench.json             items + rubrics (reconstructed from the dataset parquet)
    data/user_audio/<id>.wav         the spoken user turn
    data/reference_audio/<id>.wav    the reference delivery clip
    data/reference_audio/<id>.json   transcript, delivery instruction, voice
    data/system_prompt.txt           the examinee system prompt
    data/voice_prompt/<file>.wav     the voice prompt

Usage:  python scripts/fetch_dataset.py            # -> ./data
        python scripts/fetch_dataset.py --out DIR
"""
import argparse
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = "utkarsh4430/SteerBench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--repo_id", default=REPO)
    args = ap.parse_args()
    out = args.out

    # 1) non-parquet benchmark files (system prompt + voice prompt)
    snapshot_download(args.repo_id, repo_type="dataset", local_dir=out,
                      allow_patterns=["system_prompt.txt", "voice_prompt/*"])

    # 2) items, rubrics and audio, reconstructed from the parquet
    api = HfApi()
    parquets = [f for f in api.list_repo_files(args.repo_id, repo_type="dataset") if f.endswith(".parquet")]
    ua = os.path.join(out, "user_audio"); ra = os.path.join(out, "reference_audio")
    os.makedirs(ua, exist_ok=True); os.makedirs(ra, exist_ok=True)
    items = []
    for pqf in parquets:
        local = hf_hub_download(args.repo_id, pqf, repo_type="dataset")
        for r in pq.read_table(local).to_pylist():
            i = r["id"]
            with open(os.path.join(ua, f"{i}.wav"), "wb") as f: f.write(r["user_audio"]["bytes"])
            with open(os.path.join(ra, f"{i}.wav"), "wb") as f: f.write(r["reference_audio"]["bytes"])
            with open(os.path.join(ra, f"{i}.json"), "w") as f:
                json.dump({"sample_id": i, "split": r["split"], "sub_axis": r["sub_axis"],
                           "user_utterance": r["user_utterance"], "transcript": r["transcript"],
                           "tts_instruct": r["tts_instruct"], "voice_id": r["reference_voice"],
                           "sample_rate": 24000}, f, ensure_ascii=False, indent=1)
            items.append({
                "id": i, "split": r["split"], "steering_axis": r["steering_axis"],
                "sub_axis": r["sub_axis"], "sub_axis_group": r["sub_axis_group"], "topic": r["topic"],
                "user_utterance": r["user_utterance"],
                "audio_rubrics": [{"axis": "audio_steering", "item": x} for x in r["audio_rubrics"]],
                "text_rubrics": [{"axis": "text_content", "item": x} for x in r["text_content_rubrics"]]
                              + [{"axis": "text_constraint", "item": x} for x in r["text_constraint_rubrics"]],
            })
    items.sort(key=lambda d: d["id"])
    with open(os.path.join(out, "steerbench.json"), "w") as f:
        json.dump({"name": "steerbench", "n_items": len(items), "items": items}, f, ensure_ascii=False, indent=1)
    print(f"fetched {len(items)} items -> {out}/  (steerbench.json, user_audio/, reference_audio/, system_prompt.txt, voice_prompt/)")


if __name__ == "__main__":
    raise SystemExit(main())
