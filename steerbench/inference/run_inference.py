#!/usr/bin/env python3
"""Generate SteerBench responses for one model.

Run from the repository root:

    # reference path: public Moshi (moshiko), seed a
    python -m inference.run_inference --model moshi_family --out_dir runs/moshiko-a --seed 292500

    # a SteerDuplex checkpoint
    python -m inference.run_inference --model moshi_family \\
        --model_kwargs '{"checkpoint": "/path/to/consolidated"}' --out_dir runs/steerduplex-a

    # your own model (module path or file path, then :ClassName)
    python -m inference.run_inference --model my_pkg.my_model:MyModel --out_dir runs/mine
    python -m inference.run_inference --model inference/examples/custom_model.py:EchoModel \\
        --out_dir runs/echo --limit 3

Outputs in ``--out_dir``: ``<id>.wav``, ``<id>.json`` (the sidecar the judge reads) and
``run_config.json``. Items run one at a time (batch size 1) in benchmark order. Rerunning
the same command resumes: an item whose WAV and sidecar exist with a matching seed is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # allow `python inference/run_inference.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf  # noqa: E402

from inference import __version__  # noqa: E402
from inference.interface import (  # noqa: E402
    ITEM_SEED_DERIVATION,
    ModelResponse,
    SteerBenchModel,
    derive_item_seed,
    seed_global_rng,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SEED = 292500  # reference seed a; seed b = 292501
DEFAULT_MAX_DURATION = 90.0
SEED_MODE = "item"
BUILTIN_MODELS = {"moshi_family": "inference.moshi_family:MoshiFamilyModel"}

logger = logging.getLogger("steerbench.inference")


# ─── helpers ────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_items(data_dir: Path) -> list[dict]:
    """Benchmark items in canonical order (from steerbench.json; audio under user_audio/)."""
    bench = json.loads((data_dir / "steerbench.json").read_text())
    return [{
        "sample_id": item["id"],
        "split": item["split"],
        "sub_axis": item["sub_axis"],
        "user_utterance": item["user_utterance"],
        "user_audio": f"user_audio/{item['id']}.wav",  # relative to data_dir
    } for item in bench["items"]]


def parse_ids(spec: str | None) -> list[str] | None:
    """``--ids a,b,c`` or ``--ids @file`` (JSON list or one id per line)."""
    if not spec:
        return None
    if spec.startswith("@"):
        text = Path(spec[1:]).read_text().strip()
        ids = json.loads(text) if text.startswith("[") else text.split()
    else:
        ids = [s for s in spec.replace(" ", "").split(",") if s]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("--ids must be non-empty and unique")
    return [str(i) for i in ids]


def load_model_class(spec: str) -> type:
    spec = BUILTIN_MODELS.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"--model must be 'moshi_family' or 'module.path:ClassName' / 'file.py:ClassName', got {spec!r}")
    module_name, class_name = spec.rsplit(":", 1)
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[path.stem] = module
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, SteerBenchModel)):
        raise TypeError(f"{spec} is not a subclass of inference.interface.SteerBenchModel")
    return cls


def is_complete(out_dir: Path, sample_id: str, base_seed: int) -> bool:
    """A resumable output is a non-empty WAV plus a sidecar for the same item and seed."""
    wav, meta_path = out_dir / f"{sample_id}.wav", out_dir / f"{sample_id}.json"
    try:
        if wav.stat().st_size <= 0 or meta_path.stat().st_size <= 0:
            return False
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(meta, dict)
        and meta.get("sample_id") == sample_id
        and meta.get("seed_mode") == SEED_MODE
        and meta.get("generation_seed") == derive_item_seed(base_seed, sample_id)
    )


# ─── main ───────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="'moshi_family' (reference Moshi/PersonaPlex/SteerDuplex path), "
                        "'package.module:ClassName' or 'path/to/file.py:ClassName' (a SteerBenchModel subclass).")
    p.add_argument("--model_kwargs", default="{}",
                   help="JSON object passed to the model constructor, e.g. "
                        "'{\"checkpoint\": \"/path/consolidated\"}' or '{\"hf_repo\": \"nvidia/personaplex-7b-v1\"}'.")
    p.add_argument("--out_dir", required=True, help="Output directory, e.g. runs/<name>.")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help="Directory holding steerbench.json and user_audio/ (default: <repo>/data; populate with scripts/fetch_dataset.py).")
    p.add_argument("--system_prompt_file", default=None,
                   help="Examinee system prompt (default: <data_dir>/system_prompt.txt). Pass 'none' for no system prompt.")
    p.add_argument("--voice_prompt", default=None,
                   help="Voice-prompt WAV (default: <data_dir>/voice_prompt/libri_hq24k_1226.wav). Pass 'none' to disable.")
    p.add_argument("--max_duration", type=float, default=DEFAULT_MAX_DURATION,
                   help="Generation timeline per item in seconds (reference: 90).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Base seed; each item gets sha256-derived seed from (seed, id). Reference runs: 292500 and 292501.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N selected items (after sharding).")
    p.add_argument("--ids", default=None, help="Comma-separated item ids, or @file (JSON list / one id per line).")
    p.add_argument("--num_shards", type=int, default=1, help="Split items round-robin into N shards (by index).")
    p.add_argument("--shard_index", type=int, default=0, help="Which shard this process runs, in [0, num_shards).")
    p.add_argument("--asr", choices=("auto", "never", "always"), default="auto",
                   help="auto: transcribe only when the model returns text=None; always: always use ASR text "
                        "(model text kept as model_text); never: never run ASR.")
    p.add_argument("--asr_model", default="openai/whisper-large-v3", help="Hugging Face Whisper model id for ASR.")
    p.add_argument("--asr_device", default=None, help="Device for ASR (default: cuda if available).")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard_index must be in [0, num_shards)")
    if args.max_duration <= 0:
        raise SystemExit("--max_duration must be positive")

    # Prompts.
    sp_arg = args.system_prompt_file
    if sp_arg and sp_arg.lower() == "none":
        system_prompt, sp_name = "", None
    else:
        sp_path = Path(sp_arg) if sp_arg else data_dir / "system_prompt.txt"
        system_prompt, sp_name = sp_path.read_text().strip(), sp_path.name
    vp_arg = args.voice_prompt
    if vp_arg and vp_arg.lower() == "none":
        voice_prompt = None
    else:
        voice_prompt = Path(vp_arg) if vp_arg else data_dir / "voice_prompt" / "libri_hq24k_1226.wav"
        if not voice_prompt.is_file():
            raise SystemExit(f"voice prompt not found: {voice_prompt}")

    # Items: canonical order -> --ids -> shard -> --limit.
    all_items = load_items(data_dir)
    items = all_items
    ids = parse_ids(args.ids)
    if ids is not None:
        known = {it["sample_id"] for it in all_items}
        unknown = sorted(set(ids) - known)
        if unknown:
            raise SystemExit(f"unknown --ids: {unknown[:10]}")
        wanted = set(ids)
        items = [it for it in items if it["sample_id"] in wanted]
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        items = items[: args.limit]

    pending = [it for it in items if not is_complete(out_dir, it["sample_id"], args.seed)]
    logger.info("%d items selected, %d already complete, %d pending", len(items), len(items) - len(pending), len(pending))

    model_kwargs = json.loads(args.model_kwargs)
    if not isinstance(model_kwargs, dict):
        raise SystemExit("--model_kwargs must be a JSON object")
    model_cls = load_model_class(args.model)
    if args.model == "moshi_family":
        model_kwargs.setdefault("init_seed", args.seed)  # as in the reference runs

    run_config = {
        "schema": "steerbench-run-config",
        "steerbench_inference_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_spec": args.model,
        "model_class": f"{model_cls.__module__}.{model_cls.__qualname__}",
        "model_kwargs": {k: v for k, v in model_kwargs.items() if k != "checkpoint"},
        "model": None,  # filled after construction (model.describe())
        "system_prompt_file": sp_name,
        "system_prompt_sha256": sha256_text(system_prompt) if system_prompt else None,
        "voice_prompt_file": voice_prompt.name if voice_prompt else None,
        "voice_prompt_sha256": sha256_file(voice_prompt) if voice_prompt else None,
        "max_duration_sec": args.max_duration,
        "batch_size": 1,
        "seed_mode": SEED_MODE,
        "base_seed": args.seed,
        "seed_derivation": ITEM_SEED_DERIVATION,
        "asr": args.asr,
        "asr_model": args.asr_model if args.asr != "never" else None,
        "bench_sha256": sha256_file(data_dir / "steerbench.json"),
        "n_items_selected": len(items),
        "ids_filter": bool(ids),
        "limit": args.limit,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "environment": {"python": platform.python_version()},
    }
    if ":" in args.model and args.model.rsplit(":", 1)[0].endswith(".py"):  # no local paths in run_config
        file_part, class_part = args.model.rsplit(":", 1)
        run_config["model_spec"] = f"{Path(file_part).name}:{class_part}"
    config_name = "run_config.json" if args.num_shards == 1 else f"run_config.shard{args.shard_index}of{args.num_shards}.json"

    if not pending:
        logger.info("nothing to do")
        return 0

    # Seed before construction (the reference engine seeds again internally), then build.
    seed_global_rng(args.seed)
    model: SteerBenchModel = model_cls(**model_kwargs)
    run_config["model"] = model.describe()
    try:
        import torch
        run_config["environment"].update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        pass
    atomic_write_json(out_dir / config_name, run_config)

    asr = None
    n_ok = n_fail = 0
    t_start = time.time()
    for index, item in enumerate(pending, 1):
        sid = item["sample_id"]
        item_seed = derive_item_seed(args.seed, sid)
        user_audio_path = data_dir / item["user_audio"]
        t0 = time.time()
        try:
            seed_global_rng(item_seed)
            response = model.respond(
                user_audio_path=str(user_audio_path),
                system_prompt=system_prompt,
                voice_prompt_path=str(voice_prompt) if voice_prompt else None,
                seed=item_seed,
                max_duration_sec=args.max_duration,
            )
            if not isinstance(response, ModelResponse):
                raise TypeError(f"respond() must return ModelResponse, got {type(response).__name__}")

            model_text = response.text
            use_asr = args.asr == "always" or (args.asr == "auto" and model_text is None)
            if use_asr:
                if asr is None:
                    from inference.asr import WhisperASR
                    asr = WhisperASR(model_id=args.asr_model, device=args.asr_device)
                text = asr.transcribe(response.audio, response.sample_rate)
                text_source = f"asr:{args.asr_model}"
            elif model_text is not None:
                text, text_source = model_text, "model_text_stream"
            else:
                text, text_source = "", "none"
                logger.warning("%s: model returned no text and --asr never; text is empty", sid)

            sf.write(out_dir / f"{sid}.wav", response.audio, response.sample_rate)
            sidecar = {
                "sample_id": sid,
                "split": item["split"],
                "sub_axis": item["sub_axis"],
                "user_utterance": item["user_utterance"],
                "user_audio_path": item["user_audio"],
                "text": text,
                "moshi_text": text,  # same value; field name read by the SteerBench judge
                "text_source": text_source,
                "sample_rate": int(response.sample_rate),
                "audio_duration_sec": float(len(response.audio)) / float(response.sample_rate),
                "generation_seed": item_seed,
                "seed_mode": SEED_MODE,
                "base_seed": args.seed,
                "generation_metadata": response.metadata,
            }
            if use_asr and model_text is not None:
                sidecar["model_text"] = model_text
            atomic_write_json(out_dir / f"{sid}.json", sidecar)
            n_ok += 1
            logger.info("[%d/%d] %s %.1fs audio in %.1fs (%s)", index, len(pending), sid,
                        sidecar["audio_duration_sec"], time.time() - t0,
                        (response.metadata or {}).get("stop_reason", "-"))
        except Exception:
            n_fail += 1
            logger.error("[%d/%d] %s FAILED\n%s", index, len(pending), sid, traceback.format_exc())

    missing = [it["sample_id"] for it in items if not is_complete(out_dir, it["sample_id"], args.seed)]
    logger.info("done: %d ok, %d failed, %d/%d complete in %s (%.0fs)", n_ok, n_fail,
                len(items) - len(missing), len(items), out_dir, time.time() - t_start)
    if missing:
        logger.error("incomplete items (%d): %s", len(missing), missing[:10])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
