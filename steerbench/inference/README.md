# SteerBench inference

This step generates one spoken response per benchmark item. The judge
(`../judge/`) then scores those responses.

**Scope.** This runner is for **Moshi-family and SteerDuplex-style** speech models only (see the `moshi_family`
path below). The judge and scorer are model-agnostic: to evaluate a different architecture, generate the
run-directory format (`<id>.wav` + `<id>.json` per item) with your own inference and skip this step.

```bash
pip install -r inference/requirements.txt
python scripts/fetch_dataset.py  # downloads the benchmark into data/ (items, audio, prompts)

# reference path: public Moshi, seed a
python -m inference.run_inference --model moshi_family --out_dir runs/moshiko-a --seed 292500
```

Each run directory contains `<id>.wav`, `<id>.json` (the sidecar the judge
reads) and `run_config.json`. The run config records the model, the
checkpoint name, the SHA-256 of the prompts and benchmark files, the settings,
the package version and the date. Sidecars hold only relative paths.
Rerunning the same command resumes: an item whose WAV and sidecar both exist
with the matching seed is skipped. To run in parallel, use
`--num_shards N --shard_index k` with the same `--out_dir`.

## Plugging in a model

Subclass `inference.interface.SteerBenchModel` and implement one method:

```python
from inference.interface import ModelResponse, SteerBenchModel

class MyModel(SteerBenchModel):
    name = "my_model"
    def __init__(self, **kwargs): ...           # receives --model_kwargs (JSON)
    def respond(self, user_audio_path, system_prompt, voice_prompt_path, seed, max_duration_sec):
        ...
        return ModelResponse(audio=wav_float32_mono, sample_rate=24000,
                             text=my_text_or_None, metadata={"stop_reason": "..."})
```

```bash
python -m inference.run_inference --model my_pkg.my_model:MyModel --out_dir runs/my_model
python -m inference.run_inference --model path/to/my_model.py:MyModel --out_dir runs/my_model
```

The benchmark is single-turn. Feed the user turn and return only the
assistant turn's audio: no user audio, no prompt audio, mono. `text` is the
model's own text stream, if it has one, with control tokens stripped;
otherwise return `None`. `seed` is the per-item seed. The driver has already
reset the Python, NumPy and torch RNGs with it. The docstrings in
`interface.py` give the full contract. `examples/custom_model.py` is a
runnable template with echo and silence baselines.

Defaults match the reference runs:

| flag | default |
|---|---|
| `--system_prompt_file` | `data/system_prompt.txt` |
| `--voice_prompt` | `data/voice_prompt/libri_hq24k_1226.wav` (`none` disables it) |
| `--max_duration` | 90 s timeline per item |
| `--seed` | 292500 (seed a); per-item seed derived from the base seed and the item id |

## Text source and ASR

Text rubrics score a transcript. Moshi, PersonaPlex and SteerDuplex emit their
own text stream, and the reference numbers score that stream
(`text_source: "model_text_stream"`). With `--asr auto` (the default), a
model that returns `text=None` is transcribed with Whisper (`--asr_model`,
default `openai/whisper-large-v3`), and `text_source` is set to
`"asr:<model>"`. Use `--asr always` to transcribe every model. In that mode
the model's own text is kept as `model_text`.

**Text-rubric scores from ASR transcripts are not directly comparable to
scores from a model's own text stream.** ASR introduces recognition errors and
normalises disfluencies, while a text stream can contain words that were never
clearly spoken. To compare an audio-only system like for like, rerun the
reference models with `--asr always`. Audio rubrics are unaffected.

## Reference path (`--model moshi_family`)

`moshi_family/` is a self-contained port of the code behind the published
numbers. It contains the vendored Kyutai `moshi` 0.2.13 with NVIDIA
PersonaPlex's system/voice-prompt `LMGen`; see
`moshi_family/vendor/NOTICE.md` for licences and modifications. The adapter
runs one item at a time:

1. Prefill the voice prompt (-24 LUFS), silence, `<system> {prompt} <system>`
   and silence.
2. Stream the user turn, padded with silence to the 90 s timeline. Sampling
   uses audio temperature 0.8 / top-k 250 and text temperature 0.7 / top-k 25.
3. Stop after 5 s of text silence or 5 s of audio silence, or when there is
   still no answer 5 s after the user stops.
4. A generation that hits the 90 s cap is regenerated once without reseeding
   (the lossless retry). The retry timeline is the smallest of 300 s, the
   300 s ceiling and the model's attention window (240 s for moshiko's
   3000-frame window).

| model | `--model_kwargs` |
|---|---|
| Moshi (`kyutai/moshiko-pytorch-bf16`) | `{}` |
| PersonaPlex (`nvidia/personaplex-7b-v1`; gated: accept the licence on the Hub, set `HF_TOKEN`) | `{"hf_repo": "nvidia/personaplex-7b-v1"}` |
| SteerDuplex checkpoint (`consolidated/` with `consolidated.safetensors` + `config.json`) | `{"checkpoint": "/path/to/consolidated"}` |

*The SteerDuplex checkpoint is not yet publicly released; the `moshi_family` path runs public Moshi and PersonaPlex out of the box, and any local Moshi-style `consolidated/` checkpoint via `--model_kwargs`.*

Other keys: `device` (default `cuda`), `greedy` (default `false`), `family`
(`auto` / `moshi` / `personaplex`), `lossless_retry_max_duration`,
`context_ceiling_sec` and `label`. A SteerDuplex checkpoint loads its weights
and `config.json` (including its attention window) on top of
`kyutai/moshiko-pytorch-bf16`, which supplies the Mimi codec and the text
tokenizer.

## Reproducing the reference runs

The published numbers average two seeds per model: base seed 292500 (a) and
292501 (b). Runs used item seed mode, batch size 1, the default system and
voice prompts, `--max_duration 90` and a lossless-retry cap of 300 s.

```bash
for s in 292500 292501; do
  python -m inference.run_inference --model moshi_family --seed $s --out_dir runs/moshiko-$s
  python -m inference.run_inference --model moshi_family --seed $s --out_dir runs/personaplex-$s \
      --model_kwargs '{"hf_repo": "nvidia/personaplex-7b-v1"}'
  python -m inference.run_inference --model moshi_family --seed $s --out_dir runs/steerduplex-$s \
      --model_kwargs '{"checkpoint": "/path/to/consolidated"}'
done
```

Per item, the seed is `sha256(b"steerbench-item-seed-v1\0" + base_seed +
b"\0" + id)`: the first 8 bytes, big-endian, masked to 63 bits
(`interface.derive_item_seed`). Outputs therefore do not depend on sharding.
On the same GPU type and software stack, the port reproduces the original
code's WAVs and text exactly. Other GPUs, drivers or library versions can
legitimately produce different samples from the same seed.
