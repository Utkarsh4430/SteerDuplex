# SteerBench

**SteerBench** measures *spoken steerability*: can a speech-to-speech assistant follow an explicit spoken request to
change **how** it speaks (tone, persona, style/accent or speed/length) while still doing the task it was asked to do?

- **Items.** 390 single-turn English items: 100 tone, 136 persona, 104 style/accent, 50 speed/length.
- **Rubrics.** 1,067 binary rubrics: 438 audio, 478 text-content and 151 text-constraint.
- **Audio.** Each item has a spoken user turn and a reference clip that realises the requested delivery.
- **Judging.** An LLM judge grades the response audio against the reference (style only) and the response text against the text rubrics.

- 📄 **Paper:** [SteerDuplex: Steerable Duplex Speech Dialogue Models (arXiv:2609.12623)](https://arxiv.org/abs/2609.12623)
- 🤗 **Benchmark data:** [huggingface.co/datasets/utkarsh4430/SteerBench](https://huggingface.co/datasets/utkarsh4430/SteerBench)

This repository holds **only the code** (inference, judge, scorer). The benchmark data lives on the Hugging Face
dataset and is downloaded into `data/` by `scripts/fetch_dataset.py`.

```
inference/     reference inference for Moshi-family / SteerDuplex-style models + ASR step
judge/         model-agnostic judge: prompts (verbatim), config, OpenAI-SDK client, interleaved session runner
score.py       model-agnostic scorer: per-category / per-axis scores with 95% item-bootstrap CIs
scripts/       fetch_dataset.py, downloads the benchmark from Hugging Face into data/
```

> **Scope.** `inference/` is the **reference runner for Moshi-family / SteerDuplex-style** speech models only. The
> **judge and scorer are model-agnostic**: they work on any system's outputs in the run-directory format
> (`<id>.wav` + `<id>.json` per item). To evaluate a different architecture, generate that format with your own
> inference and skip `inference/`; then run the judge and scorer unchanged.

## Quickstart

```bash
pip install -r requirements.txt                    # judge + scoring
pip install -r inference/requirements.txt          # only to run the reference inference path (GPU)

# the dataset is gated: open its Hugging Face page, click to accept the terms, then log in once:
#   huggingface-cli login
# then download the benchmark into data/ (items, rubrics, user + reference audio, system & voice prompts)
python scripts/fetch_dataset.py

# 1) Inference: write <id>.wav + <id>.json (response text) for every item, K >= 2 decode seeds
python -m inference.run_inference --model moshi_family --seed 292500 --out_dir runs/mymodel_s292500
python -m inference.run_inference --model moshi_family --seed 292501 --out_dir runs/mymodel_s292501
#    a SteerDuplex checkpoint:  --model_kwargs '{"checkpoint": "path/to/consolidated"}'
#    your own model:            --model my_package.my_module:MyModel   (see inference/README.md)

# 2) Judge all the runs you compare in ONE session (J = 3 votes per rubric list).
#    The judge uses the OpenAI SDK: set your key (and OPENAI_BASE_URL for an OpenAI-compatible endpoint).
export OPENAI_API_KEY=...
python -m judge.run --session_dir judgments/my_session --model gpt-4o \
    --run mymodel=runs/mymodel_s292500 --run mymodel=runs/mymodel_s292501

# 3) Score with CIs (and a paired comparison between two systems)
python score.py --session_dir judgments/my_session --compare mymodel other_system
```

For a smoke test, add `--limit 5` to steps 1 and 2. To just browse the data, `load_dataset("utkarsh4430/SteerBench")` (decoding the embedded audio needs `pip install torchcodec`).

## What gets measured

Each item has a list of **audio rubrics** (`audio_steering`) and **text rubrics** (`text_content`: engages with the
task; `text_constraint`: textual markers of the requested style, e.g. vocabulary or length).

| metric | definition |
|---|---|
| **audio APR** (headline) | Share of items whose audio rubrics **all** pass. |
| audio rubric rate | Share of individual audio rubrics passed. |
| sample APR | Share of items whose audio **and** text rubrics all pass. |
| text_content / text_constraint | Per-rubric pass rates. |
| empty responses | Items whose response text has fewer than 3 words. Read low scores together with this column. |
| judge errors | Items whose judge calls failed after retries; excluded from denominators. A read with > 5 % errors is **invalid**. |

The examinee setup:

- **Input.** The item's user WAV, the system prompt in `data/system_prompt.txt` and, for voice-conditioned models, the voice prompt `data/voice_prompt/libri_hq24k_1226.wav`.
- **Output.** The assistant's single response turn, as audio plus text. Text is the model's own text stream if it has one; otherwise `inference/asr.py` (Whisper) transcribes the audio.
- **ASR caveat.** ASR text scores are **not** directly comparable to text-stream scores, so report which you used (the sidecar records `text_source`).

## Judge

| setting | value |
|---|---|
| model | you choose (`--model`); any OpenAI or OpenAI-compatible chat model. Key via `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for other endpoints) |
| thinking | `--reasoning_effort low/medium/high` for reasoning models (omitted by default) |
| sampling | temperature 0, `max_tokens` 4096, JSON response format |
| votes | per-rubric strict majority over **J = 3** independent calls; ties fail |
| text judge | sees the user utterance, the response text and the text rubrics |
| audio judge | sees the reference clip (labelled "style anchor; ignore content"), the response clip and the audio rubrics; no transcript |

The prompts in `judge/prompts.py` are verbatim; the config is in `judge/config.py`. Every session writes
`session.json` recording the judge settings, dates, backend-reported model id, token/latency stats, and
`repeat_reply_identity` (the share of items whose J replies are byte-identical, near 0 for a thinking judge).

## Protocol: read this before comparing numbers

SteerBench scores depend on the judge configuration more than on most model differences of interest.

- **Thinking matters.** The same judge model with thinking off passed ~20 pp fewer audio rubrics on identical audio.
- **Session effect.** One configuration judged on different days drifts by about 3 to 4 pp SD per read, roughly 4x the within-session noise.
- **Decode noise.** One read (K = 1) moved up to 5.8 pp between decode seeds.

So:

1. Use **K ≥ 2 decode seeds** per system (the reference runs use item-derived seeds with base 292500/292501).
2. Use **J = 3 judge votes** (the default).
3. **Judge every system you compare in one interleaved session**: pass all their runs to a single `judge.run` call, which shuffles every (run, item, vote) into one queue. Numbers from different sessions are not comparable.
4. **Fix one judge model and configuration** across a comparison, and report the model, thinking level and dates from `session.json`.
5. Report **95 % item-bootstrap CIs** and paired differences (`score.py --compare`); also look at the per-read values.
6. Do not report per-persona / per-accent / per-sub-axis numbers (n ≤ 4 items each).

Decoding one read with a 7B Moshi-family model takes ~2.3 GPU-hours on H100s.

## Scope and limitations

- Single-turn, explicit requests only, in English.
- One user speaking style (General American) across 30 prebuilt voices; synthetic references from one TTS family.
- One judge model per comparison; no archived human-agreement study.
- The benchmark is frozen for comparability.

## Licence

Code: **MIT** (see [LICENSE](LICENSE)). Benchmark data: **CC BY-NC-SA 4.0**, on the
[Hugging Face dataset](https://huggingface.co/datasets/utkarsh4430/SteerBench). Vendored third-party components
(the Moshi code under `inference/moshi_family/vendor/`) keep their own licences.

## Citation

```bibtex
@misc{tyagi2026steerduplexsteerableduplexspeech,
      title={SteerDuplex: Steerable Duplex Speech Dialogue Models},
      author={Utkarsh Tyagi and Ramaneswaran Selvakumar and Advait Gosai and Sonal Kumar and Nikhil Barhate and Isabell Sagar and Steven Li and Miheer Bavare and Daniel Quigley and Fabiola Tapia Carrillo and Jose M Patron E and Diego Macías Gutiérrez and Paul Song and Ramani Duraiswami and Dinesh Manocha and Yunzhong He},
      year={2026},
      eprint={2609.12623},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.12623},
}
```
