# Third-party code in `inference/moshi_family/vendor/`

`vendor/moshi/` is a pruned copy of the Python `moshi` package used to run the
reference (Moshi-family) models. It is redistributed under the original
licences; every file keeps its upstream copyright header.

| Origin | What | Licence | Licence file |
|---|---|---|---|
| [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi), Python package `moshi` 0.2.13 | everything under `vendor/moshi/` (Mimi codec, LM, transformer, quantization, utils) | MIT (Python code) | `vendor/moshi/LICENSE` |
| [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) (via Kyutai moshi) | portions of `modules/`, `quantization/`, `utils/`, `models/lm.py` marked "Copyright (c) Meta Platforms" | MIT | `vendor/moshi/LICENSE.audiocraft` |
| [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) | system-prompt / voice-prompt `LMGen` and helpers in `models/lm.py`; `get_moshi_lm_personaplex` in `models/loaders.py` | MIT (code only; PersonaPlex **weights** are under the NVIDIA Open Model License and are not included here) | `vendor/LICENSE-personaplex-MIT` |

## Modifications relative to upstream

* Only the modules needed for offline inference are kept: the client, server,
  offline/TTS entry points and the upstream unit tests were removed.
* Every file except `models/lm.py`, `models/loaders.py` and
  `models/__init__.py` is byte-identical to `moshi` 0.2.13.
* `models/lm.py`: upstream `LMGen` replaced by the PersonaPlex `LMGen`, which
  injects the voice prompt, silence and `<system> ... <system>` text prompt
  through an explicit `provided` mask before the conversation starts. It also
  records the text-token log-probability at each step and accepts an optional
  per-row `provided_mask`. Unused experimental decoding hooks were removed;
  removing them does not change outputs.
* `models/loaders.py`: adds `get_moshi_lm_personaplex` (dep_q = 16 expansion
  from NVIDIA PersonaPlex). `_lm_kwargs` lists the deprecated
  `depformer_causal` key, which the loader drops before building the model.
* `models/__init__.py`: also exports `get_moshi_lm_personaplex`.

No code from `kyutai-labs/moshi-finetune` (Apache-2.0) is included. The eight
PersonaPlex silence-codec token ids it also defines are already in
`models/lm.py` (`SILENCE_TOKENS`).
