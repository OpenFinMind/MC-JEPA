# Chronos-2 (vendored)

Unmodified copy of the `chronos` package (`src/chronos/`, version `2.3.3.dev0`) from
https://github.com/amazon-science/chronos-forecasting, commit `1224fafa91e1ecd6674f5204066b159e40b5e4a1`
(2026-10-05), Apache-2.0 license (`LICENSE`, `NOTICE`). The whole package is kept because `chronos/__init__.py`
imports Chronos, Chronos-Bolt and Chronos-2; only `chronos2/` (and `InstanceNorm` / `Patch` from `chronos_bolt.py`)
is used. Tests, notebooks and scripts are left out.

Paper: Chronos-2: From Univariate to Universal Forecasting, https://arxiv.org/abs/2510.15821.
Weights load from the Hugging Face Hub (`amazon/chronos-2`, Apache-2.0).

It needs only torch, transformers, accelerate, einops, numpy, pandas and packaging, already in the training
environment; nothing is pip-installed (boto3, peft and fev are imported lazily, only for S3 models, LoRA adapters and
fev benchmarks). `tools/baselines/chronos2.py` puts this directory on `sys.path` so `import chronos` finds it.
