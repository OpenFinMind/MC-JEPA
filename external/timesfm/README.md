# TimesFM 2.5 (vendored)

Unmodified copy of the TimesFM 2.5 PyTorch code from https://github.com/google-research/timesfm, commit
`e51928e27119cb17bebc005be2696b75e0a9e688` (2026-09-29), Apache-2.0 license (`LICENSE`): `src/timesfm/__init__.py`,
`configs.py`, `torch/` and `timesfm_2p5/timesfm_2p5_{base,torch}.py`. The Flax model, the XReg utilities and TimesFM 3
(`src/timesfm3`) are left out; `timesfm/__init__.py` imports them in `try` blocks, so they are skipped.

Paper: A decoder-only foundation model for time-series forecasting, ICML 2024, https://arxiv.org/abs/2310.10688.
Weights load from the Hugging Face Hub (`google/timesfm-2.5-200m-pytorch`, Apache-2.0).

It needs only torch, numpy, safetensors and huggingface_hub, already in the training environment; nothing is
pip-installed. `tools/baselines/timesfm.py` puts this directory on `sys.path` so `import timesfm` finds it.
