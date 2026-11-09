# MOMENT (vendored)

Unmodified copy of the `momentfm` package (version `0.1.5`) from
https://github.com/moment-timeseries-foundation-model/moment, commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e`
(2026-02-10), MIT license (`LICENSE`): `momentfm/__init__.py`, `common.py`, `models/` (`moment.py`, `layers/embed.py`,
`layers/revin.py`), `data/base.py` and `utils/{utils,masking}.py`, the files `momentfm.models.moment` imports. The
dataset loaders, metrics, statistical classifiers, tests and tutorials are left out.

Paper: MOMENT: A Family of Open Time-series Foundation Models, ICML 2024, https://arxiv.org/abs/2402.03885.
Weights load from the Hugging Face Hub (`AutonLab/MOMENT-1-base`, MIT).

It needs only torch, transformers (>= 4.54.1), huggingface_hub and numpy, already in the training environment;
nothing is pip-installed (`pip install momentfm` would pull its own pins into the env). `tools/baselines/moment.py`
puts this directory on `sys.path` so `import momentfm` finds it.
