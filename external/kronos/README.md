# Kronos (vendored)

Unmodified copy of `model/` and `tests/test_kronos_regression.py` (with its data) from
https://github.com/shiyu-coder/Kronos (mirrored at https://github.com/Douglans/kronos), commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13), MIT license (`LICENSE`).

Paper: Kronos: A Foundation Model for the Language of Financial Markets, https://arxiv.org/abs/2508.02739.
Weights load from the Hugging Face Hub (`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`).

`model/kronos.py` imports `model.module`, so this directory must be on `sys.path`; `tools/baselines/kronos.py`
adds it. Upstream regression test (run from this directory):

    cd external/kronos && python -m pytest tests/test_kronos_regression.py
