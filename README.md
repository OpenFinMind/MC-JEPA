# MC-JEPA

Self-supervised representation learning for stock time series with a factorized panel JEPA
(`mc_jepa/models/qtsmfjepa`), plus series SSL baselines (CPC, MAE, TS2Vec, autoregressive) and
pretrained time-series foundation-model baselines (Chronos-2, Kronos, MOMENT, TimesFM).

## Layout

```
mc_jepa/
  models/        qtsmfjepa (ours), qtsmcpc, qtsmmae, qtsmts2vec, qtsmar
  datasets/      v3 per-stock series and v4 cross-sectional panel mmap datasets
  trainer/       panel_jepa_trainer.py (Hydra entry point for all models above)
  configs/panel/ training configs
tools/
  mmap/          build the v3/v4 mmap datasets from OHLCV bars
  daily_bars.py  daily-bar table for the optional daily context (data.daily_dir)
  eval_jepa_probe.py   embedding extraction and linear probes
  baselines/     feature (alpha158) and foundation-model baselines
external/        vendored upstream code for the foundation-model baselines (see each LICENSE)
```

## Usage

Requires Python >= 3.12. Install a CUDA build of PyTorch for your driver first, then the rest:

```bash
conda create -n mc-jepa python=3.12 -y && conda activate mc-jepa
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Data

Training reads 1-min OHLCV bars from mmap datasets: v3 (per-stock series) for the `*-series-*` configs and
v4 (cross-sectional panels) for `qtsmfjepa-panel-ohlcv-10m`. Build them from by-symbol parquet chunks
(`chunk_*.parquet` with `stock_symbol`, `date_int`, `time_int`, `open`, `high`, `low`, `close`, `volume`):

```bash
PYTHONPATH=. python tools/mmap/process_mmap_v3_dataset.py \
    --src-dir /path/to/ohlcv-1min-by-symbol --out-prefix /path/to/mmap-v3/1min/train-ohlcv --feature-set ohlcv
PYTHONPATH=. python tools/mmap/process_mmap_v4_dataset.py \
    --src-dir /path/to/ohlcv-1min-by-symbol --out-prefix /path/to/mmap-v4/1min/train-ohlcv --feature-set ohlcv
```

### Data paths

The configs in `mc_jepa/configs/panel/` ship with placeholder data paths
(`data.train.prefix: /path/to/mmap-v3/1min/train-ohlcv`, or `/path/to/mmap-v4/...` for the panel config);
`data.val.prefix` reuses it. Edit the config or override it on the command line.

### Train

```bash
torchrun --nproc_per_node=1 -m mc_jepa.trainer.panel_jepa_trainer --config-name=qtsmfjepa-panel-ohlcv-10m \
    data.train.prefix=/path/to/mmap-v4/1min/train-ohlcv
```

### Evaluate

Frozen-encoder ridge probes on (stock, day) windows ending at the 15:00 close, using the v3 data
(train/val/test split by date; see `tools/eval_jepa_probe.py` for the task definitions):

```bash
PREFIX=/path/to/mmap-v3/1min/train-ohlcv
CKPT=experiments/<run>/global_step_20000

# 1. labels (future returns, beta, volatility, volume, ...)
python -m tools.eval_jepa_probe labels --prefix $PREFIX --out var/probe/labels.parquet
# 2. embed with the frozen encoder -> $CKPT/probe/embeddings.npz
python -m tools.eval_jepa_probe embed --checkpoint $CKPT --prefix $PREFIX
# 3. fit probes -> metrics.json, metrics.txt and test predictions next to the embeddings
python -m tools.eval_jepa_probe probe --embeddings $CKPT/probe/embeddings.npz --labels var/probe/labels.parquet
```

Baselines are written as npz on the same rows and probed with `--extra`:

```bash
python -m tools.eval_jepa_probe alpha158 --prefix $PREFIX --out var/probe/alpha158.npz  # also: kronos, timesfm, chronos2, moment
python -m tools.eval_jepa_probe probe --embeddings $CKPT/probe/embeddings.npz --labels var/probe/labels.parquet \
    --extra var/probe/alpha158.npz --features alpha158
```

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Code vendored under `external/` keeps its own license.
