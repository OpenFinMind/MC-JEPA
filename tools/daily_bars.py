"""Build the daily-bar table of ``mc_jepa/datasets/daily.py`` (``data.daily_dir``) from a v3 1-min mmap:
dates.npy [D], symbols.npy [S], ohlcv.npy [D, S, 5] float32. Run from the repo root:
    python -m tools.daily_bars --prefix <v3 OHLCV mmap prefix> --out <output dir>
"""

import click

from mc_jepa.datasets.daily import build_daily_bars, save_daily_bars
from mc_jepa.datasets.v3.indexed_dataset import MMapIndexedDatasetV3


@click.command()
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, help="output directory")
def main(prefix, out):
    dates, symbols, ohlcv = build_daily_bars(MMapIndexedDatasetV3(prefix))
    save_daily_bars(out, dates, symbols, ohlcv)
    print(f"{len(dates)} dates {dates[0]}-{dates[-1]}, {len(symbols)} symbols -> {out}")


if __name__ == "__main__":
    main()
