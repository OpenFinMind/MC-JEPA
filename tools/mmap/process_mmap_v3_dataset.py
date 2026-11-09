"""Convert the by-symbol 1-min OHLCV parquet chunks into one v3 dataset (mc_jepa/datasets/v3/indexed_dataset.py).

Each ``chunk_*.parquet`` holds whole symbols, rows grouped by ``stock_symbol`` and sorted by
(``date_int``, ``time_int``) within a symbol. Each feature is stored with its dtype from
tools/feature_constants.py. ``--feature-set`` keeps a subset from the same file and ``--start``/``--end``
keep an inclusive ``date_int`` range; the default is the training split (up to 2022).

    PYTHONPATH=. python tools/mmap/process_mmap_v3_dataset.py \
        --src-dir <by-symbol parquet dir> \
        --out-prefix <output prefix> \
        --feature-set ohlcv --end 20221231
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import polars as pl
from tqdm import tqdm

from mc_jepa.datasets.v3.indexed_dataset import MMapIndexedDatasetV3, MMapIndexedDatasetV3Builder, make_timestamps
from tools.feature_constants import FEATURE_SUBSETS, FEATURES

SYMBOL_COLUMN = "stock_symbol"
TIME_COLUMNS = ["date_int", "time_int"]
FEATURE_DTYPES = {f.name: np.dtype(f.dtype) for f in FEATURES}


def _chunk_files(src_dir: str) -> list[str]:
    files = list(Path(src_dir).glob("chunk_*.parquet"))
    assert files, f"no chunk_*.parquet under {src_dir}"
    return [str(f) for f in sorted(files, key=lambda f: int(f.stem.removeprefix("chunk_")))]


def _feature_dtypes(feature_set: str) -> dict[str, np.dtype]:
    """Feature -> storage dtype, as defined in tools/feature_constants.py."""
    names = FEATURE_SUBSETS[feature_set]
    missing = [name for name in names if name not in FEATURE_DTYPES]
    if missing:
        raise ValueError(f"no dtype defined in tools/feature_constants.py for {missing}")
    return {name: FEATURE_DTYPES[name] for name in names}


def _read_chunk(file: str, columns: list[str], start: int | None, end: int | None) -> pl.DataFrame:
    df = pl.read_parquet(file, columns=columns)
    if start is not None:
        df = df.filter(pl.col("date_int") >= start)
    if end is not None:
        df = df.filter(pl.col("date_int") <= end)
    return df


@click.command()
@click.option("--src-dir", required=True)
@click.option("--out-prefix", required=True)
@click.option("--time-interval", default=1, type=int, help="bar length in minutes, stored as metadata")
@click.option("--feature-set", default="ohlcv", type=click.Choice(list(FEATURE_SUBSETS)), help="features to keep")
@click.option("--start", default=None, type=int, help="first date_int (YYYYMMDD) to keep, inclusive")
@click.option("--end", default=20221231, type=int, help="last date_int (YYYYMMDD) to keep, inclusive")
def main(
    src_dir: str,
    out_prefix: str,
    time_interval: int,
    feature_set: str,
    start: int | None,
    end: int | None,
):
    """Convert by-symbol 1-min OHLCV parquet chunks into a v3 dataset."""
    src_dir, out_prefix = os.path.expanduser(src_dir), os.path.expanduser(out_prefix)
    if MMapIndexedDatasetV3.exists(out_prefix):
        print(f"{out_prefix} already exists, skipping")
        return
    files = _chunk_files(src_dir)
    features = _feature_dtypes(feature_set)
    columns = list(dict.fromkeys([*features, *TIME_COLUMNS, SYMBOL_COLUMN]))
    print(f"{len(files)} chunks, {len(features)} features ({feature_set}), date_int in [{start}, {end}]")

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    builder = MMapIndexedDatasetV3Builder(out_prefix, features)
    # read the next chunk while the current one is written
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_read_chunk, files[0], columns, start, end)
        for i in tqdm(range(len(files))):
            df = future.result()
            if i + 1 < len(files):
                future = pool.submit(_read_chunk, files[i + 1], columns, start, end)
            for symbol_df in df.partition_by(SYMBOL_COLUMN, maintain_order=True):
                timestamps = make_timestamps(symbol_df["date_int"].to_numpy(), symbol_df["time_int"].to_numpy())
                arrays = {name: symbol_df[name].to_numpy() for name in features}
                builder.add_symbol(symbol_df[SYMBOL_COLUMN][0], timestamps, arrays)
            del df
    builder.finalize(
        metadata={
            "source": src_dir,
            "time_interval": time_interval,
            "feature_set": feature_set,
            "start": start,
            "end": end,
        }
    )
    _report(out_prefix)


def _report(prefix: str) -> None:
    data = MMapIndexedDatasetV3(prefix)
    size = os.path.getsize(f"{prefix}.bin") + os.path.getsize(f"{prefix}.idx")
    print(
        f"{prefix}: {data.num_rows} rows, {data.num_symbols} symbols, {data.num_timestamps} timestamps, "
        f"{len(data.feature_names)} features, {size / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
