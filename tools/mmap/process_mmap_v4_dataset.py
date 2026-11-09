"""Convert the by-symbol 1-min OHLCV parquet chunks into one v4 dataset (mc_jepa/datasets/v4/indexed_dataset.py).

Each ``chunk_*.parquet`` holds whole symbols, rows grouped by ``stock_symbol`` and sorted by
(``date_int``, ``time_int``) within a symbol. Each feature is stored with its dtype from
tools/feature_constants.py. ``--feature-set`` keeps a subset from the same file and ``--start``/``--end``
keep an inclusive ``date_int`` range; the default is the training split (up to 2022).

v4 grids are dense in time x symbol, so they are filled in memory: a first pass reads the symbol and
time columns to build the timeline and every bar's grid cell, then each pass reads a group of feature
columns that fits in ``--max-memory-gb`` and writes their grids. One grid is
``ceil(T / block_size) * block_size * S * itemsize`` bytes (~9.4 GB per float32 feature for 1-min
2015-2022).

    PYTHONPATH=. python tools/mmap/process_mmap_v4_dataset.py \
        --src-dir <by-symbol parquet dir> \
        --out-prefix <output prefix> \
        --feature-set ohlcv --end 20251231
"""

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import polars as pl
from tqdm import tqdm

from mc_jepa.datasets.v4.indexed_dataset import (
    MMapIndexedDatasetV4,
    MMapIndexedDatasetV4Builder,
    cast_checked,
    make_timestamps,
)
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
    df = pl.read_parquet(file, columns=list(dict.fromkeys([*columns, "date_int"])))
    if start is not None:
        df = df.filter(pl.col("date_int") >= start)
    if end is not None:
        df = df.filter(pl.col("date_int") <= end)
    return df


def _iter_chunks(files: list[str], read: Callable[[str], pl.DataFrame], desc: str) -> Iterator[pl.DataFrame]:
    """Yield ``read(file)`` per file, reading the next file while the caller processes the current one."""
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(read, files[0])
        for i in tqdm(range(len(files)), desc=desc):
            df = future.result()
            if i + 1 < len(files):
                future = pool.submit(read, files[i + 1])
            yield df


def _scan_bars(files: list[str], start: int | None, end: int | None):
    """First pass: symbol names, and per chunk every bar's symbol id and timestamp."""
    symbols: list[str] = []
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    read = lambda f: _read_chunk(f, [SYMBOL_COLUMN, *TIME_COLUMNS], start, end)  # noqa: E731
    for df in _iter_chunks(files, read, "scan"):
        runs = df[SYMBOL_COLUMN].rle()
        if len(runs) != df[SYMBOL_COLUMN].n_unique():
            raise ValueError("rows of a symbol must be contiguous within a chunk")
        run_ids = df.select(pl.col(SYMBOL_COLUMN).rle_id()).to_series().to_numpy().astype(np.int64)
        timestamps = make_timestamps(df["date_int"].to_numpy(), df["time_int"].to_numpy())
        same_symbol = run_ids[1:] == run_ids[:-1]
        if np.any(np.diff(timestamps)[same_symbol] <= 0):
            raise ValueError("timestamps must be strictly increasing within a symbol")
        chunks.append((run_ids + len(symbols), timestamps))
        symbols.extend(runs.struct.field("value").to_list())
    if len(set(symbols)) != len(symbols):
        raise ValueError("a symbol appears in more than one chunk")
    return symbols, chunks


def _group_features(features: dict[str, np.dtype], nbytes: Callable[[np.dtype], int], budget: int):
    """Split features into consecutive groups whose grids fit in ``budget`` bytes (at least one each)."""
    groups: list[list[str]] = [[]]
    used = 0
    for name, dtype in features.items():
        if groups[-1] and used + nbytes(dtype) > budget:
            groups.append([])
            used = 0
        groups[-1].append(name)
        used += nbytes(dtype)
    return groups


@click.command()
@click.option("--src-dir", required=True)
@click.option("--out-prefix", required=True)
@click.option("--time-interval", default=1, type=int, help="bar length in minutes, stored as metadata")
@click.option("--feature-set", default="ohlcv", type=click.Choice(list(FEATURE_SUBSETS)), help="features to keep")
@click.option("--start", default=None, type=int, help="first date_int (YYYYMMDD) to keep, inclusive")
@click.option("--end", default=20221231, type=int, help="last date_int (YYYYMMDD) to keep, inclusive")
@click.option("--block-size", default=64, type=int, help="timestamps per storage block")
@click.option("--max-memory-gb", default=128.0, type=float, help="grid memory per feature pass")
def main(
    src_dir: str,
    out_prefix: str,
    time_interval: int,
    feature_set: str,
    start: int | None,
    end: int | None,
    block_size: int,
    max_memory_gb: float,
):
    """Convert by-symbol 1-min OHLCV parquet chunks into a v4 dataset."""
    src_dir, out_prefix = os.path.expanduser(src_dir), os.path.expanduser(out_prefix)
    if MMapIndexedDatasetV4.exists(out_prefix):
        print(f"{out_prefix} already exists, skipping")
        return
    files = _chunk_files(src_dir)
    features = _feature_dtypes(feature_set)
    print(f"{len(files)} chunks, {len(features)} features ({feature_set}), date_int in [{start}, {end}]")

    symbols, chunks = _scan_bars(files, start, end)
    timeline = np.unique(np.concatenate([np.unique(ts) for _, ts in chunks]))
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    builder = MMapIndexedDatasetV4Builder(out_prefix, symbols, timeline, block_size)
    cells = [builder.cell_index(np.searchsorted(timeline, ts), sids) for sids, ts in chunks]
    num_bars = sum(len(c) for c in cells)
    del chunks
    print(
        f"{num_bars} bars, {len(symbols)} symbols, {len(timeline)} timestamps; grid {builder.grid_shape}, "
        f"{num_bars / np.prod(builder.grid_shape):.1%} filled"
    )

    valid = builder.new_grid(np.uint8)
    for c in cells:
        valid.reshape(-1)[c] = 1
    builder.add_valid(valid)
    del valid

    groups = _group_features(features, builder.grid_nbytes, int(max_memory_gb * 2**30))
    for g, names in enumerate(groups):
        grids = {name: builder.new_grid(features[name]) for name in names}
        read = lambda f, names=names: _read_chunk(f, names, start, end)  # noqa: E731
        for df, c in zip(_iter_chunks(files, read, f"features {g + 1}/{len(groups)}"), cells, strict=True):
            assert len(df) == len(c)
            for name, grid in grids.items():
                grid.reshape(-1)[c] = cast_checked(name, df[name].to_numpy(), grid.dtype)
        for name in names:
            builder.add_feature(name, grids.pop(name))

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
    data = MMapIndexedDatasetV4(prefix)
    size = os.path.getsize(f"{prefix}.bin") + os.path.getsize(f"{prefix}.idx")
    print(
        f"{prefix}: {data.num_symbols} symbols, {data.num_timestamps} timestamps, "
        f"{len(data.feature_names)} features, block {data.block_size}, {size / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
