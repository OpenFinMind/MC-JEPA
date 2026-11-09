"""Cut a v4 dataset (mc_jepa/datasets/v4/indexed_dataset.py) down to a date range, e.g. to keep one split in memory
at a time.

The cut starts at the block holding the first bar on or after ``--start``, so whole blocks are copied as they are
(the first block may add a few earlier bars), and ends after the last bar before ``--end``. Symbols, features and
dtypes are kept; one grid of the cut is held in memory at a time (~1 GB per float32 feature-year of 1-min bars).

    PYTHONPATH=. python tools/mmap/slice_mmap_v4_dataset.py \
        --prefix <v4 mmap prefix> \
        --out-prefix <output prefix> --end 20230101
"""

import os

import click
import numpy as np
from tqdm import tqdm

from mc_jepa.datasets.v4.indexed_dataset import VALID, MMapIndexedDatasetV4, MMapIndexedDatasetV4Builder


@click.command()
@click.option("--prefix", required=True, help="source v4 prefix (without .bin/.idx)")
@click.option("--out-prefix", required=True)
@click.option("--start", type=int, help="first date kept (YYYYMMDD), inclusive [default: the first]")
@click.option("--end", type=int, help="dates before this (YYYYMMDD) [default: all]")
def main(prefix: str, out_prefix: str, start: int | None, end: int | None):
    """Copy the blocks covering ``[start, end)`` of every grid into a new v4 dataset."""
    prefix, out_prefix = os.path.expanduser(prefix), os.path.expanduser(out_prefix)
    data = MMapIndexedDatasetV4(prefix)
    timeline, B = data.timeline, data.block_size
    t0 = 0 if start is None else int(np.searchsorted(timeline, start * 1_000_000))
    t1 = len(timeline) if end is None else int(np.searchsorted(timeline, end * 1_000_000))
    assert t0 < t1, f"no bars in [{start}, {end})"
    t0 = t0 // B * B  # whole blocks
    k0, k1 = t0 // B, (t1 - 1) // B + 1
    builder = MMapIndexedDatasetV4Builder(out_prefix, data.symbols, timeline[t0:t1], block_size=B)
    print(f"{prefix}: time ids [{t0}, {t1}) = {timeline[t0]} .. {timeline[t1 - 1]}, blocks [{k0}, {k1})")
    tail = t1 - (k1 - 1) * B  # time ids of the last block inside the cut
    for name in tqdm([VALID, *data.feature_names], desc="grids"):
        grid = data._read_blocks(name, k0, k1)
        grid[-1, :, tail:] = 0  # padding past the last bar, as the builder writes it
        if name == VALID:
            builder.add_valid(grid)
        else:
            builder.add_feature(name, grid)
    builder.finalize(metadata={**data.metadata, "sliced_from": prefix, "start": start, "end": end})

    out = MMapIndexedDatasetV4(out_prefix)  # spot check: a few random cells match the source
    rng = np.random.default_rng(0)
    for name in [VALID, *data.feature_names[:4]]:
        s, t = int(rng.integers(data.num_symbols)), int(rng.integers(t0, t1 - 64))
        cells = out.grid(name, t - t0, t - t0 + 64, np.array([s]))
        assert np.array_equal(cells, data.grid(name, t, t + 64, np.array([s]))), (name, s, t)
    gib = os.path.getsize(out_prefix + ".bin") / 2**30
    print(f"wrote {out_prefix}: {out.num_timestamps} timestamps, {gib:.0f} GiB")


if __name__ == "__main__":
    main()
