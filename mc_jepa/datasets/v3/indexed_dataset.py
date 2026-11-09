"""V3 memory-mapped panel dataset.

Differences from v2:

- Every feature has its own dtype. Features with the same dtype form one *group*, stored as a
  row-major ``[num_rows, num_group_features]`` block, so a time window costs one contiguous read
  per group and only the requested features are copied out.
- The index is self-describing: a JSON header holds feature names, dtypes, symbol names and
  user metadata.
- Rows carry a global time id, so decision times can be filtered by timestamp. v3 serves single-stock
  series only; cross-sectional panels are v4 (mc_jepa/datasets/v4).

Layout of ``{prefix}.bin`` (feature data only)::

    [group 0 block][pad][group 1 block][pad]...       each block is [num_rows, n_g] of one dtype

Layout of ``{prefix}.idx``::

    MAGIC (8B) | header_len (u64) | header JSON | pad | arrays...

    symbol_offsets  int64 [S + 1]   rows of symbol s are [symbol_offsets[s], symbol_offsets[s + 1])
    timeline        int64 [T]       sorted unique timestamps (YYYYMMDDHHMMSS) over all symbols
    time_ids        int32 [R]       time_ids[r] indexes ``timeline`` for row r

Rows are stored symbol-major and strictly increasing in time within a symbol, so a single-stock
history window is a contiguous row range.

.bin:
  [int32 block]    AAPL t0: [date, time]
                   AAPL t1: [date, time]
                   ...
                   MSFT t0: [date, time]
                   ...
  [pad]
  [float32 block]  AAPL t0: [f_a, f_b, f_c, ...]
                   AAPL t1: [f_a, f_b, f_c, ...]
                   ...
                   MSFT t0: [f_a, f_b, f_c, ...]
                   ...

"""

import json
import os
import shutil
import struct
from collections.abc import Mapping, Sequence
from functools import cached_property, lru_cache
from typing import Any

import numpy as np

MAGIC = b"QTSMIDX3"
VERSION = 3
_ALIGN = 4096

SUPPORTED_DTYPES = {
    np.dtype(name)
    for name in [
        "bool",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    ]
}


def index_file_path(prefix: str) -> str:
    return prefix + ".idx"


def data_file_path(prefix: str) -> str:
    return prefix + ".bin"


def make_timestamps(date_int: np.ndarray, time_int: np.ndarray) -> np.ndarray:
    """(YYYYMMDD, HHMMSS) -> YYYYMMDDHHMMSS as int64."""
    return np.asarray(date_int, dtype=np.int64) * 1_000_000 + np.asarray(time_int, dtype=np.int64)


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _as_index(cols: list[int]) -> np.ndarray | slice:
    """A slice when ``cols`` is a contiguous run (cheaper basic indexing), else an index array."""
    if cols == list(range(cols[0], cols[0] + len(cols))):
        return slice(cols[0], cols[0] + len(cols))
    return np.asarray(cols)


def _cast_checked(name: str, values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast ``values`` to ``dtype``, refusing silent integer overflow/truncation or float overflow."""
    if values.dtype == dtype:
        return values
    if dtype.kind in "iub" and values.size > 0:
        if values.dtype.kind == "f":
            if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
                raise ValueError(f"feature {name!r} has non-integral values and cannot be stored as {dtype}")
        if dtype.kind != "b":
            info = np.iinfo(dtype)
            lo, hi = values.min(), values.max()
            if lo < info.min or hi > info.max:
                raise ValueError(f"feature {name!r} range [{lo}, {hi}] overflows {dtype}")
    out = values.astype(dtype)
    if dtype.kind == "f" and np.any(np.isinf(out) & ~np.isinf(values)):
        raise ValueError(f"feature {name!r} overflows {dtype}")
    return out


class MMapIndexedDatasetV3Builder:
    """Streams symbols into a v3 dataset.

    Example::

        builder = MMapIndexedDatasetV3Builder(prefix, {"close": np.float32, "volume": np.uint32})
        for symbol, df in ...:
            builder.add_symbol(symbol, timestamps, {"close": df["close"], "volume": df["volume"]})
        builder.finalize(metadata={"time_interval": 30})

    Each feature group is streamed to its own temporary file, which ``finalize`` concatenates
    into ``{prefix}.bin`` and removes, so memory use stays bounded by one symbol.
    """

    def __init__(self, prefix: str, features: Mapping[str, Any] | Sequence[tuple[str, Any]]):
        items = list(features.items()) if isinstance(features, Mapping) else list(features)
        self._prefix = prefix
        self._feature_names = [name for name, _ in items]
        self._feature_dtypes = [np.dtype(dtype) for _, dtype in items]
        if len(set(self._feature_names)) != len(self._feature_names):
            raise ValueError("duplicate feature names")
        for name, dtype in zip(self._feature_names, self._feature_dtypes, strict=True):
            if dtype not in SUPPORTED_DTYPES:
                raise ValueError(f"unsupported dtype {dtype} for feature {name!r}")

        # dtype groups keep the first-appearance order of their dtype and the user order of features
        self._groups: dict[np.dtype, list[int]] = {}
        for i, dtype in enumerate(self._feature_dtypes):
            self._groups.setdefault(dtype, []).append(i)

        self._group_files = {dtype: open(self._tmp_path(dtype.name), "wb") for dtype in self._groups}
        self._timestamp_file = open(self._tmp_path("timestamps"), "wb")
        self._symbols: list[str] = []
        self._symbol_offsets = [0]
        self._timeline = np.empty(0, dtype=np.int64)

    def _tmp_path(self, tag: str) -> str:
        return f"{self._prefix}.{tag}.tmp"

    def add_symbol(
        self,
        symbol: str,
        timestamps: np.ndarray,
        features: Mapping[str, np.ndarray] | np.ndarray,
    ) -> None:
        """Append one symbol.

        Args:
            symbol: unique symbol name.
            timestamps: int64 YYYYMMDDHHMMSS per row, strictly increasing.
            features: ``{name: 1-D array}`` (e.g. a polars/pandas frame via ``dict(df)``) or a 2-D
                array whose columns follow the builder's feature order.
        """
        timestamps = np.asarray(timestamps, dtype=np.int64)
        n = len(timestamps)
        if n == 0:
            return
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"timestamps of {symbol!r} must be strictly increasing")
        if isinstance(features, np.ndarray):
            if features.shape != (n, len(self._feature_names)):
                raise ValueError(f"{symbol!r}: expected shape {(n, len(self._feature_names))}, got {features.shape}")
            columns = [features[:, i] for i in range(features.shape[1])]
        else:
            columns = [np.asarray(features[name]) for name in self._feature_names]
            for name, col in zip(self._feature_names, columns, strict=True):
                if col.shape != (n,):
                    raise ValueError(f"{symbol!r}: feature {name!r} has shape {col.shape}, expected {(n,)}")

        for dtype, feature_idx in self._groups.items():
            block = np.empty((n, len(feature_idx)), dtype=dtype)
            for j, i in enumerate(feature_idx):
                block[:, j] = _cast_checked(self._feature_names[i], columns[i], dtype)
            self._group_files[dtype].write(block.tobytes(order="C"))
        self._timestamp_file.write(timestamps.tobytes())

        self._symbols.append(symbol)
        self._symbol_offsets.append(self._symbol_offsets[-1] + n)
        self._timeline = np.union1d(self._timeline, timestamps)

    def finalize(self, metadata: dict[str, Any] | None = None) -> None:
        for f in self._group_files.values():
            f.close()
        self._timestamp_file.close()
        if len(set(self._symbols)) != len(self._symbols):
            raise ValueError("duplicate symbol names")

        num_rows = self._symbol_offsets[-1]
        groups = self._write_data_file(num_rows)
        arrays = self._build_index_arrays(num_rows)

        header_arrays = {}
        offset = 0
        for name, arr in arrays.items():
            header_arrays[name] = {"dtype": arr.dtype.name, "shape": list(arr.shape), "offset": offset}
            offset = _align(offset + arr.nbytes)
        header = {
            "version": VERSION,
            "num_rows": num_rows,
            "features": [
                {"name": n, "dtype": d.name} for n, d in zip(self._feature_names, self._feature_dtypes, strict=True)
            ],
            "groups": groups,
            "symbols": self._symbols,
            "arrays": header_arrays,
            "metadata": metadata or {},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        array_base = _align(len(MAGIC) + 8 + len(header_bytes))

        with open(index_file_path(self._prefix), "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for name, arr in arrays.items():
                f.seek(array_base + header_arrays[name]["offset"])
                f.write(np.ascontiguousarray(arr).tobytes())
        os.remove(self._tmp_path("timestamps"))

    def _write_data_file(self, num_rows: int) -> list[dict[str, Any]]:
        groups = []
        with open(data_file_path(self._prefix), "wb") as out:
            for dtype, feature_idx in self._groups.items():
                offset = out.tell()
                tmp = self._tmp_path(dtype.name)
                with open(tmp, "rb") as f:
                    shutil.copyfileobj(f, out, length=64 * 1024 * 1024)
                os.remove(tmp)
                assert out.tell() - offset == num_rows * len(feature_idx) * dtype.itemsize
                out.write(b"\0" * (_align(out.tell()) - out.tell()))
                groups.append(
                    {
                        "dtype": dtype.name,
                        "features": [self._feature_names[i] for i in feature_idx],
                        "offset": offset,
                    }
                )
        return groups

    def _build_index_arrays(self, num_rows: int) -> dict[str, np.ndarray]:
        timeline = self._timeline
        if len(timeline) >= np.iinfo(np.int32).max:
            raise ValueError("timeline too long for int32 time ids")
        symbol_offsets = np.asarray(self._symbol_offsets, dtype=np.int64)
        timestamps = np.fromfile(self._tmp_path("timestamps"), dtype=np.int64)
        time_ids = np.searchsorted(timeline, timestamps).astype(np.int32)
        assert len(time_ids) == num_rows
        return {"symbol_offsets": symbol_offsets, "timeline": timeline, "time_ids": time_ids}


class MMapIndexedDatasetV3:
    """Read-only v3 dataset. Memory maps open lazily, so instances are cheap to pickle to workers."""

    def __init__(self, prefix: str, warmup: bool = False):
        self._prefix = prefix
        with open(index_file_path(prefix), "rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{index_file_path(prefix)} is not a v3 index file")
            (header_len,) = struct.unpack("<Q", f.read(8))
            self._header = json.loads(f.read(header_len).decode("utf-8"))
        if self._header["version"] != VERSION:
            raise ValueError(f"unsupported index version {self._header['version']}")
        self._array_base = _align(len(MAGIC) + 8 + header_len)

        self.num_rows: int = self._header["num_rows"]
        self.feature_names: list[str] = [f["name"] for f in self._header["features"]]
        self.feature_dtypes: dict[str, np.dtype] = {f["name"]: np.dtype(f["dtype"]) for f in self._header["features"]}
        self.symbols: list[str] = self._header["symbols"]
        self.metadata: dict[str, Any] = self._header["metadata"]
        # feature name -> (group index, column in group)
        self._feature_loc = {
            name: (g, j) for g, group in enumerate(self._header["groups"]) for j, name in enumerate(group["features"])
        }
        if warmup:
            for path in (data_file_path(prefix), index_file_path(prefix)):
                with open(path, "rb") as f:
                    while f.read(100 * 1024 * 1024):
                        pass

    @staticmethod
    def exists(prefix: str) -> bool:
        return os.path.exists(index_file_path(prefix)) and os.path.exists(data_file_path(prefix))

    def __getstate__(self):
        # drop memory maps; they are re-opened lazily in the worker process
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, np.ndarray) and k != "_blocks"}

    def __len__(self) -> int:
        return self.num_rows

    # ---- lazily mapped arrays -------------------------------------------------------------------

    def _index_array(self, name: str) -> np.ndarray:
        spec = self._header["arrays"][name]
        return np.memmap(
            index_file_path(self._prefix),
            dtype=spec["dtype"],
            mode="r",
            offset=self._array_base + spec["offset"],
            shape=tuple(spec["shape"]),
        )

    @cached_property
    def _blocks(self) -> list[np.ndarray]:
        return [
            np.memmap(
                data_file_path(self._prefix),
                dtype=group["dtype"],
                mode="r",
                offset=group["offset"],
                shape=(self.num_rows, len(group["features"])),
            )
            for group in self._header["groups"]
        ]

    @cached_property
    def symbol_offsets(self) -> np.ndarray:
        return np.array(self._index_array("symbol_offsets"))

    @cached_property
    def timeline(self) -> np.ndarray:
        return np.array(self._index_array("timeline"))

    @cached_property
    def time_ids(self) -> np.ndarray:
        return self._index_array("time_ids")

    # ---- symbol / time lookups ------------------------------------------------------------------

    @property
    def num_symbols(self) -> int:
        return len(self.symbols)

    @property
    def num_timestamps(self) -> int:
        return len(self.timeline)

    def symbol_range(self, symbol_id: int) -> tuple[int, int]:
        """Half-open row range of one symbol."""
        return int(self.symbol_offsets[symbol_id]), int(self.symbol_offsets[symbol_id + 1])

    def symbol_ranges(self) -> list[tuple[int, int]]:
        return list(zip(self.symbol_offsets[:-1].tolist(), self.symbol_offsets[1:].tolist(), strict=True))

    def symbol_of_rows(self, rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.symbol_offsets, rows, side="right") - 1

    def time_id(self, timestamp: int) -> int:
        t = int(np.searchsorted(self.timeline, timestamp))
        if t == len(self.timeline) or self.timeline[t] != timestamp:
            raise KeyError(f"timestamp {timestamp} not in dataset")
        return t

    def timestamps(self, start: int, stop: int) -> np.ndarray:
        return self.timeline[self.time_ids[start:stop]]

    # ---- feature reads --------------------------------------------------------------------------

    @lru_cache(maxsize=32)  # noqa: B019 - keyed on a handful of feature tuples per dataset
    def _read_plan(self, features: tuple[str, ...]) -> list[tuple[int, np.ndarray | slice, np.ndarray | slice]]:
        """Per group: (group index, columns to take from the group, columns to fill in the output)."""
        by_group: dict[int, tuple[list[int], list[int]]] = {}
        for out_col, name in enumerate(features):
            if name not in self._feature_loc:
                raise KeyError(f"feature {name!r} not in dataset")
            g, j = self._feature_loc[name]
            src, dst = by_group.setdefault(g, ([], []))
            src.append(j)
            dst.append(out_col)
        return [(g, _as_index(src), _as_index(dst)) for g, (src, dst) in by_group.items()]

    def _features_key(self, features: Sequence[str] | None) -> tuple[str, ...]:
        return tuple(self.feature_names) if features is None else tuple(features)

    def read(self, start: int, stop: int, features: Sequence[str] | None = None, dtype=np.float32) -> np.ndarray:
        """Rows ``[start, stop)`` as a ``[stop - start, F]`` array, features in the requested order."""
        features = self._features_key(features)
        out = np.empty((stop - start, len(features)), dtype=dtype)
        for g, src, dst in self._read_plan(features):
            out[:, dst] = self._blocks[g][start:stop, src]
        return out

    def gather(self, rows: np.ndarray, features: Sequence[str] | None = None, dtype=np.float32) -> np.ndarray:
        """Arbitrary rows (any shape) as a ``[*rows.shape, F]`` array. Sorted rows read fastest."""
        rows = np.asarray(rows, dtype=np.int64)
        features = self._features_key(features)
        flat = rows.reshape(-1)
        out = np.empty((flat.size, len(features)), dtype=dtype)
        for g, src, dst in self._read_plan(features):
            out[:, dst] = self._blocks[g][flat][:, src]
        return out.reshape(*rows.shape, len(features))

    # ---- windows --------------------------------------------------------------------------------

    def window_mask(self, anchor_rows: np.ndarray, lookback: int, lookahead: int = 0) -> np.ndarray:
        """Whether ``[anchor - lookback + 1, anchor + lookahead]`` stays inside the anchor's symbol."""
        assert lookback >= 1 and lookahead >= 0
        anchor_rows = np.asarray(anchor_rows, dtype=np.int64)
        symbol_ids = self.symbol_of_rows(anchor_rows)
        first = anchor_rows - (lookback - 1)
        last = anchor_rows + lookahead
        return (first >= self.symbol_offsets[symbol_ids]) & (last < self.symbol_offsets[symbol_ids + 1])
