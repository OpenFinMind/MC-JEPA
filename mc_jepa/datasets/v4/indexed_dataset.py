"""V4 time-major memory-mapped panel dataset.

v3 stores rows symbol-major, so the cross-section of N stocks around one timestamp costs N random
reads per dtype group, which is latency bound on network storage (~16 s per 3000-stock panel on NFS).
v4 stores every feature as a dense time x symbol grid, so the same panel is one contiguous read per
feature.

- The grid spans the sorted union ``timeline`` of all timestamps (T) and all symbols (S). Cells
  without a bar hold 0 and are 0 in ``valid``.
- The time axis is cut into blocks of ``block_size`` (B) timestamps and each grid is stored as
  ``[K, S, B]`` with ``K = ceil(T / B)`` (the tail of the last block is padding). Any time window of
  all stocks is the contiguous byte range of the blocks it touches; one stock's history is one
  ``B``-long run per block.

Layout of ``{prefix}.bin`` (every grid starts 4 KiB aligned)::

    [valid uint8 [K, S, B]][pad][feature 0 [K, S, B]][pad][feature 1 [K, S, B]][pad]...

Layout of ``{prefix}.idx``::

    MAGIC (8B) | header_len (u64) | header JSON | pad | timeline int64 [T]

The header JSON holds the block size, symbol names, per-feature dtype and byte offset, and user
metadata.
"""

import json
import math
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

MAGIC = b"QTSMIDX4"
VERSION = 4
_ALIGN = 4096
VALID = "valid"

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


def cast_checked(name: str, values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast ``values`` to ``dtype``, refusing silent integer overflow/truncation or float overflow."""
    values = np.asarray(values)
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


def _pread_into(fd: int, buf: np.ndarray, offset: int) -> None:
    view = memoryview(buf.reshape(-1).view(np.uint8))
    while len(view):
        n = os.preadv(fd, [view], offset)
        if n == 0:
            raise EOFError(f"short read at offset {offset}")
        view, offset = view[n:], offset + n


class MMapIndexedDatasetV4Builder:
    """Writes a v4 dataset one full grid at a time.

    Example::

        builder = MMapIndexedDatasetV4Builder(prefix, symbols, timeline, block_size=64)
        cells = builder.cell_index(time_ids, symbol_ids)  # one per bar
        valid = builder.new_grid(np.uint8)
        valid.reshape(-1)[cells] = 1
        builder.add_valid(valid)
        for name, dtype in features.items():
            grid = builder.new_grid(dtype)
            grid.reshape(-1)[cells] = cast_checked(name, values[name], grid.dtype)
            builder.add_feature(name, grid)
        builder.finalize(metadata={"time_interval": 1})

    Files are written under ``.tmp`` names and renamed by ``finalize``, so an interrupted build
    never looks complete to ``MMapIndexedDatasetV4.exists``.
    """

    def __init__(self, prefix: str, symbols: Sequence[str], timeline: np.ndarray, block_size: int = 64):
        timeline = np.asarray(timeline, dtype=np.int64)
        if len(timeline) == 0 or np.any(np.diff(timeline) <= 0):
            raise ValueError("timeline must be non-empty and strictly increasing")
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate symbol names")
        assert block_size >= 1
        self._prefix = prefix
        self.symbols = list(symbols)
        self.timeline = timeline
        self.block_size = block_size
        self.num_blocks = -(-len(timeline) // block_size)
        self.grid_shape = (self.num_blocks, len(self.symbols), block_size)
        self._data = open(self._tmp_path(data_file_path(prefix)), "wb")
        self._features: list[dict[str, Any]] = []
        self._valid_offset: int | None = None

    @staticmethod
    def _tmp_path(path: str) -> str:
        return path + ".tmp"

    def cell_index(self, time_ids: np.ndarray, symbol_ids: np.ndarray) -> np.ndarray:
        """Flat index into a ``[K, S, B]`` grid of each (time id, symbol id) cell."""
        time_ids = np.asarray(time_ids, dtype=np.int64)
        symbol_ids = np.asarray(symbol_ids, dtype=np.int64)
        block, pos = np.divmod(time_ids, self.block_size)
        return (block * len(self.symbols) + symbol_ids) * self.block_size + pos

    def new_grid(self, dtype) -> np.ndarray:
        return np.zeros(self.grid_shape, dtype=dtype)

    def grid_nbytes(self, dtype) -> int:
        return int(np.prod(self.grid_shape)) * np.dtype(dtype).itemsize

    def _write_grid(self, grid: np.ndarray) -> int:
        if grid.shape != self.grid_shape:
            raise ValueError(f"expected grid shape {self.grid_shape}, got {grid.shape}")
        offset = self._data.tell()
        np.ascontiguousarray(grid).tofile(self._data)
        self._data.write(b"\0" * (_align(self._data.tell()) - self._data.tell()))
        return offset

    def add_valid(self, valid: np.ndarray) -> None:
        """``valid[k, s, j]`` is nonzero where symbol s has a bar at time ``k * B + j``."""
        if self._valid_offset is not None:
            raise ValueError("valid already added")
        self._valid_offset = self._write_grid(valid.astype(np.uint8, copy=False))

    def add_feature(self, name: str, grid: np.ndarray) -> None:
        if name == VALID or name in {f["name"] for f in self._features}:
            raise ValueError(f"duplicate or reserved feature name {name!r}")
        if grid.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {grid.dtype} for feature {name!r}")
        self._features.append({"name": name, "dtype": grid.dtype.name, "offset": self._write_grid(grid)})

    def finalize(self, metadata: dict[str, Any] | None = None) -> None:
        if self._valid_offset is None:
            raise ValueError("add_valid must be called before finalize")
        self._data.close()
        header = {
            "version": VERSION,
            "block_size": self.block_size,
            "num_timestamps": len(self.timeline),
            "symbols": self.symbols,
            "valid_offset": self._valid_offset,
            "features": self._features,
            "metadata": metadata or {},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        index_path = index_file_path(self._prefix)
        with open(self._tmp_path(index_path), "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.seek(_align(len(MAGIC) + 8 + len(header_bytes)))
            f.write(self.timeline.tobytes())
        data_path = data_file_path(self._prefix)
        os.replace(self._tmp_path(data_path), data_path)
        os.replace(self._tmp_path(index_path), index_path)


@dataclass
class Panel:
    """Multi-stock windows around one timestamp.

    ``values[i]`` is symbol ``symbol_ids[i]``'s own last ``lookback`` bars up to and including the anchor
    timestamp, then its next ``lookahead`` bars. With ``span_ratio == 1`` these are exactly the time ids
    ``[time_id - lookback + 1, time_id + lookahead]``; with a larger ratio a stock may skip timestamps (halts,
    minutes without trades), so the same position ``k`` can be a different time id for different stocks.
    """

    time_id: int
    timestamp: int
    symbol_ids: np.ndarray  # int64 [N]
    values: np.ndarray  # [N, lookback + lookahead, F]


class MMapIndexedDatasetV4:
    """Read-only v4 dataset. Reads use ``pread`` on a file opened lazily per process, so instances
    are cheap to pickle to DataLoader workers."""

    def __init__(self, prefix: str, warmup: bool = False):
        self._prefix = prefix
        with open(index_file_path(prefix), "rb") as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{index_file_path(prefix)} is not a v4 index file")
            (header_len,) = struct.unpack("<Q", f.read(8))
            self._header = json.loads(f.read(header_len).decode("utf-8"))
        if self._header["version"] != VERSION:
            raise ValueError(f"unsupported index version {self._header['version']}")
        self._timeline_offset = _align(len(MAGIC) + 8 + header_len)

        self.block_size: int = self._header["block_size"]
        self.num_timestamps: int = self._header["num_timestamps"]
        self.symbols: list[str] = self._header["symbols"]
        self.metadata: dict[str, Any] = self._header["metadata"]
        self.feature_names: list[str] = [f["name"] for f in self._header["features"]]
        self.feature_dtypes: dict[str, np.dtype] = {f["name"]: np.dtype(f["dtype"]) for f in self._header["features"]}
        self.num_blocks = -(-self.num_timestamps // self.block_size)
        # grid name -> (dtype, byte offset in .bin)
        self._grids = {f["name"]: (np.dtype(f["dtype"]), f["offset"]) for f in self._header["features"]}
        self._grids[VALID] = (np.dtype(np.uint8), self._header["valid_offset"])
        self._fd: int | None = None
        self._fd_pid: int | None = None
        self._maps: dict[str, np.memmap] = {}  # grid name -> [K, S, B] memmap, per process like _fd
        if warmup:
            for path in (data_file_path(prefix), index_file_path(prefix)):
                with open(path, "rb") as f:
                    while f.read(100 * 1024 * 1024):
                        pass

    @staticmethod
    def exists(prefix: str) -> bool:
        return os.path.exists(index_file_path(prefix)) and os.path.exists(data_file_path(prefix))

    def __getstate__(self):
        # file descriptors do not survive pickling; they are re-opened lazily in the worker process
        state = dict(self.__dict__)
        state.pop("timeline", None)
        state["_fd"] = state["_fd_pid"] = None
        state["_maps"] = {}
        return state

    def __del__(self):
        if getattr(self, "_fd", None) is not None and self._fd_pid == os.getpid():
            os.close(self._fd)

    @property
    def num_symbols(self) -> int:
        return len(self.symbols)

    @cached_property
    def timeline(self) -> np.ndarray:
        return np.fromfile(
            index_file_path(self._prefix), dtype=np.int64, count=self.num_timestamps, offset=self._timeline_offset
        )

    def time_id(self, timestamp: int) -> int:
        t = int(np.searchsorted(self.timeline, timestamp))
        if t == len(self.timeline) or self.timeline[t] != timestamp:
            raise KeyError(f"timestamp {timestamp} not in dataset")
        return t

    # ---- grid reads -----------------------------------------------------------------------------

    def _file(self) -> int:
        if self._fd is None or self._fd_pid != os.getpid():
            self._fd = os.open(data_file_path(self._prefix), os.O_RDONLY)
            self._fd_pid = os.getpid()
            self._maps = {}
        return self._fd

    def _read_blocks(self, name: str, k0: int, k1: int) -> np.ndarray:
        """Blocks ``[k0, k1)`` of one grid as ``[k1 - k0, S, B]``, in a single read."""
        dtype, offset = self._grids[name]
        out = np.empty((k1 - k0, self.num_symbols, self.block_size), dtype=dtype)
        _pread_into(self._file(), out, offset + k0 * self.num_symbols * self.block_size * dtype.itemsize)
        return out

    def _map(self, name: str) -> np.memmap:
        self._file()  # resets the maps in a new process
        if name not in self._maps:
            dtype, offset = self._grids[name]
            shape = (self.num_blocks, self.num_symbols, self.block_size)
            self._maps[name] = np.memmap(data_file_path(self._prefix), dtype, "r", offset, shape)
        return self._maps[name]

    def _read_runs(self, name: str, k0: int, k1: int, symbol_ids: np.ndarray) -> np.ndarray:
        """Blocks ``[k0, k1)`` of one grid for a few symbols as ``[k1 - k0, len(symbol_ids), B]``: a gather from a
        memmap of the grid, since one ``pread`` per ``B``-long run is syscall bound (~1250 per feature for a
        32-stock 960-bar panel at span 2)."""
        return np.asarray(self._map(name)[k0:k1, symbol_ids])

    def grid(self, name: str, t0: int, t1: int, symbol_ids: np.ndarray | None = None, dtype=None) -> np.ndarray:
        """Time ids ``[t0, t1)`` of a feature (or ``"valid"``) as ``[S or len(symbol_ids), t1 - t0]``.

        A subset of at most ``S / 16`` symbols is read run by run instead of as whole blocks."""
        assert 0 <= t0 < t1 <= self.num_timestamps, (t0, t1)
        B = self.block_size
        k0, k1 = t0 // B, (t1 - 1) // B + 1
        if symbol_ids is not None and len(symbol_ids) * 16 <= self.num_symbols:
            blocks = self._read_runs(name, k0, k1, symbol_ids)
        else:
            blocks = self._read_blocks(name, k0, k1)
            if symbol_ids is not None:
                blocks = blocks[:, symbol_ids]
        out = blocks.transpose(1, 0, 2).reshape(blocks.shape[1], -1)[:, t0 - k0 * B : t1 - k0 * B]
        return out if dtype is None else out.astype(dtype, copy=False)

    def valid(self, t0: int, t1: int, symbol_ids: np.ndarray | None = None) -> np.ndarray:
        return self.grid(VALID, t0, t1, symbol_ids).astype(bool)

    def read(self, symbol_id: int, t0: int, t1: int, features: Sequence[str] | None = None, dtype=np.float32):
        """One symbol over time ids ``[t0, t1)`` as ``[t1 - t0, F]`` (missing bars read as 0).

        Costs one ``B``-long read per block and feature; use ``grid`` for many symbols at once."""
        assert 0 <= t0 < t1 <= self.num_timestamps, (t0, t1)
        features = self.feature_names if features is None else list(features)
        B, S = self.block_size, self.num_symbols
        k0, k1 = t0 // B, (t1 - 1) // B + 1
        out = np.empty((t1 - t0, len(features)), dtype=dtype)
        for j, name in enumerate(features):
            grid_dtype, offset = self._grids[name]
            run = np.empty((k1 - k0, B), dtype=grid_dtype)
            for i, k in enumerate(range(k0, k1)):
                _pread_into(self._file(), run[i], offset + ((k * S + symbol_id) * B) * grid_dtype.itemsize)
            out[:, j] = run.reshape(-1)[t0 - k0 * B : t1 - k0 * B]
        return out

    # ---- panels ---------------------------------------------------------------------------------
    #
    # A symbol is eligible at anchor time id t if it has a bar at t, at least ``lookback`` bars in the
    # ``span(lookback)`` time ids ending at t and at least ``lookahead`` bars in the ``span(lookahead)`` time
    # ids after t, where ``span(n) = ceil(n * span_ratio)``. ``span_ratio = 1`` asks for a bar at every time
    # id of ``[t - lookback + 1, t + lookahead]``; a larger ratio tolerates halts and missing minutes.

    @staticmethod
    def _spans(lookback: int, lookahead: int, span_ratio: float) -> tuple[int, int]:
        assert lookback >= 1 and lookahead >= 0 and span_ratio >= 1.0
        return math.ceil(lookback * span_ratio), math.ceil(lookahead * span_ratio)

    def count_eligible(
        self,
        lookback: int,
        lookahead: int = 0,
        time_mask: np.ndarray | None = None,
        symbol_mask: np.ndarray | None = None,
        span_ratio: float = 1.0,
        chunk: int = 8192,
    ) -> np.ndarray:
        """Per time id, the number of eligible symbols (within ``symbol_mask``); 0 where ``time_mask`` is
        False."""
        back, ahead = self._spans(lookback, lookahead, span_ratio)
        counts = np.zeros(self.num_timestamps, dtype=np.int64)
        lo, hi = back - 1, self.num_timestamps - ahead  # anchors whose spans are in range
        for c0 in range(lo, hi, chunk):
            c1 = min(c0 + chunk, hi)
            anchors = np.arange(c0, c1) if time_mask is None else np.flatnonzero(time_mask[c0:c1]) + c0
            if len(anchors) == 0:
                continue
            valid = self.valid(c0 - back + 1, c1 + ahead)  # [S, c1 - c0 + back + ahead - 1]
            if symbol_mask is not None:
                valid = valid[symbol_mask]
            bars = np.zeros((valid.shape[0], valid.shape[1] + 1), dtype=np.int32)
            np.cumsum(valid, axis=1, out=bars[:, 1:])
            at = anchors - c0 + back  # bars[:, at] counts the bars up to and including anchor a
            ok = valid[:, at - 1] & (bars[:, at] - bars[:, at - back] >= lookback)
            ok &= bars[:, at + ahead] - bars[:, at] >= lookahead
            counts[anchors] = ok.sum(axis=0)
        return counts

    def eligible_symbols(
        self,
        time_id: int,
        lookback: int,
        lookahead: int = 0,
        symbol_mask: np.ndarray | None = None,
        span_ratio: float = 1.0,
    ) -> np.ndarray:
        back, ahead = self._spans(lookback, lookahead, span_ratio)
        t0, t1 = time_id - back + 1, time_id + ahead + 1
        if t0 < 0 or t1 > self.num_timestamps:
            return np.zeros(0, dtype=np.int64)
        valid = self.valid(t0, t1)
        keep = valid[:, back - 1] & (valid[:, :back].sum(axis=1) >= lookback)
        keep &= valid[:, back:].sum(axis=1) >= lookahead
        if symbol_mask is not None:
            keep &= symbol_mask
        return np.flatnonzero(keep)

    def panel(
        self,
        time_id: int,
        lookback: int,
        lookahead: int = 0,
        features: Sequence[str] | None = None,
        symbol_mask: np.ndarray | None = None,
        span_ratio: float = 1.0,
        dtype=np.float32,
    ) -> Panel:
        """Every eligible symbol's own last ``lookback`` bars up to ``time_id`` and next ``lookahead`` bars.

        Args:
            symbol_mask: optional bool ``[num_symbols]`` universe filter (e.g. held-out stocks).
            span_ratio: how far (as a multiple of ``lookback`` / ``lookahead``) a symbol's bars may spread
                over the timeline; 1 asks for no missing bar in the window.
        """
        symbol_ids = self.eligible_symbols(time_id, lookback, lookahead, symbol_mask, span_ratio)
        return self.read_panel(time_id, symbol_ids, lookback, lookahead, features, span_ratio, dtype)

    def read_panel(
        self,
        time_id: int,
        symbol_ids: np.ndarray,
        lookback: int,
        lookahead: int = 0,
        features: Sequence[str] | None = None,
        span_ratio: float = 1.0,
        dtype=np.float32,
    ) -> Panel:
        """``panel`` for given symbols, which must be eligible (e.g. a subset of ``eligible_symbols``)."""
        features = self.feature_names if features is None else list(features)
        back, ahead = self._spans(lookback, lookahead, span_ratio)
        t0, t1 = time_id - back + 1, time_id + ahead + 1
        values = np.empty((len(symbol_ids), lookback + lookahead, len(features)), dtype=dtype)
        if len(symbol_ids):
            select = None  # [N, back + ahead] cells kept, exactly lookback + lookahead per row, in time order
            if (back, ahead) != (lookback, lookahead):
                valid = self.valid(t0, t1, symbol_ids)
                past, future = valid[:, :back], valid[:, back:]
                past_rank = np.cumsum(past[:, ::-1], axis=1)[:, ::-1]  # bars from here to t, inclusive
                select = np.concatenate(
                    [past & (past_rank <= lookback), future & (np.cumsum(future, axis=1) <= lookahead)], axis=1
                )
            for j, name in enumerate(features):
                grid = self.grid(name, t0, t1, symbol_ids)
                values[:, :, j] = grid if select is None else grid[select].reshape(len(symbol_ids), -1)
        return Panel(time_id=time_id, timestamp=int(self.timeline[time_id]), symbol_ids=symbol_ids, values=values)
