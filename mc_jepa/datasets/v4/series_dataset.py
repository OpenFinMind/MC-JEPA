"""Per-stock windows of the v3 ``StockSeriesDataset``, read from a v4 time-major dataset one decision time at a time.

Each item is one decision time with every stock that has a bar there and at least ``lookback`` bars before it (its
own last ``lookback`` bars, however far back they reach). This is the v3 window rule, where a window is the stock's
last ``lookback`` rows with no check for gaps, so a v4 copy of the same bars gives the same (stock, time) rows.
Stocks whose bars fit in ``span_ratio`` x ``lookback`` timestamps are read as one panel (whole blocks); the few left
(halts, suspensions) are read over up to ``max_span`` timestamps, and dropped if they still miss bars.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mc_jepa.datasets.daily import DailyBars
from mc_jepa.datasets.v4.indexed_dataset import MMapIndexedDatasetV4


class CrossSectionSeriesDataset(Dataset):
    """Items are decision times; ``collate_fn`` stacks their windows like ``series_collate_fn`` (one-stock panels).

    An item holds ``symbol_id`` [N], ``timestamp`` [N], ``context`` [N, lookback, F] and, with ``normalize``,
    ``mean`` / ``std`` [N, F] (context statistics, as in v3), plus ``daily`` with ``daily_lookback``.
    """

    def __init__(
        self,
        data_prefix: str,
        features: Sequence[str],
        lookback: int,
        times_of_day: Sequence[int] | None = None,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
        span_ratio: float = 2.0,
        max_span: int = 30_000,
        normalize: bool = True,
        daily_dir: str | None = None,
        daily_lookback: int = 0,
        daily_feature_set: str = "features-1",
    ):
        self.data = MMapIndexedDatasetV4(data_prefix)
        self.features = list(features)
        missing = [f for f in self.features if f not in self.data.feature_dtypes]
        if missing:
            raise KeyError(f"features {missing} not in {data_prefix}")
        self.lookback = lookback
        self.span_ratio = span_ratio
        self.max_span = max_span
        self.normalize = normalize
        self.daily = None
        if daily_lookback > 0:
            assert daily_dir is not None, "daily_lookback needs daily_dir"
            self.daily = DailyBars(daily_dir, self.data.symbols, daily_lookback, daily_feature_set)
        timeline = self.data.timeline
        time_mask = np.ones(len(timeline), dtype=bool)
        if times_of_day is not None:
            time_mask &= np.isin(timeline % 1_000_000, list(times_of_day))
        if start_timestamp is not None:
            time_mask &= timeline >= start_timestamp
        if end_timestamp is not None:
            time_mask &= timeline < end_timestamp
        self.time_ids = np.flatnonzero(time_mask)
        self.time_ids = self.time_ids[self.time_ids >= lookback - 1]

    def __len__(self) -> int:
        return len(self.time_ids)

    def _read_wide(self, time_id: int, symbol_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Last ``lookback`` bars of a few symbols over up to ``max_span`` timestamps; those with fewer are dropped."""
        t0 = max(0, time_id + 1 - self.max_span)
        valid = self.data.valid(t0, time_id + 1, symbol_ids)
        rank = np.cumsum(valid[:, ::-1], axis=1)[:, ::-1]  # bars from here to t, inclusive
        keep = rank[:, 0] >= self.lookback
        select = valid[keep] & (rank[keep] <= self.lookback)
        values = np.empty((int(keep.sum()), self.lookback, len(self.features)), dtype=np.float32)
        for j, name in enumerate(self.features):
            grid = self.data.grid(name, t0, time_id + 1, symbol_ids[keep])
            values[:, :, j] = grid[select].reshape(len(values), -1)
        return symbol_ids[keep], values

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        time_id = int(self.time_ids[idx])
        data = self.data
        symbol_ids = data.eligible_symbols(time_id, self.lookback, 0, span_ratio=self.span_ratio)
        values = data.read_panel(time_id, symbol_ids, self.lookback, 0, self.features, self.span_ratio).values
        at_t = np.flatnonzero(data.valid(time_id, time_id + 1)[:, 0])
        rest = np.setdiff1d(at_t, symbol_ids)  # a bar at t but fewer than lookback in the span
        if len(rest):
            rest, rest_values = self._read_wide(time_id, rest)
            symbol_ids = np.concatenate([symbol_ids, rest])
            values = np.concatenate([values, rest_values])
            order = np.argsort(symbol_ids)
            symbol_ids, values = symbol_ids[order], values[order]
        timestamp = int(data.timeline[time_id])
        item = {
            "symbol_id": torch.from_numpy(symbol_ids),
            "timestamp": torch.full((len(symbol_ids),), timestamp, dtype=torch.long),
        }
        if self.daily is not None:
            item["daily"] = torch.from_numpy(self.daily(symbol_ids, timestamp // 1_000_000))
        if self.normalize:
            mean = values.mean(axis=1, keepdims=True)
            std = values.std(axis=1, keepdims=True) + 1e-5
            values = (values - mean) / std
            item.update(mean=torch.from_numpy(mean[:, 0]), std=torch.from_numpy(std[:, 0]))
        item["context"] = torch.from_numpy(np.ascontiguousarray(values))
        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Concatenate decision times into one-stock panels: context [N, 1, lookback, F], symbol_mask [N, 1]."""
        cat = {k: torch.cat([item[k] for item in batch]) for k in batch[0]}
        out = {"timestamp": cat.pop("timestamp"), "symbol_ids": cat.pop("symbol_id")[:, None]}
        out.update({k: v[:, None] for k, v in cat.items()})
        out["symbol_mask"] = torch.ones(len(out["timestamp"]), 1, dtype=torch.bool)
        return out
