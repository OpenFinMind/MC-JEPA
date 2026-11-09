from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mc_jepa.datasets.daily import DailyBars
from mc_jepa.datasets.interfaces import PanelSample
from mc_jepa.datasets.v4.indexed_dataset import MMapIndexedDatasetV4


class CrossSectionPanelDataset(Dataset):
    """One sample per decision timestamp: every eligible stock's history (and optional future) at that time.

    Sample ``i`` is decision time ``t = self.time_ids[i]``. For each eligible stock (a bar at t, and with
    the default ``span_ratio=1`` a bar at every timestamp of ``[t - lookback + 1, t + lookahead]`` of the
    global timeline; see ``span_ratio``), it returns

        context  [N, lookback, F]   the stock's own bars up to and including t
        future   [N, lookahead, F]  its next bars after t (only if lookahead > 0)
        symbol_ids [N]

    With ``normalize``, each stock and feature is z-scored by its *context* statistics only, so the
    future block never leaks into the normalization; ``mean``/``std`` are returned for inversion.

    Each item costs one read per feature instead of one per stock (see
    mc_jepa/datasets/v4/indexed_dataset.py).
    """

    is_collate_needed = True

    def __init__(
        self,
        data_prefix: str,
        features: Sequence[str],
        lookback: int,
        lookahead: int = 0,
        times_of_day: Sequence[int] | None = None,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
        symbols: Sequence[str] | None = None,
        min_symbols: int = 1,
        span_ratio: float = 1.0,
        max_symbols: int | None = None,
        deterministic: bool = False,
        seed: int = 0,
        normalize: bool = True,
        warmup: bool = False,
        daily_dir: str | None = None,
        daily_lookback: int = 0,
        daily_feature_set: str = "features-1",
    ):
        """
        Args:
            times_of_day: keep decision times whose HHMMSS is listed, e.g. ``[150000]`` for day ends.
            start_timestamp, end_timestamp: keep decision times in ``[start, end)`` (YYYYMMDDHHMMSS).
            symbols: universe allowlist (e.g. to hold stocks out); ``None`` keeps all.
            min_symbols: drop decision times with fewer eligible stocks.
            span_ratio: a stock's own ``lookback`` (``lookahead``) bars may spread over ``ceil(lookback *
                span_ratio)`` (``ceil(lookahead * span_ratio)``) timestamps of the global timeline, so halts and
                minutes without trades do not drop it; 1 asks for a bar at every timestamp of the window. The
                stock must still have a bar at the decision time.
            max_symbols: keep a random subset of at most this many eligible stocks per panel, chosen before
                the read so only their bars are loaded. The subset comes from the worker's torch RNG, or with
                ``deterministic`` is seeded by ``seed`` and the decision time, so every pass sees the same stocks.
            daily_dir, daily_lookback, daily_feature_set: the ``tools.daily_bars`` table, how many days of it each
                stock gets, as ``daily`` [N, daily_lookback, DAILY_FEATURES], and which features
                (``mc_jepa/datasets/daily.py``).
        """
        self.data = MMapIndexedDatasetV4(data_prefix, warmup=warmup)
        self.features = list(features)
        missing = [f for f in self.features if f not in self.data.feature_dtypes]
        if missing:
            raise KeyError(f"features {missing} not in {data_prefix}")
        self.lookback = lookback
        self.lookahead = lookahead
        self.normalize = normalize
        self.span_ratio = span_ratio
        self.max_symbols = max_symbols
        self.deterministic = deterministic
        self.seed = seed
        self.daily = None
        if daily_lookback > 0:
            assert daily_dir is not None, "daily_lookback needs daily_dir"
            self.daily = DailyBars(daily_dir, self.data.symbols, daily_lookback, daily_feature_set)

        if symbols is None:
            self.symbol_mask = np.ones(self.data.num_symbols, dtype=bool)
        else:
            self.symbol_mask = np.isin(np.asarray(self.data.symbols), list(symbols))

        timeline = self.data.timeline
        time_mask = np.ones(len(timeline), dtype=bool)
        if times_of_day is not None:
            time_mask &= np.isin(timeline % 1_000_000, list(times_of_day))
        if start_timestamp is not None:
            time_mask &= timeline >= start_timestamp
        if end_timestamp is not None:
            time_mask &= timeline < end_timestamp

        counts = self.data.count_eligible(lookback, lookahead, time_mask, self.symbol_mask, span_ratio)
        self.time_ids = np.flatnonzero(counts >= max(min_symbols, 1))

    def __len__(self) -> int:
        return len(self.time_ids)

    def _subsample(self, time_id: int, symbol_ids: np.ndarray) -> np.ndarray:
        n = len(symbol_ids)
        if self.max_symbols is None or n <= self.max_symbols:
            return symbol_ids
        generator = None
        if self.deterministic:
            generator = torch.Generator().manual_seed(self.seed + int(self.data.timeline[time_id]))
        keep = torch.randperm(n, generator=generator)[: self.max_symbols].sort().values.numpy()
        return symbol_ids[keep]

    def __getitem__(self, idx: int) -> PanelSample:
        time_id = int(self.time_ids[idx])
        args = (self.lookback, self.lookahead)
        symbol_ids = self.data.eligible_symbols(time_id, *args, self.symbol_mask, self.span_ratio)
        symbol_ids = self._subsample(time_id, symbol_ids)
        panel = self.data.read_panel(time_id, symbol_ids, *args, self.features, self.span_ratio)
        values = panel.values  # [N, lookback + lookahead, F]
        sample: PanelSample = {
            "symbol_ids": torch.from_numpy(panel.symbol_ids),
            "timestamp": torch.tensor(panel.timestamp, dtype=torch.long),
        }
        if self.daily is not None:
            sample["daily"] = torch.from_numpy(self.daily(panel.symbol_ids, int(panel.timestamp) // 1_000_000))
        if self.normalize:
            context = values[:, : self.lookback]
            mean = context.mean(axis=1, keepdims=True)
            std = context.std(axis=1, keepdims=True) + 1e-5
            values = (values - mean) / std
            sample.update(mean=torch.from_numpy(mean[:, 0]), std=torch.from_numpy(std[:, 0]))
        sample["context"] = torch.from_numpy(np.ascontiguousarray(values[:, : self.lookback]))
        if self.lookahead > 0:
            sample["future"] = torch.from_numpy(np.ascontiguousarray(values[:, self.lookback :]))
        return sample

    @staticmethod
    def collate_fn(batch: list[PanelSample]) -> dict[str, torch.Tensor]:
        """Pad the stock axis to the largest panel; ``symbol_mask[b, i]`` marks real stocks."""
        max_n = max(len(item["symbol_ids"]) for item in batch)
        collated: dict[str, torch.Tensor] = {
            "timestamp": torch.stack([item["timestamp"] for item in batch]),
            "symbol_mask": torch.zeros(len(batch), max_n, dtype=torch.bool),
        }
        for b, item in enumerate(batch):
            collated["symbol_mask"][b, : len(item["symbol_ids"])] = True
        for key in batch[0]:
            if key == "timestamp":
                continue
            first = batch[0][key]
            fill = -1 if key == "symbol_ids" else 0
            out = first.new_full((len(batch), max_n, *first.shape[1:]), fill)
            for b, item in enumerate(batch):
                out[b, : len(item[key])] = item[key]
            collated[key] = out
        return collated
