from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mc_jepa.datasets.daily import DailyBars
from mc_jepa.datasets.interfaces import SeriesSample
from mc_jepa.datasets.v3.indexed_dataset import MMapIndexedDatasetV3


class StockSeriesDataset(Dataset):
    """One sample per (stock, decision timestamp): that stock's own history (and optional future).

    Sample ``i`` is anchored at row ``r = self.anchor_rows[i]`` of a single stock. Every row whose window
    ``[r - lookback + 1, r + lookahead]`` lies in its own series is a candidate (gaps such as halts are
    not checked), and it returns

        context  [lookback, F]   rows up to and including r
        future   [lookahead, F]  rows after r (only if lookahead > 0)

    A window is one contiguous row range, so each item costs one read per dtype group. With
    ``normalize``, each feature is z-scored by its *context* statistics only, so the future block never
    leaks into the normalization; ``mean``/``std`` are returned for inversion.

    With ``daily_dir`` and ``daily_lookback``, ``daily`` [daily_lookback, DAILY_FEATURES] holds the stock's last
    completed trading days before the decision date (``mc_jepa/datasets/daily.py``).
    """

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
        num_samples: int | None = None,
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
            num_samples: keep a random subset (drawn with ``seed``) of this many anchors; ``None`` keeps all.
            daily_dir, daily_lookback, daily_feature_set: the ``tools.daily_bars`` table, how many days of it each
                sample gets and which features (``mc_jepa/datasets/daily.py``).
        """
        self.data = MMapIndexedDatasetV3(data_prefix, warmup=warmup)
        self.features = list(features)
        missing = [f for f in self.features if f not in self.data.feature_dtypes]
        if missing:
            raise KeyError(f"features {missing} not in {data_prefix}")
        self.lookback = lookback
        self.lookahead = lookahead
        self.normalize = normalize
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

        anchor_rows = self._eligible_rows(time_mask)
        if num_samples is not None and num_samples < len(anchor_rows):
            pick = np.random.default_rng(seed).choice(len(anchor_rows), num_samples, replace=False)
            anchor_rows = anchor_rows[np.sort(pick)]
        self.anchor_rows = anchor_rows

    def _eligible_rows(self, time_mask: np.ndarray, chunk: int = 1 << 24) -> np.ndarray:
        """Anchor rows with a valid window, scanning all rows in chunks."""
        data = self.data
        out = []
        for start in range(0, data.num_rows, chunk):
            rows = np.arange(start, min(start + chunk, data.num_rows), dtype=np.int64)
            tids = np.asarray(data.time_ids[rows[0] : rows[-1] + 1])
            keep = time_mask[tids] & self.symbol_mask[data.symbol_of_rows(rows)]
            keep[keep] = data.window_mask(rows[keep], self.lookback, self.lookahead)
            out.append(rows[keep])
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.anchor_rows)

    def __getitem__(self, idx: int) -> SeriesSample:
        anchor = int(self.anchor_rows[idx])
        values = self.data.read(anchor - self.lookback + 1, anchor + self.lookahead + 1, self.features)
        symbol_id = int(self.data.symbol_of_rows(np.array([anchor]))[0])
        timestamp = int(self.data.timeline[self.data.time_ids[anchor]])
        sample: SeriesSample = {
            "symbol_id": torch.tensor(symbol_id, dtype=torch.long),
            "timestamp": torch.tensor(timestamp, dtype=torch.long),
        }
        if self.daily is not None:
            sample["daily"] = torch.from_numpy(self.daily(np.array([symbol_id]), timestamp // 1_000_000)[0])
        if self.normalize:
            context = values[: self.lookback]
            mean = context.mean(axis=0)
            std = context.std(axis=0) + 1e-5
            values = (values - mean) / std
            sample.update(mean=torch.from_numpy(mean), std=torch.from_numpy(std))
        sample["context"] = torch.from_numpy(np.ascontiguousarray(values[: self.lookback]))
        if self.lookahead > 0:
            sample["future"] = torch.from_numpy(np.ascontiguousarray(values[self.lookback :]))
        return sample
