"""Daily bars as a second, longer context for the JEPA encoder (``model.daily_lookback``).

The 1-min context covers ~4 trading days. ``DailyBars`` adds each stock's last ``lookback`` completed trading days
before the decision date (never the decision day itself, so nothing after the decision time leaks), as scale-free
features per day. ``feature_set`` (``model.daily_feature_set``) picks them:

    features-1   valid, gap log(O_d / C_{d-1}), body log(C_d / O_d), range log(H_d / L_d), return log(C_d / C_{d-1}),
                 volume (gap + body = return, so 3 independent price directions)
    features-2   valid, log(O_d / C_{d-1}), log(H_d / C_{d-1}), log(L_d / C_{d-1}), log(C_d / C_{d-1}), volume: the
                 day's full OHLC relative to the previous close (features-1 is a linear function of it)

where valid is 1 if the stock traded that day and the day before, else 0 (and the other features are 0), and volume
is log(volume_d) minus its mean over the window's valid days.

Price features are times ``PRICE_SCALE`` (daily moves of ~2-5% become ~0.5-1) and dropped when the close-to-close
move exceeds ``MAX_DAILY_MOVE`` (corporate actions in the unadjusted prices). The table is built once from a v3
1-min mmap with ``python -m tools.daily_bars`` and loaded whole into memory (shared by forked loader workers).
"""

import os

import numpy as np
from tqdm import tqdm

DAILY_FEATURES = 6
FEATURE_SETS = ("features-1", "features-2")
PRICE_SCALE = 20.0
MAX_DAILY_MOVE = 0.205  # as tools/eval_jepa_probe.py: A-share limits are 10% / 20%


def build_daily_bars(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trading dates [D] (YYYYMMDD), symbols [S] and daily open / high / low / close / volume [D, S, 5] (NaN where
    a stock has no bar that day) of a ``MMapIndexedDatasetV3``."""
    calendar = np.unique(data.timeline // 1_000_000)
    ohlcv = np.full((len(calendar), data.num_symbols, 5), np.nan, dtype=np.float32)
    for s in tqdm(range(data.num_symbols), desc="daily bars"):
        a, b = data.symbol_range(s)
        if b <= a:
            continue
        bars = data.read(a, b, ["open", "high", "low", "close", "volume"], dtype=np.float64)
        bars[:, :4] = np.where(bars[:, :4] > 0, bars[:, :4], np.nan)
        date = data.timeline[np.asarray(data.time_ids[a:b])] // 1_000_000
        starts = np.flatnonzero(np.r_[True, date[1:] != date[:-1]])
        ends = np.r_[starts[1:], len(date)]
        day = np.searchsorted(calendar, date[starts])
        ohlcv[day, s, 0] = bars[starts, 0]
        ohlcv[day, s, 1] = np.fmax.reduceat(bars[:, 1], starts)
        ohlcv[day, s, 2] = np.fmin.reduceat(bars[:, 2], starts)
        ohlcv[day, s, 3] = bars[ends - 1, 3]
        ohlcv[day, s, 4] = np.add.reduceat(bars[:, 4], starts)
    return calendar.astype(np.int32), np.asarray(data.symbols), ohlcv


def save_daily_bars(out_dir: str, dates: np.ndarray, symbols: np.ndarray, ohlcv: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dates.npy"), dates)
    np.save(os.path.join(out_dir, "symbols.npy"), symbols)
    np.save(os.path.join(out_dir, "ohlcv.npy"), ohlcv)


class DailyBars:
    """Daily context features of one dataset's stocks, looked up by symbol id and decision date."""

    def __init__(self, daily_dir: str, symbols, lookback: int, feature_set: str = "features-1"):
        """``symbols``: the dataset's symbol names, in symbol-id order (stocks missing from the table get no
        valid day). ``feature_set``: one of ``FEATURE_SETS``."""
        assert feature_set in FEATURE_SETS, f"daily feature_set must be one of {FEATURE_SETS}, got {feature_set}"
        self.feature_set = feature_set
        daily_dir = os.path.expanduser(daily_dir)
        self.dates = np.load(os.path.join(daily_dir, "dates.npy"))
        table_symbols = np.load(os.path.join(daily_dir, "symbols.npy"))
        ohlcv = np.load(os.path.join(daily_dir, "ohlcv.npy"))
        # one extra all-NaN column for stocks missing from the table, and one leading all-NaN day for windows
        # that start before the first date
        self.ohlcv = np.full((ohlcv.shape[0] + 1, ohlcv.shape[1] + 1, 5), np.nan, dtype=np.float32)
        self.ohlcv[1:, :-1] = ohlcv
        lookup = {name: i for i, name in enumerate(table_symbols)}
        self.columns = np.array([lookup.get(name, ohlcv.shape[1]) for name in symbols], dtype=np.int64)
        self.lookback = lookback

    def __call__(self, symbol_ids: np.ndarray, date: int) -> np.ndarray:
        """float32 [n, lookback, DAILY_FEATURES] for the stocks ``symbol_ids`` [n] at decision date ``date``."""
        end = int(np.searchsorted(self.dates, date, side="left")) + 1  # +1: the leading NaN day; days < date
        rows = np.arange(end - self.lookback - 1, end)  # one more day for the first day's previous close
        rows = np.where(rows >= 0, rows, 0)
        columns = self.columns[np.asarray(symbol_ids)]
        bars = self.ohlcv[rows[None, :], columns[:, None]].astype(np.float64)  # [n, L + 1, 5]
        return daily_features(bars, self.feature_set)


def daily_features(bars: np.ndarray, feature_set: str = "features-1") -> np.ndarray:
    """[n, L + 1, 5] daily open / high / low / close / volume (the first day only gives a previous close) ->
    float32 [n, L, DAILY_FEATURES] of ``feature_set``."""
    prev_close, day = bars[:, :-1, 3], bars[:, 1:]
    open_, high, low, close, volume = (day[..., k] for k in range(5))
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(close / prev_close)
        valid = np.isfinite(ret) & (np.abs(np.expm1(ret)) <= MAX_DAILY_MOVE) & (volume > 0)
        valid &= np.isfinite(open_) & np.isfinite(high) & np.isfinite(low)
        log_volume = np.where(valid, np.log(np.where(volume > 0, volume, 1.0)), 0.0)
        mean_volume = log_volume.sum(axis=1, keepdims=True) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        if feature_set == "features-1":
            prices = [np.log(open_ / prev_close), np.log(close / open_), np.log(high / low), ret]
        elif feature_set == "features-2":
            prices = [np.log(open_ / prev_close), np.log(high / prev_close), np.log(low / prev_close), ret]
        else:
            raise ValueError(f"daily feature_set must be one of {FEATURE_SETS}, got {feature_set}")
        out = np.stack(
            [valid.astype(np.float64), *(PRICE_SCALE * p for p in prices), log_volume - mean_volume], axis=-1
        )
    out[~valid] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
