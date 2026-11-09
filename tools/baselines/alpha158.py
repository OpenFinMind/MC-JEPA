"""Qlib's Alpha158 factors on daily bars aggregated from the v3 1-min data.

Follows ``qlib.contrib.data.loader.Alpha158DL`` with its default config: 9 K-bar features, OPEN0 / HIGH0 / LOW0 /
VWAP0, and 29 rolling operators over 5 / 10 / 20 / 30 / 60 trading days (158 in all). Rolling windows take the
available history (qlib's ``min_periods=1``). Differences from qlib:

- prices are not adjusted for corporate actions (the 1-min data has no adjustment factor), so a split or bonus
  issue shows up as a jump in the windows that span it;
- VWAP is the volume-weighted mean of the day's 1-min closes (there is no amount column);
- the day's bar is the 09:30-15:00 session, so day t's features use exactly the data up to the 15:00 decision time;
- ``history`` (optional) limits day t's features to its last ``history`` trading days, e.g. 4 for the 960-bar
  encoder window: rolling windows shrink to it, and a change from the day before is dropped where that day is out.

``normalize`` is qlib's default ``RobustZScoreNorm(clip_outlier=True)`` + ``Fillna``: per-feature median and MAD
over the fit rows, clipped to +-3, missing values set to 0.
"""

import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from mc_jepa.datasets.v3.indexed_dataset import MMapIndexedDatasetV3

WINDOWS = (5, 10, 20, 30, 60)
ROLLING = (
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD",
    "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
)  # fmt: skip
NAMES = [
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
    "OPEN0",
    "HIGH0",
    "LOW0",
    "VWAP0",
] + [f"{op}{d}" for op in ROLLING for d in WINDOWS]
EPS = 1e-12

_data: MMapIndexedDatasetV3 | None = None


def daily_bars(data: MMapIndexedDatasetV3, s: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Dates and daily open / high / low / close / volume / vwap of symbol ``s``; non-positive prices are NaN."""
    a, b = data.symbol_range(s)
    bars = data.read(a, b, ["open", "high", "low", "close", "volume"], dtype=np.float64)
    price = np.where(bars[:, :4] > 0, bars[:, :4], np.nan)
    volume = bars[:, 4]
    date = data.timeline[np.asarray(data.time_ids[a:b])] // 1_000_000
    starts = np.flatnonzero(np.r_[True, date[1:] != date[:-1]])
    ends = np.r_[starts[1:], len(date)]
    traded = np.where(np.isfinite(price[:, 3]), volume, 0.0)
    day_volume = np.add.reduceat(volume, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.add.reduceat(np.nan_to_num(price[:, 3]) * traded, starts) / np.add.reduceat(traded, starts)
    bars = {
        "open": price[starts, 0],
        "high": np.fmax.reduceat(price[:, 1], starts),
        "low": np.fmin.reduceat(price[:, 2], starts),
        "close": price[ends - 1, 3],
        "volume": day_volume,
    }
    bars["vwap"] = np.where(np.isfinite(vwap), vwap, bars["close"])
    return date[starts].astype(np.int32), bars


def _win(x: np.ndarray, d: int, drop_first: bool = False) -> np.ndarray:
    """[T, d] trailing windows of x, NaN before the series starts (and at the oldest position with ``drop_first``)."""
    w = sliding_window_view(np.r_[np.full(d - 1, np.nan), x], d)
    return np.c_[np.full(len(w), np.nan), w[:, 1:]] if drop_first else w


def _ref(x: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(x))
    return np.r_[np.full(k, np.nan), x[: len(x) - k]]


def _std(w: np.ndarray) -> np.ndarray:
    return np.nanstd(w, axis=1, ddof=1)


def _sum(w: np.ndarray) -> np.ndarray:
    """Window sum, NaN when the window has no value (pandas' ``rolling(min_periods=1).sum``)."""
    return np.where(np.isfinite(w).any(axis=1), np.nansum(w, axis=1), np.nan)


def _corr(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation over the positions where both windows are finite."""
    ok = np.isfinite(wx) & np.isfinite(wy)
    n = ok.sum(axis=1)
    x, y = np.where(ok, wx, 0.0), np.where(ok, wy, 0.0)
    mx, my = x.sum(axis=1) / n, y.sum(axis=1) / n
    dx, dy = np.where(ok, x - mx[:, None], 0.0), np.where(ok, y - my[:, None], 0.0)
    return np.where(n >= 2, (dx * dy).sum(axis=1) / np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1)), np.nan)


def _regress(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope, R^2 and last-point residual of the window values on their position."""
    ok = np.isfinite(w)
    n = ok.sum(axis=1)
    p = np.broadcast_to(np.arange(w.shape[1], dtype=np.float64), w.shape)
    y = np.where(ok, w, 0.0)
    mp, my = np.where(ok, p, 0.0).sum(axis=1) / n, y.sum(axis=1) / n
    dp, dy = np.where(ok, p - mp[:, None], 0.0), np.where(ok, y - my[:, None], 0.0)
    sxx, sxy, syy = (dp * dp).sum(axis=1), (dp * dy).sum(axis=1), (dy * dy).sum(axis=1)
    slope = np.where(n >= 2, sxy / sxx, np.nan)
    rsqr = np.where(n >= 2, sxy * sxy / (sxx * syy), np.nan)
    resi = w[:, -1] - (my + slope * (w.shape[1] - 1 - mp))
    return slope, rsqr, resi


def _idx(w: np.ndarray, fill: float, arg) -> np.ndarray:
    """1-based position of the window max / min counted from the first available day, as qlib's IdxMax / IdxMin."""
    ok = np.isfinite(w)
    pad = (~ok).cumprod(axis=1).sum(axis=1)  # leading NaNs (history shorter than the window)
    return np.where(ok.any(axis=1), arg(np.where(ok, w, fill), axis=1) - pad + 1.0, np.nan)


def features(bars: dict[str, np.ndarray], history: int | None = None) -> np.ndarray:
    """[T, 158] Alpha158 of one stock's daily bars, columns in ``NAMES`` order; each row sees ``history`` days."""
    o, h, low, c, v, vwap = (bars[k] for k in ("open", "high", "low", "close", "volume", "vwap"))
    hl = h - low + EPS
    out = [
        (c - o) / o,
        (h - low) / o,
        (c - o) / hl,
        (h - np.fmax(o, c)) / o,
        (h - np.fmax(o, c)) / hl,
        (np.fmin(o, c) - low) / o,
        (np.fmin(o, c) - low) / hl,
        (2 * c - h - low) / o,
        (2 * c - h - low) / hl,
        o / c,
        h / c,
        low / c,
        vwap / c,
    ]
    c1, v1 = _ref(c, 1), _ref(v, 1)
    up, down, move = np.fmax(c - c1, 0), np.fmax(c1 - c, 0), np.abs(c - c1)
    vup, vdown, vmove = np.fmax(v - v1, 0), np.fmax(v1 - v, 0), np.abs(v - v1)
    log_v1 = np.log(v + 1)
    log_dv = np.log(v / v1 + 1)
    wv = np.abs(c / c1 - 1) * v
    has_prev = np.isfinite(c1) & np.isfinite(c)
    gt, lt = np.where(has_prev, c > c1, np.nan), np.where(has_prev, c < c1, np.nan)
    by_op = {op: [] for op in ROLLING}
    for d in WINDOWS:
        e = d if history is None else min(d, history)
        # windows of a day-over-day change; at full history the oldest day's change needs the day before it
        dwin = partial(_win, d=e, drop_first=e == history)

        wc, wh, wl = _win(c, e), _win(h, e), _win(low, e)
        hi, lo = np.nanmax(wh, axis=1), np.nanmin(wl, axis=1)
        slope, rsqr, resi = _regress(wc)
        n = np.isfinite(wc).sum(axis=1)
        rank = ((wc < c[:, None]).sum(axis=1) + (wc <= c[:, None]).sum(axis=1) + 1) / (2 * n)
        imax, imin = _idx(wh, -np.inf, np.argmax), _idx(wl, np.inf, np.argmin)
        cntp, cntn = np.nanmean(dwin(gt), axis=1), np.nanmean(dwin(lt), axis=1)
        sump, sumn = _sum(dwin(up)), _sum(dwin(down))
        summ = _sum(dwin(move)) + EPS
        vsump, vsumn = _sum(dwin(vup)), _sum(dwin(vdown))
        vsumm = _sum(dwin(vmove)) + EPS
        wwv = dwin(wv)
        values = {
            "ROC": _ref(c, d) / c if history is None or d < history else np.full_like(c, np.nan),
            "MA": np.nanmean(wc, axis=1) / c,
            "STD": _std(wc) / c,
            "BETA": slope / c,
            "RSQR": rsqr,
            "RESI": resi / c,
            "MAX": hi / c,
            "MIN": lo / c,
            "QTLU": np.nanquantile(wc, 0.8, axis=1) / c,
            "QTLD": np.nanquantile(wc, 0.2, axis=1) / c,
            "RANK": np.where(np.isfinite(c), rank, np.nan),
            "RSV": (c - lo) / (hi - lo + EPS),
            "IMAX": imax / d,
            "IMIN": imin / d,
            "IMXD": (imax - imin) / d,
            "CORR": _corr(wc, _win(log_v1, e)),
            "CORD": _corr(dwin(c / c1), dwin(log_dv)),
            "CNTP": cntp,
            "CNTN": cntn,
            "CNTD": cntp - cntn,
            "SUMP": sump / summ,
            "SUMN": sumn / summ,
            "SUMD": (sump - sumn) / summ,
            "VMA": np.nanmean(_win(v, e), axis=1) / (v + EPS),
            "VSTD": _std(_win(v, e)) / (v + EPS),
            "WVMA": _std(wwv) / (np.nanmean(wwv, axis=1) + EPS),
            "VSUMP": vsump / vsumm,
            "VSUMN": vsumn / vsumm,
            "VSUMD": (vsump - vsumn) / vsumm,
        }
        for op in ROLLING:
            by_op[op].append(values[op])
    out += [x for op in ROLLING for x in by_op[op]]
    x = np.stack(out, axis=1)
    return np.where(np.isfinite(x), x, np.nan).astype(np.float32)


def _symbol(args: tuple[int, int, int, int | None]) -> tuple[int, np.ndarray, np.ndarray]:
    s, start, end, history = args
    date, bars = daily_bars(_data, s)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        x = features(bars, history)
    keep = (date >= start) & (date < end)
    return s, date[keep], x[keep]


def _init(prefix: str):
    global _data
    _data = MMapIndexedDatasetV3(prefix)


def compute(
    prefix: str, start: int, end: int, num_workers: int = 16, history: int | None = None
) -> dict[str, np.ndarray]:
    """Raw Alpha158 of every (symbol, trading day) in ``[start, end)``; the windows also use the days before."""
    data = MMapIndexedDatasetV3(prefix)
    jobs = [(s, start, end, history) for s in range(data.num_symbols)]
    symbols, dates, xs = [], [], []
    with Pool(num_workers, initializer=_init, initargs=(prefix,)) as pool:
        for s, date, x in tqdm(pool.imap(_symbol, jobs, chunksize=8), total=len(jobs), desc="alpha158"):
            symbols.append(np.full(len(date), data.symbols[s]))
            dates.append(date)
            xs.append(x)
    return {"symbol": np.concatenate(symbols), "date": np.concatenate(dates), "alpha158": np.concatenate(xs)}


def normalize(x: np.ndarray, fit: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """qlib's RobustZScoreNorm(clip_outlier=True) + Fillna, fit on the rows where ``fit``; in place, returns x."""
    for j in range(x.shape[1]):
        col = x[:, j]
        median = np.nanmedian(col[fit])
        mad = np.nanmedian(np.abs(col[fit] - median)) * 1.4826
        col -= median
        col /= mad + EPS
        np.clip(col, -clip, clip, out=col)
        col[~np.isfinite(col)] = 0.0
    return x
