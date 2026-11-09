"""Frozen-encoder ridge probes for series and panel JEPA checkpoints (``qtsmfjepa``) on v3 1-min data.

One sample is one (stock, trading day): the encoder embeds the 240 bars up to the 15:00 close of day ``t``. Labels
are computed over the stock's own consecutive trading days, and are dropped when the stock skips a market trading day:

    return         ``return_1d_delay_1d`` = close(t+2) / close(t+1) - 1, the next-day return a daily strategy trades
    return_5d          ``return_5d_delay_1d`` = close(t+6) / close(t+1) - 1
    return_20d         ``return_20d_delay_1d`` = close(t+21) / close(t+1) - 1
    beta_20d           OLS slope of the stock's daily returns on the market's over days t+1 .. t+20; the market is
                       the equal-weight mean of all stocks' daily returns, both clipped to +-20%
    jump_1d            1 if |close(t+1) / close(t) - 1| >= 9.5% (a limit-scale move), else 0; moves beyond 20.5%
                       (corporate actions in the unadjusted prices) are dropped
    volatility         std of the 1-min close-to-close returns of day t+1 (the first includes the overnight gap)
    volatility_5d      the same std over days t+1 .. t+5
    max_drawdown_5d    largest drop of the 1-min close from its running peak (starting at close(t)) over days
                       t+1 .. t+5
    volume             day t+1 volume / day t volume
    volume_5d          mean daily volume over days t+1 .. t+5 / day t volume
    open_volume_share  share of day t+1 volume traded in its first 30 bars
    volume_profile_{k} the intraday volume profile, for VWAP execution: log of day t+1's volume share in 30-min bin k
                       (1 .. 8, 9:31 .. 15:00) over the stock's mean share of that bin over days t-19 .. t, so 0 is
                       the usual historical-profile forecast. The labels also hold that mean (``volume_profile_avg_k``),
                       from which the 8 test predictions can be turned into a profile and scored by its L1 error
    return_cond        ``return`` given the realized market: the probe sees [x, x * m_t], where m_t is that date's
                       equal-weight mean label (standardized over train dates), so it ranks stocks by how they
                       respond to the market move the label window actually had. It measures whether the state
                       encodes market exposures (market-conditioned JEPA)
    return_5d_cond     the same for ``return_5d``
    beta_20d_idx       OLS beta on the HS300 index over days t+2 .. t+21, the hedge ratio of a tranche bought at
                       close(t+1); needs index closes, so ``labels`` does not build it; it only runs when the column is
                       added; fit on the level itself (clipped to +-3), since a hedge needs beta units, not a rank

The minute tasks are decided at the 11:30 morning close of day ``t`` instead (``MIDDAY_TIME``), so their next k bars,
13:01 .. 13:0k, hold no overnight gap or opening auction (after 15:00 they would be mostly the gap). Their labels are
same-day, for k in 5 / 30 / 60 / 120 (120 ends at the 15:00 close, so it is the whole afternoon session), and
dropped unless the day has the 11:30 bar and all 120 after it, and the k after it have finite closes:

    return_{k}min        close(13:0k) / close(11:30) - 1
    return_skip1_{k}min  close(13:0k) / close(13:01) - 1, the same without the first bar, so the bid-ask bounce and
                         lunch-break news of the 11:30 and 13:01 bars cannot drive it
    volatility_{k}min    std of the k 1-min close-to-close returns after 11:30 (the lunch break is one ordinary minute)
    volume_{k}min        mean volume per bar over the k bars after 11:30 / that of the day's bars up to 11:30
    beta_120min          OLS slope of the stock's 120 1-min returns after 11:30 on the market's, the equal-weight mean
                         of all stocks' (both clipped to +-20%); the same-day, intraday ``beta_20d``

Return and beta tasks fit the per-date z-scored (clipped) label, ``jump_1d`` the 0/1 label (and also reports a pooled
AUC, which then picks the ridge penalty), ``volume_profile_k`` the log deviation itself (clipped), ``beta_20d_idx`` the
level (clipped), the others its log.

Splits are by decision date: probe-train ``[--train-start, 20220701)``, eval ``[20220701, 20230101)`` to pick the
ridge penalty, test ``[20230101, 20260101)`` to report. Pretraining also stops before 20220701.

Feature sets: ``online`` (CLS state of the online encoder), ``hand`` (handcrafted statistics of
the same raw window, a baseline with the same information) and ``online+hand`` (adds back the price and volume scale
that per-window normalization hides from the encoder). Run ``embed --random-init`` for the untrained-encoder baseline.
``stats`` is exactly what the z-score removes (log mean and log coefficient of variation per bar feature), and
``hand_norm`` the ``hand`` analogues computed on the z-scored window the encoder sees, so ``hand_norm`` vs
``hand_norm+stats`` measures how much a task depends on that removed scale. ``window-features`` computes these
model-free sets on the CPU; ``probe --extra`` joins them to older embeddings, e.g. for ``online+stats``.

Reconstruction baselines (``qtsmmae``) embed the same way; ``embed --readout cls|mean`` picks the CLS state or the
mean patch state (written to ``embeddings-<readout>.npz``), since the masked objective never trains the CLS directly.

Panel checkpoints (a model with a peer pooler) embed the same windows: ``online`` is still the stock's own CLS state,
so it compares directly with a series checkpoint. ``embed`` then pools the ``online`` states of every stock embedded
at the same date, giving ``peer`` (the leave-one-out summary m_-i, with ``use_peer_context``) and ``market`` (the
panel summary m, the same for every stock of a date, so it only moves the non-cross-sectional tasks). ``probe`` adds
``online+peer+market`` and ``online+peer+market+hand`` when these are present. As in pretraining, each date's stocks
are pooled in random panels of at most the run's ``data.train.max_symbols`` stocks (``embed --panel-size``; 0 pools
the whole date). Pretraining panels only hold stocks that are eligible under the run's ``span_ratio``, while the
probe pools every stock with a 15:00 window.

Checkpoints with a panel mixer (``model.panel_mixer_layers``) also get ``mixed``: the mixer's states over the same
random date panels (of the run's ``context_symbols``, else ``max_symbols``, stocks; ``embed --panel-size``), and
``probe`` adds ``mixed`` and ``mixed+hand``. ``online`` stays the stock's own state.

Usage::

    PREFIX=<v3 OHLCV mmap prefix>
    python -m tools.eval_jepa_probe labels --prefix $PREFIX --out var/probe/labels.parquet
    python -m tools.eval_jepa_probe embed --checkpoint experiments/<run>/global_step_20000 --prefix $PREFIX
    python -m tools.eval_jepa_probe probe --embeddings experiments/<run>/global_step_20000/probe/embeddings.npz \\
        --labels var/probe/labels.parquet

For the minute tasks, embed (and compute ``window-features`` and the pretrained baselines) with
``--decision-time 113000``, which writes ``embeddings-1130.npz``. ``probe`` reads the decision time from the npz and
only runs the tasks decided then; every ``--extra`` must match it, and daily features (``alpha158``) count as
15:00 ones, since at 11:30 they would see the rest of the day.

``probe --model gbdt`` fits LightGBM instead of ridge (CPU), a nonlinear baseline for the model-free sets.

Feature baselines live in ``tools.baselines`` (one module each) and are written as npz like the embeddings, then
probed with ``--extra`` on an encoder's (or ``window-features``') npz, so they are scored on the same rows:

    python -m tools.eval_jepa_probe alpha158 --prefix $PREFIX --out var/probe/alpha158.npz
    python -m tools.eval_jepa_probe probe --embeddings <embeddings.npz> --extra var/probe/alpha158.npz \\
        --labels var/probe/labels.parquet --features alpha158,online+alpha158 [--model gbdt]

Pretrained-model baselines embed the same 15:00 windows; ``kronos`` (``tools.baselines.kronos``, Kronos-small by
default, lookback up to its 512-bar context) writes ``kronos`` (last-bar state) and ``kronos_mean``:

    python -m tools.eval_jepa_probe kronos --prefix $PREFIX --out var/probe/kronos-small-lb512.npz
    python -m tools.eval_jepa_probe probe --embeddings <embeddings.npz> --extra var/probe/kronos-small-lb512.npz \\
        --labels var/probe/labels.parquet --features kronos,kronos_mean,kronos+hand

``timesfm`` (``tools.baselines.timesfm``, TimesFM 2.5, lookback a multiple of its 32-bar patch) writes ``timesfm``
(last-patch state) and ``timesfm_mean``, each the [close, volume] states of the univariate model:

    python -m tools.eval_jepa_probe timesfm --prefix $PREFIX --out var/probe/timesfm-lb960.npz

``chronos2`` (``tools.baselines.chronos2``, Chronos-2, lookback a multiple of its 16-bar patch) writes ``chronos2``
(future-patch state) and ``chronos2_mean``, each the [close, volume] states of one multivariate (group-attended) series:

    python -m tools.eval_jepa_probe chronos2 --prefix $PREFIX --out var/probe/chronos2-lb960.npz

``moment`` (``tools.baselines.moment``, MOMENT-1-base by default, lookback a multiple of its 8-bar patch up to its
512-bar context) writes ``moment`` (mean patch state, its embedding readout) and ``moment_last``, each the
[close, volume] states of the univariate model:

    python -m tools.eval_jepa_probe moment --prefix $PREFIX --out var/probe/moment-lb512.npz

``probe`` writes ``metrics.json``, a ``metrics.txt`` table and, for the return tasks,
``pred-<task>-<features>-test.parquet`` (symbol,date,time,pred,label) for
backtesting.
"""

import itertools
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

import mc_jepa.models  # noqa: F401
from mc_jepa.datasets.v3.indexed_dataset import MMapIndexedDatasetV3
from mc_jepa.datasets.v3.series_dataset import StockSeriesDataset
from mc_jepa.datasets.v4 import indexed_dataset as v4_index
from mc_jepa.datasets.v4.series_dataset import CrossSectionSeriesDataset
from mc_jepa.trainer.panel_jepa_trainer import series_collate_fn
from tools.baselines import alpha158

EVAL_START, TEST_START, TEST_END = 20220701, 20230101, 20260101
DECISION_TIME = 150000
MIN_BARS_PER_DAY = 200  # volatility label needs most of the day's bars
JUMP_MOVE, MAX_DAILY_MOVE = 0.095, 0.205  # A-share limits are 10% / 20%; larger moves are corporate actions
MIDDAY_TIME, MINUTES = 113000, (5, 30, 60, 120)  # the minute tasks: the k bars after the morning close
PROFILE_BINS, PROFILE_DAYS = 8, 20  # volume profile: 30-min bins of the session, days of its baseline profile
# task -> (label column, regression target: "xs" per-date z-score, "log", "binary" 0/1, or "dev" / "level" the label)
TASKS = {
    "return": ("return_1d_delay_1d", "xs"),
    "return_5d": ("return_5d_delay_1d", "xs"),
    "return_20d": ("return_20d_delay_1d", "xs"),
    **{f"return_{k}min": (f"return_{k}min", "xs") for k in MINUTES},
    **{f"volatility_{k}min": (f"volatility_{k}min", "log") for k in MINUTES},
    **{f"return_skip1_{k}min": (f"return_skip1_{k}min", "xs") for k in MINUTES},
    **{f"volume_{k}min": (f"volume_{k}min", "log") for k in MINUTES},
    f"beta_{max(MINUTES)}min": (f"beta_{max(MINUTES)}min", "xs"),
    "beta_20d": ("beta_20d", "xs"),
    "jump_1d": ("jump_1d", "binary"),
    "volatility": ("volatility_1d", "log"),
    "volatility_5d": ("volatility_5d", "log"),
    "max_drawdown_5d": ("max_drawdown_5d", "log"),
    "volume": ("volume_ratio_1d", "log"),
    "volume_5d": ("volume_5d", "log"),
    "open_volume_share": ("open_volume_share_1d", "log"),
    **{f"volume_profile_{k}": (f"volume_profile_{k}", "dev") for k in range(1, PROFILE_BINS + 1)},
    "return_cond": ("return_1d_delay_1d", "xs_cond"),
    "return_5d_cond": ("return_5d_delay_1d", "xs_cond"),
    "beta_20d_idx": ("beta_20d_idx", "level"),
}
TASK_TIME = {task: MIDDAY_TIME for task in TASKS if task.endswith("min")}  # decision time, DECISION_TIME if absent
PROFILE_TASKS = tuple(task for task in TASKS if task.startswith("volume_profile_"))
# tasks whose test predictions are written, for the backtests and the profile error
PRED_TASKS = ("return", "return_5d", "return_20d", "beta_20d_idx") + PROFILE_TASKS
# defaults, each kept when all its parts are in the embeddings
FEATURE_SETS = (
    "online", "hand", "online+hand", "stats", "hand_norm", "hand_norm+stats", "online+stats", "alpha158",
    "online+alpha158", "kronos", "kronos_mean", "kronos+hand", "timesfm", "timesfm_mean", "timesfm+hand",
    "chronos2", "chronos2_mean", "chronos2+hand", "moment", "moment_last", "moment+hand",
)  # fmt: skip
HAND_FEATURES = (
    "ret_window",
    "ret_last30",
    "ret_last60",
    "rv_window",
    "rv_last30",
    "log_range",
    "close_in_range",
    "log_close",
    "log_volume",
    "volume_last30_ratio",
    "volume_cv",
)


# ---- labels ---------------------------------------------------------------------------------------


def _ahead(v: np.ndarray, k: int, day_idx: np.ndarray) -> np.ndarray:
    """v of the stock's day t+k, NaN unless days t..t+k are consecutive market days (day_idx in the calendar)."""
    out = np.full(len(v), np.nan)
    if len(v) > k:
        out[:-k] = np.where(day_idx[k:] == day_idx[:-k] + k, v[k:], np.nan)
    return out


def _realized_vol(stats: tuple, days: range, day_idx: np.ndarray) -> np.ndarray:
    """Std of the 1-min returns pooled over days t+k, k in ``days``, from per-day (count, sum, sum of squares)."""
    n, s1, s2 = (sum(_ahead(v, k, day_idx) for k in days) for v in stats)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = s2 / n - (s1 / n) ** 2
    return np.where(n >= MIN_BARS_PER_DAY * len(days), np.sqrt(np.clip(var, 0, None)), np.nan)


def _max_drawdown(day_close, day_max, day_min, day_mdd, days: range, day_idx: np.ndarray) -> np.ndarray:
    """Largest drop from the running peak of the 1-min closes over days t+k, k in ``days``, starting at close(t)."""
    peak, mdd = day_close, np.zeros(len(day_close))
    for k in days:
        mdd = np.maximum(mdd, _ahead(day_mdd, k, day_idx))  # peak and trough within day t+k
        mdd = np.maximum(mdd, 1 - _ahead(day_min, k, day_idx) / peak)  # peak before day t+k
        peak = np.maximum(peak, _ahead(day_max, k, day_idx))
    return mdd


def _beta(ret: np.ndarray, market: np.ndarray, days: range, day_idx: np.ndarray) -> np.ndarray:
    """OLS slope of the stock's daily returns on the market's over days t+k, k in ``days``."""
    n = len(days)
    sx, sy, sxx, sxy = (sum(_ahead(v, k, day_idx) for k in days) for v in (market, ret, market**2, market * ret))
    var = n * sxx - sx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0, (n * sxy - sx * sy) / var, np.nan)


def _after_bar(close: np.ndarray, volume: np.ndarray, tod: np.ndarray, day_id: np.ndarray, time: int, n: int):
    """Per day: the closes [days, n + 1] of its bar at ``time`` (HHMMSS) and the n bars after it, the volumes of those
    n bars [days, n] and the mean volume per bar of the day up to and including that bar [days]; NaN unless the day
    has that bar and n more."""
    days = day_id[-1] + 1
    window, after, before = np.full((days, n + 1), np.nan), np.full((days, n), np.nan), np.full(days, np.nan)
    i = np.flatnonzero(tod == time)
    i = i[(i + n < len(close)) & (day_id[np.minimum(i + n, len(close) - 1)] == day_id[i])]
    window[day_id[i]] = close[i[:, None] + np.arange(n + 1)]
    after[day_id[i]] = volume[i[:, None] + np.arange(1, n + 1)]
    start = np.flatnonzero(np.r_[True, day_id[1:] != day_id[:-1]])[day_id[i]]
    cum = np.cumsum(volume)
    before[day_id[i]] = (cum[i] - cum[start] + volume[start]) / (i - start + 1)
    return window, after, before


def _minute_beta(r: np.ndarray, m: np.ndarray, min_bars: int) -> np.ndarray:
    """Per row: OLS slope of the stock's 1-min returns r [rows, n] on the market's m [rows, n] over the bars where both
    are finite; NaN with fewer than ``min_bars`` such bars, or a flat stock or market."""
    ok = np.isfinite(r) & np.isfinite(m)
    n = ok.sum(axis=1)
    x, y = np.where(ok, m, 0.0), np.where(ok, r, 0.0)
    sx, sy = x.sum(axis=1), y.sum(axis=1)
    var_x, var_y = n * (x * x).sum(axis=1) - sx**2, n * (y * y).sum(axis=1) - sy**2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (n * (x * y).sum(axis=1) - sx * sy) / var_x
    return np.where((n >= min_bars) & (var_x > 0) & (var_y > 0), beta, np.nan)


def _volume_profile(volume: np.ndarray, tod: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Per day [days, PROFILE_BINS]: each bin's share of the day's volume; NaN unless the day has MIN_BARS_PER_DAY bars
    and volume in every bin."""
    minute = tod // 10000 * 60 + tod // 100 % 100
    session = np.where(minute <= 11 * 60 + 30, minute - (9 * 60 + 31), minute - (13 * 60 + 1) + 120)  # 0 .. 239
    cell = day_id * PROFILE_BINS + np.clip(session, 0, 239) * PROFILE_BINS // 240
    size = (day_id[-1] + 1) * PROFILE_BINS
    vol = np.bincount(cell, weights=volume, minlength=size).reshape(-1, PROFILE_BINS)
    bars = np.bincount(day_id, minlength=day_id[-1] + 1)
    ok = (bars >= MIN_BARS_PER_DAY) & (vol > 0).all(axis=1)
    return np.where(ok[:, None], vol / np.where(ok, vol.sum(axis=1), 1)[:, None], np.nan)


def compute_labels(data: MMapIndexedDatasetV3) -> pd.DataFrame:
    """Daily labels per (symbol, decision date) from the 1-min bars."""
    timeline = data.timeline
    calendar = np.unique(timeline // 1_000_000)
    frames, daily, minute = [], [], []
    minute_sum, minute_count = np.zeros((2, len(calendar), max(MINUTES)))
    for s in tqdm(range(data.num_symbols), desc="labels"):
        a, b = data.symbol_range(s)
        if b - a < 2:
            continue
        bars = data.read(a, b, ["close", "volume"], dtype=np.float64)
        close, volume = np.where(bars[:, 0] > 0, bars[:, 0], np.nan), bars[:, 1]
        ts = timeline[np.asarray(data.time_ids[a:b])]
        date = ts // 1_000_000
        starts = np.flatnonzero(np.r_[True, date[1:] != date[:-1]])
        ends = np.r_[starts[1:], len(date)]
        day_id = np.repeat(np.arange(len(starts)), ends - starts)

        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.r_[np.nan, close[1:] / close[:-1] - 1]
        valid = np.isfinite(ret)
        r = np.where(valid, ret, 0.0)
        count = np.add.reduceat(valid.astype(np.int64), starts).astype(np.float64)
        sum_r = np.add.reduceat(r, starts)
        sum_r2 = np.add.reduceat(r * r, starts)

        day = date[starts]
        day_idx = np.searchsorted(calendar, day)
        day_close = close[ends - 1]
        day_max, day_min = np.fmax.reduceat(close, starts), np.fmin.reduceat(close, starts)
        day_mdd = np.fmax.reduceat(1 - close / pd.Series(close).groupby(day_id).cummax().to_numpy(), starts)
        day_volume = np.add.reduceat(volume, starts)
        day_volume = np.where(day_volume > 0, day_volume, np.nan)
        cum_volume = np.cumsum(volume)
        open_volume = cum_volume[np.minimum(starts + 30, ends) - 1] - cum_volume[starts] + volume[starts]
        open_share = np.where(ends - starts >= MIN_BARS_PER_DAY, open_volume / day_volume, np.nan)
        day_ret = _ahead(day_close, 1, day_idx) / day_close - 1  # day t+1's return, at row t
        day_ret = np.where(np.abs(day_ret) <= MAX_DAILY_MOVE, day_ret, np.nan)

        stats = (count, sum_r, sum_r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            labels = {
                "return_1d_delay_1d": _ahead(day_close, 2, day_idx) / _ahead(day_close, 1, day_idx) - 1,
                "return_5d_delay_1d": _ahead(day_close, 6, day_idx) / _ahead(day_close, 1, day_idx) - 1,
                "return_20d_delay_1d": _ahead(day_close, 21, day_idx) / _ahead(day_close, 1, day_idx) - 1,
                "jump_1d": np.where(np.isfinite(day_ret), (np.abs(day_ret) >= JUMP_MOVE).astype(np.float64), np.nan),
                "volatility_1d": _realized_vol(stats, range(1, 2), day_idx),
                "volatility_5d": _realized_vol(stats, range(1, 6), day_idx),
                "max_drawdown_5d": _max_drawdown(day_close, day_max, day_min, day_mdd, range(1, 6), day_idx),
                "volume_ratio_1d": _ahead(day_volume, 1, day_idx) / day_volume,
                "volume_5d": sum(_ahead(day_volume, k, day_idx) for k in range(1, 6)) / 5 / day_volume,
                "open_volume_share_1d": _ahead(open_share, 1, day_idx),
            }
            # same-day labels of the MIDDAY_TIME decision
            window, after, before = _after_bar(close, volume, ts % 1_000_000, day_id, MIDDAY_TIME, max(MINUTES))
            r = window[:, 1:] / window[:, :-1] - 1
            for k in MINUTES:
                ok = np.isfinite(r[:, :k]).all(axis=1)
                labels[f"return_{k}min"] = np.where(ok, window[:, k] / window[:, 0] - 1, np.nan)
                labels[f"return_skip1_{k}min"] = np.where(ok, window[:, k] / window[:, 1] - 1, np.nan)
                labels[f"volatility_{k}min"] = np.where(ok, r[:, :k].std(axis=1), np.nan)
                labels[f"volume_{k}min"] = np.where(before > 0, after[:, :k].mean(axis=1) / before, np.nan)
            share = _volume_profile(volume, ts % 1_000_000, day_id)
            profile = pd.DataFrame(share).rolling(PROFILE_DAYS).mean().to_numpy()  # days t-19 .. t, known at t
            for k in range(PROFILE_BINS):
                labels[f"volume_profile_{k + 1}"] = np.log(_ahead(share[:, k], 1, day_idx) / profile[:, k])
                labels[f"volume_profile_avg_{k + 1}"] = profile[:, k]
        frame = pd.DataFrame({k: v.astype(np.float32) for k, v in labels.items()})
        frame.insert(0, "date", day.astype(np.int32))
        frame.insert(0, "symbol", data.symbols[s])
        frames.append(frame)
        # day t's own return, at row t, for the market series and beta
        own_ret = np.full(len(day), np.nan)
        own_ret[1:] = day_ret[:-1]
        daily.append((day_idx, np.clip(own_ret, -MAX_DAILY_MOVE, MAX_DAILY_MOVE)))
        r = np.clip(r, -MAX_DAILY_MOVE, MAX_DAILY_MOVE).astype(np.float32)  # the afternoon 1-min returns
        minute_sum[day_idx] += np.nan_to_num(r)
        minute_count[day_idx] += np.isfinite(r)
        minute.append(r)

    # equal-weight market return per calendar day, then each stock's forward beta on it
    all_idx, all_ret = (np.concatenate(v) for v in zip(*daily, strict=True))
    ok = np.isfinite(all_ret)
    total = np.bincount(all_idx[ok], minlength=len(calendar))
    sums = np.bincount(all_idx[ok], weights=all_ret[ok], minlength=len(calendar))
    market = np.where(total > 0, sums / np.maximum(total, 1), np.nan)
    for frame, (day_idx, ret) in zip(frames, daily, strict=True):
        frame["beta_20d"] = _beta(ret, market[day_idx], range(1, 21), day_idx).astype(np.float32)
    # the same for the afternoon 1-min returns after MIDDAY_TIME, and each stock's same-day beta on them
    with np.errstate(divide="ignore", invalid="ignore"):
        minute_market = np.where(minute_count > 0, minute_sum / minute_count, np.nan)
    for frame, (day_idx, _), r in zip(frames, daily, minute, strict=True):
        k = max(MINUTES)
        frame[f"beta_{k}min"] = _minute_beta(r, minute_market[day_idx], k * 5 // 6).astype(np.float32)
    return pd.concat(frames, ignore_index=True)


# ---- embeddings -----------------------------------------------------------------------------------


def handcrafted(context: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, features: list[str]) -> torch.Tensor:
    """Statistics of the raw window [B, L, F] (de-normalized), the no-learning baseline: [B, len(HAND_FEATURES)]."""
    raw = context * std[:, None] + mean[:, None]
    col = {name: raw[..., features.index(name)] for name in ("open", "high", "low", "close", "volume")}
    close = col["close"].clamp_min(1e-6)
    logret = torch.diff(close.log(), dim=1)
    high, low = col["high"].amax(dim=1).clamp_min(1e-6), col["low"].amin(dim=1).clamp_min(1e-6)
    volume = col["volume"].clamp_min(0)
    vol_mean = volume.mean(dim=1)
    L = close.shape[1]  # trailing returns start at the window's first bar when it is shorter
    out = [
        close[:, -1].log() - close[:, 0].log(),
        close[:, -1].log() - close[:, -min(31, L)].log(),
        close[:, -1].log() - close[:, -min(61, L)].log(),
        logret.std(dim=1),
        logret[:, -30:].std(dim=1),
        high.log() - low.log(),
        (close[:, -1] - low) / (high - low).clamp_min(1e-6),
        close[:, -1].log(),
        vol_mean.log1p(),
        volume[:, -30:].mean(dim=1).log1p() - vol_mean.log1p(),
        volume.std(dim=1) / vol_mean.clamp_min(1.0),
    ]
    return torch.stack(out, dim=1)


def window_stats(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """What the per-window z-score removes: log mean and log coefficient of variation per feature, [B, 2F]."""
    log_mean = mean.clamp_min(1e-6).log()
    return torch.cat([log_mean, std.clamp_min(1e-6).log() - log_mean], dim=1)


def handcrafted_norm(context: torch.Tensor, features: list[str]) -> torch.Tensor:
    """``handcrafted`` analogues on the z-scored window the encoder sees (differences for log returns, no levels)."""
    col = {name: context[..., features.index(name)] for name in ("open", "high", "low", "close", "volume")}
    close, volume = col["close"], col["volume"]
    diff = torch.diff(close, dim=1)
    high, low = col["high"].amax(dim=1), col["low"].amin(dim=1)
    L = close.shape[1]
    out = [
        close[:, -1] - close[:, 0],
        close[:, -1] - close[:, -min(31, L)],
        close[:, -1] - close[:, -min(61, L)],
        diff.std(dim=1),
        diff[:, -30:].std(dim=1),
        high - low,
        (close[:, -1] - low) / (high - low).clamp_min(1e-6),
        close[:, -1],
        volume[:, -30:].mean(dim=1),
        volume[:, -1],
    ]
    return torch.stack(out, dim=1)


def date_panels(date: np.ndarray, panel_size: int | None = None, seed: int = 0) -> list[list[np.ndarray]]:
    """Row indices of each date's panels: per date, a list of row arrays.

    With ``panel_size``, each date's stocks are shuffled (seeded by ``seed`` and the date) and split into
    near-equal panels of at most ``panel_size`` stocks, as in pretraining with ``max_symbols``; otherwise the
    whole date is one panel.
    """
    order = np.argsort(date, kind="stable")
    starts = np.flatnonzero(np.r_[True, date[order][1:] != date[order][:-1]])
    out = []
    for rows in np.split(order, starts[1:]):
        n = len(rows)
        if panel_size and n > panel_size:
            rows = np.random.default_rng(seed + int(date[rows[0]])).permutation(rows)
            out.append(np.array_split(rows, -(-n // panel_size)))
        else:
            out.append([rows])
    return out


def _stack_panels(online: np.ndarray, panels: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded states [G, size, D] and mask [G, size] of one date's panels."""
    size = max(len(r) for r in panels)
    z = torch.zeros(len(panels), size, online.shape[1], device=device)
    mask = torch.zeros(len(panels), size, dtype=torch.bool, device=device)
    for g, r in enumerate(panels):
        z[g, : len(r)] = torch.from_numpy(online[r]).to(device, torch.float32)
        mask[g, : len(r)] = True
    return z, mask


@torch.no_grad()
def pool_by_date(
    model, online: np.ndarray, date: np.ndarray, panel_size: int | None = None, seed: int = 0
) -> dict[str, np.ndarray]:
    """Panel summaries of each date's cross-section of ``online`` states: ``market`` m and ``peer`` m_-i (if any),
    pooled over the panels of ``date_panels``."""
    device = next(model.parameters()).device
    out = {"market": np.zeros_like(online)}
    if model.config.use_peer_context:
        out["peer"] = np.zeros_like(online)
    for panels in tqdm(date_panels(date, panel_size, seed), desc="pool"):
        z, mask = _stack_panels(online, panels, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            market, peers = model._pool(z, mask)
        market, peers = market.float().cpu().numpy(), None if peers is None else peers.float().cpu().numpy()
        for g, r in enumerate(panels):
            out["market"][r] = market[g]
            if peers is not None:
                out["peer"][r] = peers[g, : len(r)]
    return out


@torch.no_grad()
def mix_by_date(
    model, online: np.ndarray, date: np.ndarray, panel_size: int | None = None, seed: int = 0
) -> dict[str, np.ndarray]:
    """``mixed``: the panel mixer's states of each date's ``online`` states, over the panels of ``date_panels``."""
    device = next(model.parameters()).device
    out = np.zeros_like(online)
    for panels in tqdm(date_panels(date, panel_size, seed), desc="mix"):
        z, mask = _stack_panels(online, panels, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            mixed = model.mix(z, mask).float().cpu().numpy()
        for g, r in enumerate(panels):
            out[r] = mixed[g, : len(r)]
    return {"mixed": out}


@torch.no_grad()
def compute_embeddings(
    model,
    dataset: StockSeriesDataset | CrossSectionSeriesDataset,
    batch_size: int,
    num_workers: int,
    encode_kwargs: dict | None = None,
    panel_size: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-window features at every decision time; ``model=None`` skips ``online`` (model-free features only).
    A v4 ``CrossSectionSeriesDataset`` is batched one decision time (all its stocks) at a time, not ``batch_size``."""
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    if isinstance(dataset, CrossSectionSeriesDataset):
        loader = DataLoader(
            dataset, batch_size=1, num_workers=num_workers, collate_fn=dataset.collate_fn, pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=series_collate_fn, pin_memory=True
        )
    keys = ("symbol_id", "date", "hand", "hand_norm", "stats") + (("online",) if model is not None else ())
    out = {k: [] for k in keys}
    for batch in tqdm(loader, desc="embed"):
        context, mask = batch["context"].to(device, non_blocking=True), batch["symbol_mask"].to(device)
        if model is not None:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scale = {"mean": batch["mean"].to(device), "std": batch["std"].to(device)}
                daily = batch["daily"].to(device) if "daily" in batch else None  # with model.daily_lookback
                online = model.encode(context, mask, **scale, daily=daily, **(encode_kwargs or {}))[0][:, 0]
            out["online"].append(online.float().cpu().numpy().astype(np.float16))
        window, mean, std = context[:, 0].float(), batch["mean"][:, 0].to(device), batch["std"][:, 0].to(device)
        out["hand"].append(handcrafted(window, mean, std, dataset.features).cpu().numpy().astype(np.float32))
        out["hand_norm"].append(handcrafted_norm(window, dataset.features).cpu().numpy().astype(np.float32))
        out["stats"].append(window_stats(mean, std).cpu().numpy().astype(np.float32))
        out["symbol_id"].append(batch["symbol_ids"][:, 0].numpy())
        out["date"].append((batch["timestamp"] // 1_000_000).numpy())
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["symbol"] = np.asarray(dataset.data.symbols)[arrays.pop("symbol_id")]
    arrays["date"] = arrays["date"].astype(np.int32)
    if model is not None and getattr(model, "peer_pooler", None) is not None:
        arrays.update(pool_by_date(model, arrays["online"], arrays["date"], panel_size, seed))
        arrays["panel_size"] = np.asarray(panel_size or 0)  # 0: whole date; a scalar, dropped by ``probe``
    if model is not None and getattr(model, "panel_mixer", None) is not None:
        arrays.update(mix_by_date(model, arrays["online"], arrays["date"], panel_size, seed))
        arrays["panel_size"] = np.asarray(panel_size or 0)
    return arrays


def save_npz(out: Path, arrays: dict[str, np.ndarray], decision_time: int | None = None):
    """Zip-compressed (repeated ``symbol`` / ``date`` keys shrink to <3%); ``np.load`` reads it like ``np.savez``.

    ``decision_time`` (HHMMSS) is stored for windowed features, so ``probe`` scores them only on its tasks; daily
    features (no time) count as ``DECISION_TIME`` ones, as they see the whole day.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if decision_time is not None:
        arrays = {**arrays, "decision_time": np.asarray(decision_time)}
    np.savez_compressed(out, **arrays)
    print(f"-> {out}")


# ---- probes (on the GPU) --------------------------------------------------------------------------


class Days:
    """Segment ops over the cross-section of each date, for rows sorted by date."""

    def __init__(self, date: torch.Tensor):
        self.dates, self.ids, self.counts = torch.unique_consecutive(date, return_inverse=True, return_counts=True)

    def _expand(self, v: torch.Tensor) -> torch.Tensor:
        return self.counts.view(-1, *[1] * (v.ndim - 1)).to(v.dtype)

    def sum(self, v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(self.dates), *v.shape[1:], dtype=v.dtype, device=v.device)
        return out.index_add_(0, self.ids, v)

    def demean(self, v: torch.Tensor) -> torch.Tensor:
        return v - (self.sum(v) / self._expand(v))[self.ids]

    def rank(self, v: torch.Tensor) -> torch.Tensor:
        """Average-tie ranks of v [N] within each date, up to a per-date offset (removed by ``demean``)."""
        order = torch.argsort(v, stable=True)
        order = order[torch.argsort(self.ids[order], stable=True)]
        vs, ds = v[order], self.ids[order]
        new_run = torch.ones_like(vs, dtype=torch.bool)
        new_run[1:] = (vs[1:] != vs[:-1]) | (ds[1:] != ds[:-1])
        run = torch.cumsum(new_run, 0) - 1
        pos = torch.arange(len(v), device=v.device, dtype=torch.float64)
        mean_pos = torch.zeros(int(run[-1]) + 1, dtype=torch.float64, device=v.device).index_add_(0, run, pos)
        mean_pos /= torch.bincount(run)
        out = torch.empty_like(pos)
        out[order] = mean_pos[run]
        return out

    def zscore(self, v: torch.Tensor) -> torch.Tensor:
        """Per-date z-score with the sample std (ddof=1)."""
        c = self.demean(v)
        std = (self.sum(c * c) / (self._expand(v) - 1).clamp_min(1)).sqrt()
        return c / (std[self.ids] + 1e-8)

    def corr(self, x: torch.Tensor, y: torch.Tensor, rank: bool) -> torch.Tensor:
        """Per-date Pearson (Spearman with ``rank``) correlation of x [N] and y [N]; NaN for dates under 20 rows."""
        x, y = x.double(), y.double()
        if rank:
            x, y = self.rank(x), self.rank(y)
        x, y = self.demean(x), self.demean(y)
        ic = self.sum(x * y) / (self.sum(x * x) * self.sum(y * y)).sqrt()
        return torch.where(self.counts >= 20, ic, torch.nan)


def pooled_auc(pred: torch.Tensor, label: torch.Tensor) -> float:
    """ROC AUC of a 0/1 label over all rows."""
    rank = torch.empty_like(pred, dtype=torch.float64)
    rank[torch.argsort(pred)] = torch.arange(1, len(pred) + 1, device=pred.device, dtype=torch.float64)
    pos = label > 0.5
    n1, n0 = pos.sum().item(), (~pos).sum().item()
    return (rank[pos].sum().item() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def score(days: Days, pred: torch.Tensor, label: torch.Tensor, target: torch.Tensor, binary: bool = False) -> dict:
    """Daily IC / RankIC against the raw label, R^2 against the regression target, and AUC for a 0/1 label."""
    ic = pd.Series(days.corr(pred, label, rank=False).cpu().numpy(), index=days.dates.cpu().numpy()).dropna()
    rank_ic = pd.Series(days.corr(pred, label, rank=True).cpu().numpy(), index=days.dates.cpu().numpy()).dropna()
    resid = (target - pred).double()
    return {
        "ic": ic.mean(),
        "icir": ic.mean() / ic.std(),
        "rank_ic": rank_ic.mean(),
        "rank_icir": rank_ic.mean() / rank_ic.std(),
        "r2": (1 - resid.var() / target.double().var()).item(),
        "days": len(ic),
        "rows": len(label),
        "rank_ic_by_year": {int(k): v for k, v in rank_ic.groupby(rank_ic.index // 10000).mean().items()},
        "auc": pooled_auc(pred, label) if binary else float("nan"),
    }


def fit_target(task: str, days: Days, label: torch.Tensor) -> torch.Tensor:
    """Regression target: per-date z-scored label (clipped), its log, the 0/1 label itself, or a log deviation (clipped
    at a factor of e^3 ~ 20) or level (clipped at +-3)."""
    if TASKS[task][1] == "log":
        return label.clamp_min(1e-5).log()
    if TASKS[task][1] in ("dev", "level"):
        return label.clamp(-3, 3).float()
    if TASKS[task][1] == "binary":
        return label.float()
    return days.zscore(label).clamp(-5, 5).float()  # xs, xs_cond


def ridge_path(x: torch.Tensor, y: torch.Tensor, alphas: np.ndarray, chunk: int = 1 << 18):
    """Weights [len(alphas), D] and intercepts for centered ridge, from one float64 eigendecomposition."""
    n, d = x.shape
    # chunked float64 mean: x.double() would copy all of x (10 GB at 512 dims)
    x_mean = sum(x[i : i + chunk].double().sum(dim=0) for i in range(0, n, chunk)) / n
    y_mean = y.double().mean()
    gram = torch.zeros(d, d, dtype=torch.float64, device=x.device)
    xty = torch.zeros(d, dtype=torch.float64, device=x.device)
    for i in range(0, n, chunk):
        xc = x[i : i + chunk].double() - x_mean
        gram += xc.T @ xc
        xty += xc.T @ (y[i : i + chunk].double() - y_mean)
    eig, vec = torch.linalg.eigh(gram)
    proj = vec.T @ xty
    w = torch.stack([vec @ (proj / (eig + a * n)) for a in alphas])
    return w.float(), (y_mean - w @ x_mean).float()


FEATURE_CHUNK = 1 << 18  # rows per block when masking / gathering features


def gather_rows(
    parts: list[np.ndarray], rows: np.ndarray, device: torch.device, scale: np.ndarray | None = None
) -> torch.Tensor:
    """float32 [len(rows), D] of the feature parts side by side (then each times ``scale`` of its row, in float64),
    filled block by block on ``device``: the full-size host copies of a concatenation are never made."""
    d = sum(part.shape[1] for part in parts)
    out = torch.empty(len(rows), d if scale is None else 2 * d, dtype=torch.float32, device=device)
    for i in range(0, len(rows), FEATURE_CHUNK):
        r = rows[i : i + FEATURE_CHUNK]
        block = np.concatenate([part[r] for part in parts], axis=1)
        out[i : i + len(r), :d] = torch.from_numpy(block).to(device, torch.float32)
        if scale is not None:
            out[i : i + len(r), d:] = torch.from_numpy(block * scale[r, None]).to(device, torch.float32)
    return out


def probe_splits(
    emb: dict[str, np.ndarray], label: np.ndarray, task: str, features: str, train_start: int, device: torch.device
) -> dict[str, dict]:
    """Train / eval / test rows of ``emb`` (sorted by date) with finite features and label, and their targets."""
    date = emb["date"]
    parts = [emb[name] for name in features.split("+")]  # never concatenated: 35 GB at 2560 fp16 dims
    keep = np.isfinite(label)
    for part in parts:
        for i in range(0, len(part), FEATURE_CHUNK):
            keep[i : i + FEATURE_CHUNK] &= np.isfinite(part[i : i + FEATURE_CHUNK]).all(axis=1)
    bounds = {"train": (train_start, EVAL_START), "eval": (EVAL_START, TEST_START), "test": (TEST_START, TEST_END)}
    scale = None
    if TASKS[task][1] == "xs_cond":
        # realized market of each row's label window: the date's equal-weight mean label (clipped like beta_20d)
        clipped = pd.Series(np.clip(label, -0.2, 0.2)).where(np.isfinite(label))
        market = clipped.groupby(date).transform("mean").to_numpy()
        train_dates = np.unique(date[(date >= train_start) & (date < EVAL_START)])
        daily = pd.Series(market).groupby(date).first()
        market = market / daily.loc[train_dates].std()
        keep &= np.isfinite(market)
        scale = np.nan_to_num(market)  # features followed by features x market

    data = {}
    for name, (lo, hi) in bounds.items():
        rows = np.flatnonzero(keep & (date >= lo) & (date < hi))
        if len(rows) == 0:
            continue
        days = Days(torch.from_numpy(date[rows]).to(device))
        y = torch.from_numpy(label[rows]).to(device)
        data[name] = {
            "rows": rows,
            "days": days,
            "x": gather_rows(parts, rows, device, scale),
            "label": y,
            "target": fit_target(task, days, y),
        }
    return data


def eval_metric(split: dict, pred: torch.Tensor, task: str) -> float:
    """What picks the probe's hyperparameter: pooled AUC for a 0/1 label, minus the MSE for a log deviation (a
    profile, scored by its error, which RankIC cannot see shrink) or a level (a hedge ratio), else mean daily RankIC."""
    if TASKS[task][1] == "binary":
        return pooled_auc(pred, split["label"])
    if TASKS[task][1] in ("dev", "level"):
        return -(split["target"] - pred.to(split["target"].dtype)).square().mean().item()
    return split["days"].corr(pred, split["label"], rank=True).nanmean().item()


def npz_decision_time(npz) -> int:
    """The decision time ``save_npz`` stored (``DECISION_TIME`` for daily features and files from before it)."""
    return int(npz["decision_time"]) if "decision_time" in npz else DECISION_TIME


def probe_result(
    emb: dict[str, np.ndarray], data: dict[str, dict], task: str, features: str, binary: bool, **info
) -> dict:
    """Scores of each split's ``pred`` and the test predictions for backtesting."""
    result = {"task": task, "features": features, **info, "splits": {}}
    for name, split in data.items():
        result["splits"][name] = score(split["days"], split["pred"], split["label"], split["target"], binary)
    result["test_pred"] = None
    if "test" in data:
        rows = data["test"]["rows"]
        result["test_pred"] = pd.DataFrame(
            {
                "symbol": emb["symbol"][rows],
                "date": emb["date"][rows],
                "time": TASK_TIME.get(task, DECISION_TIME),
                "pred": data["test"]["pred"].cpu().numpy(),
                "label": data["test"]["label"].cpu().numpy(),
            }
        )
    return result


def run_probe(emb: dict[str, np.ndarray], label: np.ndarray, task: str, features: str, train_start: int) -> dict:
    """Ridge probe; ``emb`` rows sorted by date, ``label`` aligned with them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = probe_splits(emb, label, task, features, train_start, device)
    train = data["train"]
    x_mean, x_std = train["x"].mean(dim=0), train["x"].std(dim=0) + 1e-6
    for split in data.values():
        split["x"].sub_(x_mean).div_(x_std).clamp_(-10, 10)  # in place: one copy of x fits a 24 GB GPU

    alphas = np.logspace(-6, 4, 11)  # times n; standardized inputs
    w, b = ridge_path(train["x"], train["target"], alphas)
    binary = TASKS[task][1] == "binary"
    eval_pred = data["eval"]["x"] @ w.T + b
    best = int(np.nanargmax([eval_metric(data["eval"], eval_pred[:, k], task) for k in range(len(alphas))]))
    for split in data.values():
        split["pred"] = split["x"] @ w[best] + b[best]
    return probe_result(emb, data, task, features, binary, dim=train["x"].shape[1], alpha=alphas[best])


GBDT_PARAMS = {
    "learning_rate": 0.05,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "seed": 0,
    "verbosity": -1,
}


def run_gbdt_probe(
    emb: dict[str, np.ndarray],
    label: np.ndarray,
    task: str,
    features: str,
    train_start: int,
    leaves: tuple[int, ...] = (15, 63, 255),
    max_rounds: int = 2000,
    step: int = 50,
    patience: int = 200,
) -> dict:
    """LightGBM probe on the CPU, the nonlinear counterpart of ``run_probe`` on the same splits and targets.

    Each leaf count trains with early stopping on the eval loss; the leaf count and the number of trees (in steps of
    ``step``) are then picked on the eval metric, as the ridge penalty is.
    """
    import lightgbm as lgb

    data = probe_splits(emb, label, task, features, train_start, torch.device("cpu"))
    binary = TASKS[task][1] == "binary"
    x = {name: split["x"].numpy() for name, split in data.items()}
    train_set = lgb.Dataset(x["train"], data["train"]["target"].numpy(), free_raw_data=False)
    eval_set = lgb.Dataset(x["eval"], data["eval"]["target"].numpy(), reference=train_set)
    params = {**GBDT_PARAMS, "objective": "binary" if binary else "regression"}
    best = (-np.inf, None, None, None)  # eval metric, leaves, trees, booster
    for n in leaves:
        booster = lgb.train(
            {**params, "num_leaves": n},
            train_set,
            max_rounds,
            valid_sets=[eval_set],
            callbacks=[lgb.early_stopping(patience, verbose=False)],
            keep_training_booster=True,  # keep the trees past the eval-loss optimum for the eval-metric pick
        )
        raw, done = np.zeros(len(x["eval"])), 0  # summed tree outputs, monotone in the prediction
        while done < booster.current_iteration():
            k = min(done + step, booster.current_iteration())
            raw += booster.predict(x["eval"], start_iteration=done, num_iteration=k - done, raw_score=True)
            done = k
            metric = eval_metric(data["eval"], torch.from_numpy(raw), task)
            if metric > best[0]:
                best = (metric, n, k, booster)
    _, n, k, booster = best
    for name, split in data.items():
        split["pred"] = torch.from_numpy(booster.predict(x[name], num_iteration=k)).float()
    return probe_result(emb, data, task, features, binary, dim=x["train"].shape[1], num_leaves=n, num_trees=k)


# ---- CLI ------------------------------------------------------------------------------------------


@click.group()
def cli():
    pass


decision_time_option = click.option(
    "--decision-time",
    type=click.Choice([str(DECISION_TIME), str(MIDDAY_TIME)]),
    default=str(DECISION_TIME),
    show_default=True,
    callback=lambda ctx, param, value: int(value),
    help=f"HHMMSS the windows end at; {MIDDAY_TIME} for the minute tasks",
)


def time_suffix(decision_time: int) -> str:
    """File-name suffix of a non-default decision time, e.g. ``-1130``."""
    return "" if decision_time == DECISION_TIME else f"-{decision_time // 100:04d}"


@cli.command()
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def labels(prefix: str, out: Path):
    """Daily labels of every task for every (symbol, date)."""
    df = compute_labels(MMapIndexedDatasetV3(prefix))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows, {df.symbol.nunique()} symbols -> {out}")
    print(df[[column for column, _ in TASKS.values()]].describe().T.to_string())


def is_v4(prefix: str) -> bool:
    """A v4 (time-major) mmap, read with ``CrossSectionSeriesDataset``; else v3 (``StockSeriesDataset``)."""
    with open(v4_index.index_file_path(prefix), "rb") as f:
        return f.read(len(v4_index.MAGIC)) == v4_index.MAGIC


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--prefix",
    required=True,
    help="v3 mmap prefix (without .bin/.idx), or v4: the same windows read one decision time at a time",
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), help="default: <checkpoint>/probe/*.npz")
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--random-init", is_flag=True, help="untrained encoder with the checkpoint's architecture")
@click.option("--seed", default=0, show_default=True)
@click.option("--batch-size", default=4096, show_default=True)
@click.option("--num-workers", default=10, show_default=True)
@click.option(
    "--readout",
    type=click.Choice(["cls", "mean", "last", "max"]),
    help="CLS, mean, last (causal qtsmar) or max (qtsmts2vec) patch state (default: config.readout)",
)
@click.option(
    "--panel-size",
    type=int,
    help="panel models: pool / mix each date in random panels of at most this many stocks; 0 uses the whole date "
    "[default: the run's model.context_symbols, else data.train.max_symbols]",
)
@decision_time_option
def embed(
    checkpoint: Path, prefix, out, start, end, random_init, seed, batch_size, num_workers, readout, panel_size,
    decision_time,
):  # fmt: skip
    """Frozen encoder states at every decision time."""
    run_config = OmegaConf.load(checkpoint.parent / "config.yaml")
    if random_init:
        torch.manual_seed(seed)
        model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint))
    else:
        model = AutoModel.from_pretrained(checkpoint)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    series_dataset = CrossSectionSeriesDataset if is_v4(prefix) else StockSeriesDataset
    dataset = series_dataset(
        data_prefix=prefix,
        features=list(run_config.data.features),
        lookback=run_config.data.lookback,
        times_of_day=[decision_time],
        start_timestamp=start * 1_000_000,
        end_timestamp=end * 1_000_000,
        daily_dir=run_config.data.get("daily_dir"),  # daily bars, as in pretraining (model.daily_lookback)
        daily_lookback=run_config.model.get("daily_lookback", 0),
        daily_feature_set=run_config.model.get("daily_feature_set", "features-1"),
    )
    print(f"{len(dataset)} windows")
    if panel_size is None:  # the panels the pooler / mixer saw in pretraining
        panel_size = run_config.model.get("context_symbols") or run_config.data.train.get("max_symbols") or 0
    if getattr(model, "peer_pooler", None) is not None or getattr(model, "panel_mixer", None) is not None:
        print(f"pooling / mixing panels of {panel_size or 'all'} stocks per date")
    encode_kwargs = {"readout": readout} if readout else None
    arrays = compute_embeddings(model, dataset, batch_size, num_workers, encode_kwargs, panel_size, seed)
    name = (
        "embeddings" + (f"-{readout}" if readout else "") + time_suffix(decision_time)
        + ("-random-init" if random_init else "")
    )  # fmt: skip
    out = out or checkpoint / "probe" / f"{name}.npz"
    save_npz(out, arrays, decision_time)


@cli.command("window-features")
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--lookback", required=True, type=int, help="context bars, as the run's data.lookback")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--features", "bar_features", default="open,high,low,close,volume", show_default=True)
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--batch-size", default=4096, show_default=True)
@click.option("--num-workers", default=10, show_default=True)
@decision_time_option
def window_features(prefix, lookback, out: Path, bar_features, start, end, batch_size, num_workers, decision_time):
    """Model-free features (``hand``, ``hand_norm``, ``stats``) of every decision-time window, on the CPU."""
    dataset = StockSeriesDataset(
        data_prefix=prefix,
        features=bar_features.split(","),
        lookback=lookback,
        times_of_day=[decision_time],
        start_timestamp=start * 1_000_000,
        end_timestamp=end * 1_000_000,
    )
    print(f"{len(dataset)} windows")
    arrays = compute_embeddings(None, dataset, batch_size, num_workers)
    save_npz(out, arrays, decision_time)


@cli.command("alpha158")
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--num-workers", default=16, show_default=True)
@click.option("--history", type=int, default=None, help="trading days each row sees (4 = 960 bars); all if unset")
def alpha158_features(prefix, out: Path, start, end, num_workers, history):
    """Qlib Alpha158 of every (symbol, trading day) (``tools.baselines.alpha158``), robust-normalized on probe-train."""
    arrays = alpha158.compute(prefix, start, end, num_workers, history)
    fit = (arrays["date"] >= start) & (arrays["date"] < EVAL_START)
    arrays["alpha158"] = alpha158.normalize(arrays["alpha158"], fit).astype(np.float16)
    print(f"{len(arrays['date'])} rows x {arrays['alpha158'].shape[1]} features")
    save_npz(out, arrays)


@cli.command("kronos")
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--model", "name", default="Kronos-small", show_default=True, help="key of tools.baselines.kronos.MODELS")
@click.option("--lookback", default=512, show_default=True, help="context bars, at most the model's context")
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--batch-size", default=512, show_default=True)
@click.option("--num-workers", default=10, show_default=True)
@decision_time_option
def kronos_features(prefix, out: Path, name, lookback, start, end, batch_size, num_workers, decision_time):
    """Frozen Kronos states of every decision-time window (``tools.baselines.kronos``)."""
    from tools.baselines import kronos  # imports external/kronos and its deps only here

    arrays = kronos.compute(prefix, lookback, start, end, name, batch_size, num_workers, (decision_time,))
    print(f"{len(arrays['date'])} rows x {arrays['kronos'].shape[1]} dims")
    save_npz(out, arrays, decision_time)


@cli.command("timesfm")
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--lookback", default=960, show_default=True, help="context bars, a multiple of the 32-bar patch")
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--batch-size", default=512, show_default=True)
@click.option("--num-workers", default=10, show_default=True)
@decision_time_option
def timesfm_features(prefix, out: Path, lookback, start, end, batch_size, num_workers, decision_time):
    """Frozen TimesFM 2.5 states of every decision-time window (``tools.baselines.timesfm``)."""
    from tools.baselines import timesfm  # imports external/timesfm only here

    arrays = timesfm.compute(prefix, lookback, start, end, batch_size, num_workers, (decision_time,))
    print(f"{len(arrays['date'])} rows x {arrays['timesfm'].shape[1]} dims")
    save_npz(out, arrays, decision_time)


@cli.command("chronos2")
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--lookback", default=960, show_default=True, help="context bars, a multiple of the 16-bar patch")
@click.option("--channels", default="close,volume", show_default=True, help="comma-separated, one multivariate series")
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--batch-size", default=512, show_default=True)
@click.option("--num-workers", default=10, show_default=True)
@decision_time_option
def chronos2_features(prefix, out: Path, lookback, channels, start, end, batch_size, num_workers, decision_time):
    """Frozen Chronos-2 states of every decision-time window (``tools.baselines.chronos2``)."""
    from tools.baselines import chronos2  # imports external/chronos only here

    channels = tuple(channels.split(","))
    arrays = chronos2.compute(prefix, lookback, start, end, batch_size, num_workers, (decision_time,), channels)
    print(f"{len(arrays['date'])} rows x {arrays['chronos2'].shape[1]} dims")
    save_npz(out, arrays, decision_time)


@cli.command("moment")
@click.option("--prefix", required=True, help="v3 mmap prefix (without .bin/.idx)")
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--model", "name", default="MOMENT-1-base", show_default=True, help="key of tools.baselines.moment.MODELS"
)
@click.option("--lookback", default=512, show_default=True, help="context bars, a multiple of 8 up to 512")
@click.option("--channels", default="close,volume", show_default=True, help="comma-separated, each its own series")
@click.option("--start", default=20190101, show_default=True, help="first decision date (YYYYMMDD)")
@click.option("--end", default=TEST_END, show_default=True, help="decision dates before this (YYYYMMDD)")
@click.option("--batch-size", default=512, show_default=True)
@click.option("--num-workers", default=10, show_default=True)
@decision_time_option
def moment_features(prefix, out: Path, name, lookback, channels, start, end, batch_size, num_workers, decision_time):
    """Frozen MOMENT states of every decision-time window (``tools.baselines.moment``)."""
    from tools.baselines import moment  # imports external/moment only here

    channels = tuple(channels.split(","))
    arrays = moment.compute(prefix, lookback, start, end, batch_size, num_workers, (decision_time,), channels, name)
    print(f"{len(arrays['date'])} rows x {arrays['moment'].shape[1]} dims")
    save_npz(out, arrays, decision_time)


@cli.command()
@click.option("--embeddings", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--extra",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="npz of more features (e.g. from window-features) joined on (symbol, date); keys already in --embeddings win",
)
@click.option("--labels", "labels_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), help="default: next to --embeddings")
@click.option("--tasks", help="comma-separated; default: every task whose label column is in --labels")
@click.option(
    "--features",
    "feature_sets",
    help="comma-separated; default: each of FEATURE_SETS (+ panel sets) the embeddings have",
)
@click.option("--train-start", default=20190101, show_default=True, help="first probe-train decision date")
@click.option(
    "--model",
    type=click.Choice(["ridge", "gbdt"]),
    default="ridge",
    show_default=True,
    help="gbdt: LightGBM on the CPU (``run_gbdt_probe``), a nonlinear baseline for the model-free sets",
)
@click.option("--gbdt-leaves", default="15,63,255", show_default=True, help="gbdt: leaf counts to pick from on eval")
@click.option(
    "--all-preds", is_flag=True, help="save the test predictions of every task, not only PRED_TASKS (paired tests)"
)
def probe(
    embeddings: Path, extra, labels_path, out_dir, tasks, feature_sets, train_start, model, gbdt_leaves, all_preds
):
    """Probes per task and feature set: hyperparameters picked on eval RankIC (jump: AUC, profile: MSE), scored on
    eval and test."""
    emb = dict(np.load(embeddings, allow_pickle=False))
    decision_time = npz_decision_time(emb)
    emb = {k: v for k, v in emb.items() if v.ndim}  # drop scalar metadata such as panel_size
    order = np.argsort(emb["date"], kind="stable")
    emb = {k: v[order] for k, v in emb.items()}
    # keys of --extra that --features uses (all when it is not given); each is read and joined one at a time, so a
    # wide extra (e.g. 2 x 35 GB of TimesFM states) costs one copy of the key in use on top of what is kept
    used = None if feature_sets is None else {k for s in feature_sets.split(",") for k in s.split("+")}
    for path in extra:
        with np.load(path, allow_pickle=False) as more:
            # a daily feature counts as a day-end one: at an earlier decision time it would see the future
            more_time = npz_decision_time(more)
            assert more_time == decision_time, f"{path} is at {more_time}, {embeddings} at {decision_time}"
            index = pd.MultiIndex.from_arrays([more["symbol"], more["date"]])
            rows = index.get_indexer(pd.MultiIndex.from_arrays([emb["symbol"], emb["date"]]))
            assert (rows >= 0).all(), f"{(rows < 0).sum()} (symbol, date) rows of {embeddings} are missing in {path}"
            for k in more.files:
                if k not in emb and (used is None or k in used):
                    v = more[k]
                    if v.ndim:
                        emb[k] = v[rows]
                    del v
    if feature_sets is None:
        sets = [s for s in FEATURE_SETS if all(k in emb for k in s.split("+"))]
        panel = "+".join(["online"] + [k for k in ("peer", "market") if k in emb])
        if panel != "online":
            sets += [panel, f"{panel}+hand"]
        if "mixed" in emb:
            sets += ["mixed", "mixed+hand"]
        feature_sets = ",".join(sets)
    keys = pd.DataFrame({"symbol": emb["symbol"], "date": emb["date"]})
    label_df = keys.merge(pd.read_parquet(labels_path), on=["symbol", "date"], how="left")
    if tasks is None:
        timed = [task for task in TASKS if TASK_TIME.get(task, DECISION_TIME) == decision_time]
        tasks = ",".join(task for task in timed if TASKS[task][0] in label_df)
        missing = [task for task in timed if TASKS[task][0] not in label_df]
        if missing:
            print(f"skipping {missing}: no label column in {labels_path} (regenerate it with `labels`)")
    out_dir = out_dir or embeddings.parent / (
        embeddings.stem.replace("embeddings", "probe") + ("-gbdt" if model == "gbdt" else "")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results, rows = [], []
    combos = list(itertools.product(tasks.split(","), feature_sets.split(",")))
    for task, features in tqdm(combos, desc="probe"):
        task_time = TASK_TIME.get(task, DECISION_TIME)
        assert task_time == decision_time, f"{task} is decided at {task_time}, {embeddings} at {decision_time}"
        label = label_df[TASKS[task][0]].to_numpy()
        if model == "gbdt":
            leaves = tuple(int(n) for n in gbdt_leaves.split(","))
            result = run_gbdt_probe(emb, label, task, features, train_start, leaves)
            picked = f"leaves {result['num_leaves']}, trees {result['num_trees']}"
        else:
            result = run_probe(emb, label, task, features, train_start)
            picked = f"alpha {result['alpha']:.0e}"
        test_pred = result.pop("test_pred")
        if (all_preds or task in PRED_TASKS) and test_pred is not None:
            test_pred.to_parquet(out_dir / f"pred-{task}-{features}-test.parquet", index=False)
        results.append(result)
        for split, m in result["splits"].items():
            rows.append(
                {
                    "task": task,
                    "features": features,
                    "split": split,
                    "ic": m["ic"],
                    "icir": m["icir"],
                    "rank_ic": m["rank_ic"],
                    "rank_icir": m["rank_icir"],
                    "r2": m["r2"],
                    "auc": m["auc"],
                    "days": m["days"],
                }
            )
        tqdm.write(
            f"{task} {features}: {picked}, "
            + ", ".join(f"{s} ic {m['ic']:.4f} rank_ic {m['rank_ic']:.4f}" for s, m in result["splits"].items())
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    table = pd.DataFrame(rows).to_string(index=False, float_format="{:.4f}".format)
    (out_dir / "metrics.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    cli()
