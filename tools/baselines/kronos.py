"""Kronos frozen-model features of the 15:00 1-min windows of the v3 data.

Kronos (https://arxiv.org/abs/2508.02739) is a decoder-only model pretrained on K-line sequences: a tokenizer
(``KronosTokenizer``, binary spherical quantization) turns each bar into a coarse and a fine token, and a causal
transformer (``Kronos``) predicts the next bar's tokens. The official code is vendored unmodified in
``external/kronos`` (see its README); weights come from the Hugging Face Hub at pinned revisions (Kronos-small's
are the ones its regression test pins).

Each window is prepared exactly as ``KronosPredictor.predict_batch`` does it: bars ``[open, high, low, close, volume,
amount]``, where ``amount = volume * mean(open, high, low, close)`` (the predictor's own fill when amount is missing;
the 1-min data has none), z-scored by the window's own mean and population std (+1e-5) and clipped to +-5; time
features ``[minute, hour, weekday, day, month]`` of each bar's timestamp. ``KronosTokenizer.encode(half=True)`` gives
the (s1, s2) tokens and ``Kronos.decode_s1`` the final-norm hidden states [B, L, d_model], the context its s1 / s2
heads read (the first step of ``auto_regressive_inference``). Two readouts:

    kronos       state of the window's last bar (15:00), the one that predicts the next bar
    kronos_mean  mean state over the window

``lookback`` is capped at the model's context (2048 bars for Kronos-mini, 512 for -small and -base); a longer window
would be normalized over all its bars but only its last ``max_context`` tokens fed to the model. The model runs in
fp32, as the predictor does.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mc_jepa.datasets.v3.series_dataset import StockSeriesDataset

KRONOS_SRC = Path(__file__).resolve().parents[2] / "external" / "kronos"
if str(KRONOS_SRC) not in sys.path:
    sys.path.insert(0, str(KRONOS_SRC))
from model import Kronos, KronosTokenizer  # noqa: E402  (external/kronos/model)

# name -> (model repo, revision), (tokenizer repo, revision), max_context; the README's model table. Kronos-small and
# Tokenizer-base are at the revisions pinned by external/kronos/tests/test_kronos_regression.py, the others at the
# Hub's main as of 2026-10-09.
TOKENIZER_BASE = ("NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426")
MODELS = {
    "Kronos-mini": (
        ("NeoQuasar/Kronos-mini", "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"),
        ("NeoQuasar/Kronos-Tokenizer-2k", "26966d0035065a0cae0ebad7af8ece35bc1fb51c"),
        2048,
    ),
    "Kronos-small": (("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320"), TOKENIZER_BASE, 512),
    "Kronos-base": (("NeoQuasar/Kronos-base", "2b554741eca47781b64468546e77fef3e85130e6"), TOKENIZER_BASE, 512),
}
MAX_CONTEXT = 512  # Kronos-small / -base
CLIP = 5.0
BARS = ["open", "high", "low", "close", "volume"]


def load(name: str = "Kronos-small", device: str = "cuda") -> tuple[KronosTokenizer, Kronos]:
    """Pretrained tokenizer and model in eval mode."""
    (repo, revision), (tokenizer_repo, tokenizer_revision), _ = MODELS[name]
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo, revision=tokenizer_revision)
    model = Kronos.from_pretrained(repo, revision=revision)
    return tokenizer.to(device).eval(), model.to(device).eval()


def time_features(timestamp: np.ndarray) -> np.ndarray:
    """``kronos.calc_time_stamps`` of YYYYMMDDHHMMSS integers: [N, 5] minute, hour, weekday, day, month (float32)."""
    t = pd.to_datetime(pd.Series(timestamp).astype(str), format="%Y%m%d%H%M%S")
    return np.stack([t.dt.minute, t.dt.hour, t.dt.weekday, t.dt.day, t.dt.month], axis=1).astype(np.float32)


def prepare(bars: np.ndarray) -> np.ndarray:
    """``KronosPredictor.predict_batch`` input of one window's [L, 5] OHLCV bars: normalized [L, 6] float32."""
    amount = bars[:, 4] * bars[:, :4].mean(axis=1)
    # column-major like the predictor's ``DataFrame.values``: float32 column sums then match it bit for bit (prices
    # have a tiny std relative to their level, so summation order shows in the normalized window)
    x = np.asfortranarray(np.concatenate([bars, amount[:, None]], axis=1), dtype=np.float32)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    return np.clip((x - x_mean) / (x_std + 1e-5), -CLIP, CLIP)


class KronosWindows(Dataset):
    """``StockSeriesDataset`` windows (raw OHLCV, ending at each decision time) as Kronos inputs and time stamps."""

    def __init__(self, prefix: str, lookback: int, max_context: int = MAX_CONTEXT, **kwargs):
        if lookback > max_context:
            raise ValueError(f"lookback {lookback} exceeds the Kronos context of {max_context} bars")
        self.windows = StockSeriesDataset(prefix, BARS, lookback, normalize=False, **kwargs)
        self.data = self.windows.data
        self.stamps = time_features(self.data.timeline)  # [T, 5], indexed by time id

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        anchor = int(self.windows.anchor_rows[idx])
        a, b = anchor - self.windows.lookback + 1, anchor + 1
        time_ids = np.asarray(self.data.time_ids[a:b])
        return {
            "x": torch.from_numpy(np.ascontiguousarray(prepare(self.data.read(a, b, BARS, dtype=np.float64)))),
            "stamp": torch.from_numpy(self.stamps[time_ids]),
            "symbol_id": int(self.data.symbol_of_rows(np.array([anchor]))[0]),
            "timestamp": int(self.data.timeline[time_ids[-1]]),
        }


@torch.no_grad()
def hidden_states(tokenizer: KronosTokenizer, model: Kronos, x: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:
    """Final-norm states [B, L, d_model] of normalized windows x [B, L, 6] with time stamps [B, L, 5]."""
    s1, s2 = tokenizer.encode(x, half=True)
    return model.decode_s1(s1, s2, stamp)[1]


def compute(
    prefix: str,
    lookback: int = MAX_CONTEXT,
    start: int | None = None,
    end: int | None = None,
    name: str = "Kronos-small",
    batch_size: int = 512,
    num_workers: int = 10,
    times_of_day: tuple[int, ...] = (150000,),
) -> dict[str, np.ndarray]:
    """``kronos`` and ``kronos_mean`` (float16) of every window ending at ``times_of_day`` on dates [start, end)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load(name, device)
    dataset = KronosWindows(
        prefix,
        lookback,
        MODELS[name][2],
        times_of_day=list(times_of_day),
        start_timestamp=None if start is None else start * 1_000_000,
        end_timestamp=None if end is None else end * 1_000_000,
    )
    print(f"{len(dataset)} windows of {lookback} bars, {name}")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out = {k: [] for k in ("symbol_id", "date", "kronos", "kronos_mean")}
    for batch in tqdm(loader, desc="kronos"):
        h = hidden_states(tokenizer, model, batch["x"].to(device), batch["stamp"].to(device))
        out["kronos"].append(h[:, -1].cpu().numpy().astype(np.float16))
        out["kronos_mean"].append(h.mean(dim=1).cpu().numpy().astype(np.float16))
        out["symbol_id"].append(batch["symbol_id"].numpy())
        out["date"].append((batch["timestamp"] // 1_000_000).numpy())
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["symbol"] = np.asarray(dataset.data.symbols)[arrays.pop("symbol_id")]
    arrays["date"] = arrays["date"].astype(np.int32)
    return arrays
