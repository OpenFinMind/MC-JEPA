"""TimesFM 2.5 frozen-model features of the 15:00 1-min windows of the v3 data.

TimesFM (https://arxiv.org/abs/2310.10688) is a decoder-only forecaster pretrained on univariate series: each patch of
32 points is embedded by a residual MLP, a causal transformer (20 layers, 1280 dims in 2.5) reads the patches, and
point / quantile heads on each patch's state forecast the next 128 points. The official code is vendored unmodified in
``external/timesfm`` (see its README); the weights come from the Hugging Face Hub at a pinned revision.

The model is univariate, so ``close`` and ``volume`` are embedded as two series and their states concatenated. Each
series is prepared exactly as ``TimesFM_2p5_200M_torch.forecast`` prepares it under the model card's
``ForecastConfig(normalize_inputs=True)`` with ``max_context`` = the lookback (so no padding): raw values, z-scored by
the window's mean and unbiased std in float32, then ``decode`` patches them, normalizes each patch by the running
stats of the patches so far and runs the prefill. ``hidden_states`` hooks the decoder's output embeddings
[B, L / 32, 1280] of that prefill, the states the point and quantile heads read. Two readouts, each [close, volume]:

    timesfm       state of the last patch (bars 14:31-15:00 of a full day), the one that forecasts the next 128 bars
    timesfm_mean  mean state over the window's patches

``force_flip_invariance`` would also decode the negated series; only the states of the series itself are kept. The
lookback must be a multiple of the 32-bar patch; the context limit (16384 - 128) is far beyond the data's windows. The
model runs in fp32, as the forecaster does.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mc_jepa.datasets.v3.series_dataset import StockSeriesDataset

TIMESFM_SRC = Path(__file__).resolve().parents[2] / "external" / "timesfm"
if str(TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(TIMESFM_SRC))
from timesfm import configs  # noqa: E402  (external/timesfm/timesfm)
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch, TimesFM_2p5_200M_torch_module  # noqa: E402
from timesfm.torch.util import revin  # noqa: E402

# google/timesfm-2.5-200m-pytorch at the Hub's main as of 2026-10-09
REPO, REVISION = "google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
PATCH = 32
CHANNELS = ["close", "volume"]


def forecast_config(lookback: int, batch_size: int = 1) -> configs.ForecastConfig:
    """The model card's config, with ``max_context`` = lookback and the shortest ``max_horizon`` (one output patch)."""
    return configs.ForecastConfig(
        max_context=lookback,
        max_horizon=128,
        normalize_inputs=True,
        per_core_batch_size=batch_size,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )


def load(device: str = "cuda") -> TimesFM_2p5_200M_torch_module:
    """Pretrained TimesFM 2.5 (200M) transformer in eval mode, not ``torch.compile``d."""
    tfm = TimesFM_2p5_200M_torch.from_pretrained(REPO, revision=REVISION, torch_compile=False)
    return tfm.model.to(device).eval()


class TimesFMWindows(Dataset):
    """``StockSeriesDataset`` windows (raw close and volume, ending at each decision time) as TimesFM series."""

    def __init__(self, prefix: str, lookback: int, **kwargs):
        if lookback % PATCH:
            raise ValueError(f"lookback {lookback} is not a multiple of the {PATCH}-bar patch")
        self.windows = StockSeriesDataset(prefix, CHANNELS, lookback, normalize=False, **kwargs)
        self.data = self.windows.data

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        anchor = int(self.windows.anchor_rows[idx])
        a, b = anchor - self.windows.lookback + 1, anchor + 1
        return {
            # float64 like the forecaster's numpy inputs; cast to float32 on the device, as it does
            "x": torch.from_numpy(np.ascontiguousarray(self.data.read(a, b, CHANNELS, dtype=np.float64).T)),
            "symbol_id": int(self.data.symbol_of_rows(np.array([anchor]))[0]),
            "timestamp": int(self.data.timeline[self.data.time_ids[anchor]]),
        }


@torch.no_grad()
def hidden_states(model: TimesFM_2p5_200M_torch_module, x: torch.Tensor) -> torch.Tensor:
    """Output states [B, L / 32, 1280] of the prefill of raw series x [B, L] (``compiled_decode``'s first decode)."""
    x = x.to(torch.float32)
    x = revin(x, torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True), reverse=False)
    states = []
    hook = model.register_forward_hook(lambda module, args, out: states.append(out[0][1]))
    try:
        model.decode(model.o, x, torch.zeros_like(x, dtype=torch.bool))  # one output patch: the prefill alone
    finally:
        hook.remove()
    return states[0]


def compute(
    prefix: str,
    lookback: int = 960,
    start: int | None = None,
    end: int | None = None,
    batch_size: int = 512,
    num_workers: int = 10,
    times_of_day: tuple[int, ...] = (150000,),
) -> dict[str, np.ndarray]:
    """``timesfm`` and ``timesfm_mean`` (float16) of every window ending at ``times_of_day`` on dates [start, end)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(device)
    dataset = TimesFMWindows(
        prefix,
        lookback,
        times_of_day=list(times_of_day),
        start_timestamp=None if start is None else start * 1_000_000,
        end_timestamp=None if end is None else end * 1_000_000,
    )
    print(f"{len(dataset)} windows of {lookback} bars, TimesFM-2.5 on {CHANNELS}")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out = {k: [] for k in ("symbol_id", "date", "timesfm", "timesfm_mean")}
    for batch in tqdm(loader, desc="timesfm"):
        x = batch["x"].to(device)  # [B, C, L]
        h = hidden_states(model, x.flatten(0, 1)).unflatten(0, x.shape[:2])  # [B, C, L / 32, d]
        out["timesfm"].append(h[:, :, -1].flatten(1).cpu().numpy().astype(np.float16))
        out["timesfm_mean"].append(h.mean(dim=2).flatten(1).cpu().numpy().astype(np.float16))
        out["symbol_id"].append(batch["symbol_id"].numpy())
        out["date"].append((batch["timestamp"] // 1_000_000).numpy())
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["symbol"] = np.asarray(dataset.data.symbols)[arrays.pop("symbol_id")]
    arrays["date"] = arrays["date"].astype(np.int32)
    return arrays
