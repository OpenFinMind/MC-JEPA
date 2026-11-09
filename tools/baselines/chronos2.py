"""Chronos-2 frozen-model features of the 15:00 1-min windows of the v3 data.

Chronos-2 (https://arxiv.org/abs/2510.15821) is an encoder-only forecaster (120M parameters, 12 layers, 768 dims)
pretrained on real and synthetic series: each patch of 16 points, with its time index and observed mask, is embedded by
a residual MLP, a [REG] token and masked future-patch tokens are appended, and each layer alternates attention over a
series' own patches with group attention across the series of one task (the variates of a multivariate series) at the
same patch. A quantile head on each future-patch token's state forecasts its 16 points. The official code is vendored
unmodified in ``external/chronos`` (see its README); the weights come from the Hugging Face Hub at a pinned revision.

Each window is prepared exactly as ``Chronos2Pipeline.embed`` prepares a [n_variates, L] input: raw values cast to
float32, the channels of one window forming one group (so ``close`` and ``volume`` attend to each other, unlike in
TimesFM), and ``Chronos2Model.encode`` with one output patch; it scales each series by its window's mean and population
std, then arcsinh, and returns the final-norm states [n_variates, L / 16 + 2, 768] of the context patches, the [REG]
token and the future patch. Two readouts, each [close, volume]:

    chronos2       state of the masked future-patch token, the one the quantile head reads to forecast the next 16 bars
    chronos2_mean  mean state over the window's context patches

The lookback should be a multiple of the 16-bar patch (a shorter first patch would be left-padded as missing); the
context limit (8192) is far beyond the data's windows. The model runs in fp32, as the pipeline does.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mc_jepa.datasets.v3.series_dataset import StockSeriesDataset

CHRONOS_SRC = Path(__file__).resolve().parents[2] / "external" / "chronos"
if str(CHRONOS_SRC) not in sys.path:
    sys.path.insert(0, str(CHRONOS_SRC))
from chronos import Chronos2Model  # noqa: E402  (external/chronos/chronos)

# amazon/chronos-2 at the Hub's main as of 2026-10-10
REPO, REVISION = "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"
PATCH = 16
CHANNELS = ("close", "volume")


def load(device: str = "cuda") -> Chronos2Model:
    """Pretrained Chronos-2 (120M) in eval mode (``Chronos2Pipeline.from_pretrained``'s model)."""
    model = Chronos2Model.from_pretrained(REPO, revision=REVISION)
    return model.to(device).eval()


class Chronos2Windows(Dataset):
    """``StockSeriesDataset`` windows (raw channels, ending at each decision time) as Chronos-2 multivariate series."""

    def __init__(self, prefix: str, lookback: int, channels: tuple[str, ...] = CHANNELS, **kwargs):
        if lookback % PATCH:
            raise ValueError(f"lookback {lookback} is not a multiple of the {PATCH}-bar patch")
        self.channels = list(channels)
        self.windows = StockSeriesDataset(prefix, self.channels, lookback, normalize=False, **kwargs)
        self.data = self.windows.data

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        anchor = int(self.windows.anchor_rows[idx])
        a, b = anchor - self.windows.lookback + 1, anchor + 1
        x = self.data.read(a, b, self.channels, dtype=np.float64).T  # [C, L]
        return {
            # the pipeline casts its inputs to float32 on the CPU, before the model's float32 scaling
            "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "symbol_id": int(self.data.symbol_of_rows(np.array([anchor]))[0]),
            "timestamp": int(self.data.timeline[self.data.time_ids[anchor]]),
        }


@torch.no_grad()
def hidden_states(model: Chronos2Model, x: torch.Tensor) -> torch.Tensor:
    """Final-norm states [B, C, L / 16 + 2, 768] of raw windows x [B, C, L], each window's C channels one group."""
    batch, channels = x.shape[:2]
    group_ids = torch.arange(batch, device=x.device).repeat_interleave(channels)
    encoder_outputs, *_ = model.encode(context=x.flatten(0, 1).to(torch.float32), group_ids=group_ids)
    return encoder_outputs[0].unflatten(0, (batch, channels))


def compute(
    prefix: str,
    lookback: int = 960,
    start: int | None = None,
    end: int | None = None,
    batch_size: int = 512,
    num_workers: int = 10,
    times_of_day: tuple[int, ...] = (150000,),
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, np.ndarray]:
    """``chronos2`` and ``chronos2_mean`` (float16) of every window ending at ``times_of_day`` on dates [start, end)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(device)
    dataset = Chronos2Windows(
        prefix,
        lookback,
        channels,
        times_of_day=list(times_of_day),
        start_timestamp=None if start is None else start * 1_000_000,
        end_timestamp=None if end is None else end * 1_000_000,
    )
    print(f"{len(dataset)} windows of {lookback} bars, Chronos-2 on {list(channels)}")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out = {k: [] for k in ("symbol_id", "date", "chronos2", "chronos2_mean")}
    for batch in tqdm(loader, desc="chronos2"):
        h = hidden_states(model, batch["x"].to(device))  # [B, C, L / 16 + 2, d]: context patches, [REG], future
        out["chronos2"].append(h[:, :, -1].flatten(1).cpu().numpy().astype(np.float16))
        out["chronos2_mean"].append(h[:, :, :-2].mean(dim=2).flatten(1).cpu().numpy().astype(np.float16))
        out["symbol_id"].append(batch["symbol_id"].numpy())
        out["date"].append((batch["timestamp"] // 1_000_000).numpy())
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["symbol"] = np.asarray(dataset.data.symbols)[arrays.pop("symbol_id")]
    arrays["date"] = arrays["date"].astype(np.int32)
    return arrays
