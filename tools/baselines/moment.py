"""MOMENT frozen-model features of the 15:00 1-min windows of the v3 data.

MOMENT (https://arxiv.org/abs/2402.03885) is a T5 encoder (MOMENT-1-base: 12 layers, 768 dims, 85M transformer
parameters; MOMENT-1-small: 8 layers, 512 dims) pretrained by masked reconstruction on the Time-series Pile: each
univariate series of exactly 512 points is instance-normalized (RevIN: the window's mean and population std over its
observed points), cut into 64 patches of 8 points, each embedded by a linear map plus a sinusoidal position (a learned
mask embedding replaces unobserved patches), and the encoder reads the patches with T5's relative attention. Channels
are independent. The official code is vendored unmodified in ``external/moment`` (see its README); the weights come
from the Hugging Face Hub at a pinned revision.

Each window is prepared exactly as ``MOMENT.embed`` prepares a [B, n_channels, 512] input: raw values cast to float32,
a shorter lookback left-padded with zeros under ``input_mask`` = 0 (the official way to feed short series), and
``embed(reduction="none")`` gives the final-norm encoder states [B, C, 64, d]. Two readouts, each [close, volume]:

    moment       mean state over the window's observed patches (``embed``'s readout, per channel instead of averaged)
    moment_last  state of the last patch (bars 14:53-15:00)

The lookback must be a multiple of the 8-bar patch and at most the 512-bar context. The model runs in fp32.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mc_jepa.datasets.v3.series_dataset import StockSeriesDataset

MOMENT_SRC = Path(__file__).resolve().parents[2] / "external" / "moment"
if str(MOMENT_SRC) not in sys.path:
    sys.path.insert(0, str(MOMENT_SRC))
from momentfm import MOMENTPipeline  # noqa: E402  (external/moment/momentfm)

# name -> (Hub repo, revision at the Hub's main as of 2026-10-12)
MODELS = {
    "MOMENT-1-small": ("AutonLab/MOMENT-1-small", "411e288267f82cce86296dbe4d6c8bc533cc162f"),
    "MOMENT-1-base": ("AutonLab/MOMENT-1-base", "5e44b0ea26376a176360f87831124e018f876d96"),
    "MOMENT-1-large": ("AutonLab/MOMENT-1-large", "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"),
}
PATCH, CONTEXT = 8, 512
CHANNELS = ("close", "volume")


def load(name: str = "MOMENT-1-base", device: str = "cuda") -> MOMENTPipeline:
    """Pretrained MOMENT in eval mode with the embedding head (the README's ``task_name="embedding"``)."""
    repo, revision = MODELS[name]
    model = MOMENTPipeline.from_pretrained(repo, revision=revision, model_kwargs={"task_name": "embedding"})
    model.init()
    return model.to(device).eval()


class MomentWindows(Dataset):
    """``StockSeriesDataset`` windows (raw channels, ending at each decision time) as left-padded MOMENT inputs."""

    def __init__(self, prefix: str, lookback: int, channels: tuple[str, ...] = CHANNELS, **kwargs):
        if lookback % PATCH or lookback > CONTEXT:
            raise ValueError(f"lookback {lookback} is not a multiple of the {PATCH}-bar patch up to {CONTEXT}")
        self.channels = list(channels)
        self.windows = StockSeriesDataset(prefix, self.channels, lookback, normalize=False, **kwargs)
        self.data = self.windows.data
        self.input_mask = torch.zeros(CONTEXT)
        self.input_mask[CONTEXT - lookback :] = 1

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        anchor = int(self.windows.anchor_rows[idx])
        lookback = self.windows.lookback
        x = np.zeros((len(self.channels), CONTEXT), dtype=np.float32)
        x[:, CONTEXT - lookback :] = self.data.read(
            anchor - lookback + 1, anchor + 1, self.channels, dtype=np.float64
        ).T
        return {
            "x": torch.from_numpy(x),
            "input_mask": self.input_mask,
            "symbol_id": int(self.data.symbol_of_rows(np.array([anchor]))[0]),
            "timestamp": int(self.data.timeline[self.data.time_ids[anchor]]),
        }


@torch.no_grad()
def hidden_states(model: MOMENTPipeline, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    """Encoder states [B, C, 64, d] of raw windows x [B, C, 512] observed where input_mask [B, 512] is 1."""
    return model.embed(x_enc=x.to(torch.float32), input_mask=input_mask, reduction="none").embeddings


def readouts(h: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean over observed patches, last patch) of states h [B, C, P, d], each [B, C * d]."""
    observed = (input_mask.unfold(-1, PATCH, PATCH).sum(dim=-1) == PATCH).to(h.dtype)  # [B, P], as ``embed`` masks
    mean = (h * observed[:, None, :, None]).sum(dim=2) / observed.sum(dim=1)[:, None, None]
    return mean.flatten(1), h[:, :, -1].flatten(1)


def compute(
    prefix: str,
    lookback: int = CONTEXT,
    start: int | None = None,
    end: int | None = None,
    batch_size: int = 512,
    num_workers: int = 10,
    times_of_day: tuple[int, ...] = (150000,),
    channels: tuple[str, ...] = CHANNELS,
    name: str = "MOMENT-1-base",
) -> dict[str, np.ndarray]:
    """``moment`` and ``moment_last`` (float16) of every window ending at ``times_of_day`` on dates [start, end)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(name, device)
    dataset = MomentWindows(
        prefix,
        lookback,
        channels,
        times_of_day=list(times_of_day),
        start_timestamp=None if start is None else start * 1_000_000,
        end_timestamp=None if end is None else end * 1_000_000,
    )
    print(f"{len(dataset)} windows of {lookback} bars, {name} on {list(channels)}")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    out = {k: [] for k in ("symbol_id", "date", "moment", "moment_last")}
    for batch in tqdm(loader, desc="moment"):
        input_mask = batch["input_mask"].to(device)
        mean, last = readouts(hidden_states(model, batch["x"].to(device), input_mask), input_mask)
        out["moment"].append(mean.cpu().numpy().astype(np.float16))
        out["moment_last"].append(last.cpu().numpy().astype(np.float16))
        out["symbol_id"].append(batch["symbol_id"].numpy())
        out["date"].append((batch["timestamp"] // 1_000_000).numpy())
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["symbol"] = np.asarray(dataset.data.symbols)[arrays.pop("symbol_id")]
    arrays["date"] = arrays["date"].astype(np.int32)
    return arrays
