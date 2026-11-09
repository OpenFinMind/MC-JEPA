from typing import NotRequired, TypedDict

import torch


class PanelSample(TypedDict):
    """One cross-sectional sample from ``CrossSectionPanelDataset.__getitem__`` (N stocks, F features)."""

    symbol_ids: torch.Tensor  # [N] int
    timestamp: torch.Tensor  # [] long, decision time YYYYMMDDHHMMSS
    context: torch.Tensor  # [N, lookback, F]
    future: NotRequired[torch.Tensor]  # [N, lookahead, F], only if lookahead > 0
    mean: NotRequired[torch.Tensor]  # [N, F], context mean, only if normalize
    std: NotRequired[torch.Tensor]  # [N, F], context std, only if normalize
    daily: NotRequired[torch.Tensor]  # [N, daily_lookback, DAILY_FEATURES], only with daily_lookback


class SeriesSample(TypedDict):
    """One single-stock sample from ``StockSeriesDataset.__getitem__`` (F features)."""

    symbol_id: torch.Tensor  # [] long
    timestamp: torch.Tensor  # [] long, decision time YYYYMMDDHHMMSS
    context: torch.Tensor  # [lookback, F]
    future: NotRequired[torch.Tensor]  # [lookahead, F], only if lookahead > 0
    mean: NotRequired[torch.Tensor]  # [F], context mean, only if normalize
    std: NotRequired[torch.Tensor]  # [F], context std, only if normalize
    daily: NotRequired[torch.Tensor]  # [daily_lookback, DAILY_FEATURES], only with daily_lookback
