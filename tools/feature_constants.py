"""Feature constants for the 1-min OHLCV bars.

``FEATURES`` gives each feature's group method (1-min -> coarser bars) and mmap storage dtype.
``FEATURE_SUBSETS`` names the selectable feature sets: only "ohlcv".
"""

import enum
from typing import NamedTuple

import polars as pl


class GroupMethod(enum.Enum):
    """Aggregation methods for grouping time series data."""

    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    AVG = "avg"

    @staticmethod
    def get_group_fn_list(features: list[tuple[str, "GroupMethod"]]) -> list[pl.Expr]:
        group_fn_list = []
        for feature_name, method in features:
            if method == GroupMethod.FIRST:
                group_fn_list.append(pl.col(feature_name).first().alias(feature_name))
            elif method == GroupMethod.LAST:
                group_fn_list.append(pl.col(feature_name).last().alias(feature_name))
            elif method == GroupMethod.SUM:
                group_fn_list.append(pl.col(feature_name).sum().alias(feature_name))
            elif method == GroupMethod.MIN:
                group_fn_list.append(pl.col(feature_name).min().alias(feature_name))
            elif method == GroupMethod.MAX:
                group_fn_list.append(pl.col(feature_name).max().alias(feature_name))
            elif method == GroupMethod.MEDIAN:
                group_fn_list.append(pl.col(feature_name).median().alias(feature_name))
            elif method == GroupMethod.AVG:
                group_fn_list.append(pl.col(feature_name).mean().alias(feature_name))
            else:
                raise ValueError(f"Unknown group method: {method}")
        return group_fn_list


class Feature(NamedTuple):
    name: str
    group_method: GroupMethod  # how 1-min bars aggregate into coarser bars
    dtype: str  # numpy storage dtype in the v3 mmap dataset (the by-symbol parquet holds float64/int64)


FEATURES: list[Feature] = [
    Feature("open", GroupMethod.FIRST, "float32"),
    Feature("close", GroupMethod.LAST, "float32"),
    Feature("high", GroupMethod.MAX, "float32"),
    Feature("low", GroupMethod.MIN, "float32"),
    Feature("volume", GroupMethod.SUM, "int32"),
]

# Named feature sets, selectable by name (e.g. process_mmap_v3_dataset.py --feature-set).
FEATURE_SUBSETS: dict[str, list[str]] = {
    "ohlcv": ["open", "high", "low", "close", "volume"],
}
