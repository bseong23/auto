"""사서 존버 — 모든 전략이 이겨야 할 기준선(벤치마크).

전략이 이것보다 못하면, 그 전략은 수수료만 내면서 고생한 것이다.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy


class BuyAndHoldStrategy(Strategy):
    name = "사서 존버"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index, dtype=int)
