"""볼린저밴드 전략 — 하단 이탈에 매수, 중심선/상단 복귀에 매도."""
from __future__ import annotations

import pandas as pd

from ..indicators import bollinger
from .base import Strategy, _hold_between


class BollingerStrategy(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0, exit_at: str = "mid"):
        if exit_at not in ("mid", "upper"):
            raise ValueError("exit_at은 'mid' 또는 'upper'.")
        self.window = window
        self.num_std = num_std
        self.exit_at = exit_at
        self.name = f"볼린저({window}, {num_std:g}σ, 청산={exit_at})"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        lower, mid, upper = bollinger(close, self.window, self.num_std)
        exit_line = mid if self.exit_at == "mid" else upper
        return _hold_between(close < lower, close > exit_line)
