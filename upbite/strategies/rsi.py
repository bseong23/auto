"""RSI 역추세 전략 — 과매도에 사서 과매수에 판다.

추세추종의 반대 성격이라 횡보장에서 잘 먹고, 강한 추세장에서 얻어맞는다
(계속 오르는데 70 넘었다고 팔아버림).
"""
from __future__ import annotations

import pandas as pd

from ..indicators import rsi
from .base import Strategy, _hold_between


class RSIStrategy(Strategy):
    def __init__(self, window: int = 14, buy_below: float = 30, sell_above: float = 70):
        if not 0 < buy_below < sell_above < 100:
            raise ValueError("0 < buy_below < sell_above < 100 이어야 한다.")
        self.window = window
        self.buy_below = buy_below
        self.sell_above = sell_above
        self.name = f"RSI({window}, {buy_below:g}/{sell_above:g})"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        value = rsi(df["close"], self.window)
        return _hold_between(value < self.buy_below, value > self.sell_above)
