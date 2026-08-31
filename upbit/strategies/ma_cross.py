"""이동평균선 교차 — 초보 1순위 전략.

단기선이 장기선을 위로 뚫으면(골든크로스) 매수, 아래로 뚫으면(데드크로스) 매도.
추세추종형이라 방향성 있는 장에서 강하고, 횡보장에서 잦은 손실(휩쏘)이 난다.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import crossover, crossunder, ema, sma
from .base import Strategy, _hold_between


class MACrossStrategy(Strategy):
    def __init__(self, fast: int = 5, slow: int = 20, use_ema: bool = False):
        if fast >= slow:
            raise ValueError(f"fast({fast})는 slow({slow})보다 작아야 한다.")
        self.fast = fast
        self.slow = slow
        self.use_ema = use_ema
        kind = "EMA" if use_ema else "SMA"
        self.name = f"{kind}교차({fast}/{slow})"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        avg = ema if self.use_ema else sma
        fast_line = avg(df["close"], self.fast)
        slow_line = avg(df["close"], self.slow)
        return _hold_between(
            crossover(fast_line, slow_line), crossunder(fast_line, slow_line)
        )
