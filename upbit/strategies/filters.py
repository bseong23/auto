"""필터 — 기존 전략 위에 조건을 덧씌워 **안 사는 구간**을 만든다.

MA교차의 알려진 약점은 **횡보장 휩쏘**다. 방향이 없는 구간에서 교차가 계속
발생해 사고팔기를 반복하고, 매번 수수료와 슬리피지를 낸다.

필터는 전략을 바꾸지 않는다. "이 조건이 아니면 아예 참여하지 않는다"를 더한다.
안 하는 것도 전략이다 — 현금으로 있으면 최소한 잃지는 않는다.

전략을 감싸는 형태라 어떤 전략에도 붙는다:

    TrendFilter(MACrossStrategy(5, 20), window=200)
    VolatilityFilter(RSIStrategy(), min_pct=0.01, max_pct=0.06)
    TrendFilter(VolatilityFilter(MACrossStrategy(5, 20)))   # 겹쳐도 된다
"""
from __future__ import annotations

import pandas as pd

from ..indicators import atr, sma
from .base import Strategy


class FilteredStrategy(Strategy):
    """필터의 부모. 하위 전략 신호에 `allow` 마스크를 AND 로 곱한다.

    **매수만 막고 매도는 막지 않는다.** 필터가 꺼졌다고 들고 있던 걸 계속
    들고 있으면 안 되니까 — 나가는 문은 항상 열어둔다.
    """

    def __init__(self, inner: Strategy):
        self.inner = inner

    def allow(self, df: pd.DataFrame) -> pd.Series:
        """보유를 허용할 구간 (True/False). 하위 클래스가 구현한다."""
        raise NotImplementedError

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        base = self.inner.generate_positions(df)
        return (base.astype(bool) & self.allow(df).fillna(False)).astype(int)

    def params(self) -> dict:
        own = {k: v for k, v in vars(self).items() if k not in ("inner", "name")}
        return {**own, "inner": repr(self.inner)}


class TrendFilter(FilteredStrategy):
    """장기 추세가 상승일 때만 참여한다.

    "종가가 200일선 위" 라는 고전적인 조건. 하락 추세에서 반등을 노리다
    계속 얻어맞는 걸 막는 게 목적이다.

    대가: 바닥에서 반등할 때 늦게 들어간다. 200일선을 회복할 때까지 못 산다.
    """

    def __init__(self, inner: Strategy, window: int = 200):
        super().__init__(inner)
        self.window = window
        self.name = f"{inner.name} + 추세필터({window})"

    def allow(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > sma(df["close"], self.window)


class VolatilityFilter(FilteredStrategy):
    """변동성이 적당한 구간에서만 참여한다.

    ATR을 가격으로 나눈 값(변동성 비율)으로 판단한다.
    - 너무 낮으면: 방향이 없는 횡보장 → 교차가 자주 발생해 수수료만 나간다
    - 너무 높으면: 패닉 구간 → 손절이 계속 털린다

    범위는 종목·봉 종류마다 다르므로 **반드시 검증 구간에서 확인해야 한다.**
    이 숫자를 훈련 구간에 맞춰 고르면 그게 과최적화다.
    """

    def __init__(
        self,
        inner: Strategy,
        window: int = 14,
        min_pct: float | None = 0.01,
        max_pct: float | None = None,
    ):
        super().__init__(inner)
        if min_pct is not None and max_pct is not None and min_pct >= max_pct:
            raise ValueError(f"min_pct({min_pct})는 max_pct({max_pct})보다 작아야 한다.")
        if min_pct is None and max_pct is None:
            raise ValueError("min_pct 나 max_pct 중 하나는 있어야 필터 역할을 한다.")
        self.window = window
        self.min_pct = min_pct
        self.max_pct = max_pct
        bounds = f"{'' if min_pct is None else f'{min_pct:.1%}'}~{'' if max_pct is None else f'{max_pct:.1%}'}"
        self.name = f"{inner.name} + 변동성필터({bounds})"

    def allow(self, df: pd.DataFrame) -> pd.Series:
        ratio = atr(df, self.window) / df["close"]
        allowed = pd.Series(True, index=df.index)
        if self.min_pct is not None:
            allowed &= ratio >= self.min_pct
        if self.max_pct is not None:
            allowed &= ratio <= self.max_pct
        return allowed & ratio.notna()
