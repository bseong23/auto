"""손절·익절 규칙 — 전략 신호와 별개로 "언제 강제로 나갈지"를 정한다.

왜 전략과 분리했나:
전략의 `generate_positions`는 데이터프레임만 보고 계산하는 **순수 함수**다.
그런데 손절은 "내가 얼마에 샀는지"에 따라 달라진다(경로 의존).
같은 차트라도 진입가가 다르면 손절선이 다르다. 그래서 전략이 아니라
백테스터 실행 루프에서 처리한다.

ATR 손절을 쓰는 이유:
"무조건 -5%"는 변동성이 큰 장에서는 너무 자주 털리고, 잔잔한 장에서는
너무 헐렁하다. ATR(평균 변동폭)의 배수로 잡으면 시장 상태에 맞춰 자동 조절된다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr


@dataclass
class RiskRules:
    """손절/익절 설정. 아무것도 안 켜면 규칙 없음(전략 신호대로만 매매)."""

    #: 고정 비율 손절. 0.05 → 진입가 대비 -5%
    stop_loss_pct: float | None = None
    #: ATR 배수 손절. 2.0 → 진입가 - 2×ATR
    atr_multiple: float | None = None
    atr_window: int = 14
    #: 고정 비율 익절. 0.10 → +10%에서 청산
    take_profit_pct: float | None = None
    #: 고점 추적 손절 — 가격이 오르면 손절선도 따라 올린다(내려가진 않음)
    trailing: bool = False

    def __post_init__(self) -> None:
        for name in ("stop_loss_pct", "take_profit_pct"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                raise ValueError(f"{name}는 0과 1 사이의 비율이어야 한다. 받은 값: {value}")
        if self.atr_multiple is not None and self.atr_multiple <= 0:
            raise ValueError(f"atr_multiple은 양수여야 한다. 받은 값: {self.atr_multiple}")
        if self.trailing and not self.has_stop:
            raise ValueError("trailing을 켜려면 stop_loss_pct나 atr_multiple 중 하나는 있어야 한다.")

    @property
    def has_stop(self) -> bool:
        return self.stop_loss_pct is not None or self.atr_multiple is not None

    @property
    def is_active(self) -> bool:
        return self.has_stop or self.take_profit_pct is not None

    @property
    def needs_atr(self) -> bool:
        return self.atr_multiple is not None

    def stop_price(self, reference: float, atr_value: float | None) -> float | None:
        """기준가(진입가 또는 고점) 대비 손절선. 규칙이 여럿이면 더 가까운 쪽을 쓴다."""
        candidates = []
        if self.stop_loss_pct is not None:
            candidates.append(reference * (1 - self.stop_loss_pct))
        if self.atr_multiple is not None and atr_value is not None and not np.isnan(atr_value):
            candidates.append(reference - self.atr_multiple * atr_value)
        return max(candidates) if candidates else None

    def target_price(self, entry_price: float) -> float | None:
        if self.take_profit_pct is None:
            return None
        return entry_price * (1 + self.take_profit_pct)

    def describe(self) -> str:
        if not self.is_active:
            return "규칙 없음"
        parts = []
        if self.atr_multiple is not None:
            parts.append(f"ATR{self.atr_window}×{self.atr_multiple:g} 손절")
        if self.stop_loss_pct is not None:
            parts.append(f"-{self.stop_loss_pct:.1%} 손절")
        if self.take_profit_pct is not None:
            parts.append(f"+{self.take_profit_pct:.1%} 익절")
        if self.trailing:
            parts.append("추적")
        return " / ".join(parts)


def atr_series(df: pd.DataFrame, rules: RiskRules) -> pd.Series | None:
    """ATR이 필요한 규칙일 때만 계산한다."""
    return atr(df, rules.atr_window) if rules.needs_atr else None


@dataclass
class OpenPosition:
    """보유 중인 포지션의 손절/익절 상태. 봉마다 갱신된다."""

    entry_price: float
    stop: float | None
    target: float | None
    high_water: float

    def check_exit(
        self, bar_open: float, bar_low: float, bar_high: float
    ) -> tuple[float, str] | None:
        """이번 봉에서 손절/익절이 걸렸는지. 걸렸으면 (체결가, 사유).

        체결가 처리에서 정직해야 할 부분:
        - **갭 하락**: 시가가 이미 손절선 아래면 손절가가 아니라 **시가**에 체결된다.
          현실에서 스탑은 손절선을 보장하지 않는다. 이걸 손절가로 계산하면 손실이 축소된다.
        - **같은 봉에 손절·익절이 모두 닿은 경우**: 봉 안에서 어느 쪽이 먼저였는지
          알 수 없으므로 **손절이 먼저 터진 것으로 간주**한다(보수적).
        """
        if self.stop is not None:
            if bar_open <= self.stop:
                return bar_open, "손절(갭)"
            if bar_low <= self.stop:
                return self.stop, "손절"

        if self.target is not None:
            if bar_open >= self.target:
                return bar_open, "익절(갭)"
            if bar_high >= self.target:
                return self.target, "익절"

        return None

    def update_trailing(self, bar_high: float, atr_value: float | None, rules: RiskRules) -> None:
        """이번 봉의 고점을 반영해 손절선을 올린다. 절대 내리지 않는다.

        주의: 이 갱신은 손절 판정을 **끝낸 뒤에** 호출해야 한다.
        이번 봉 고점으로 손절선을 올려놓고 같은 봉에서 판정하면 미래참조가 된다.
        """
        if not rules.trailing or self.stop is None:
            return
        self.high_water = max(self.high_water, bar_high)
        raised = rules.stop_price(self.high_water, atr_value)
        if raised is not None:
            self.stop = max(self.stop, raised)
