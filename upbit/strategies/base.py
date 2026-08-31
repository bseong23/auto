"""전략 인터페이스.

핵심 규약 하나만 지키면 어떤 전략이든 백테스터에 꽂을 수 있다:

    generate_positions(df) -> pd.Series  # 각 봉마다 원하는 포지션 (1=보유, 0=현금)

그리고 **그 봉의 종가까지의 정보만** 써야 한다. 미래 값을 쓰면(lookahead)
백테스트 결과가 환상적으로 나오고 실전에서 전부 잃는다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """모든 전략의 부모."""

    #: 화면 출력용 이름. 하위 클래스에서 설정한다.
    name: str = "unnamed"

    @abstractmethod
    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """봉마다 목표 포지션(1=보유 / 0=현금)을 담은 Series를 반환."""

    def params(self) -> dict:
        """이 전략의 하이퍼파라미터 — 리포트에 찍힌다."""
        return {
            k: v for k, v in vars(self).items()
            if not k.startswith("_") and k != "name"
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.__class__.__name__}({args})"


def _hold_between(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """진입 신호 ~ 청산 신호 사이를 '보유(1)'로 채운다.

    진입/청산이 같은 봉에 겹치면 청산을 우선한다(보수적).
    """
    state = pd.Series(pd.NA, index=entries.index, dtype="Float64")
    state[entries.fillna(False)] = 1.0
    state[exits.fillna(False)] = 0.0
    return state.ffill().fillna(0.0).astype(int)
