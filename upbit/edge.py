"""엣지 검증 — 이 전략이 번 돈이 '타이밍 실력'인가 '그냥 상승장에 있어서'인가.

## 문제

상승장에서는 아무렇게나 사고팔아도 돈을 번다. 시장에 들어가 있는 시간이 길수록
더 번다. 그러니 "+33% 벌었다"는 숫자만으로는 전략이 뭔가를 안다는 증거가 못 된다.
같은 시간만큼 **아무 때나** 들어가 있었어도 비슷하게 벌었을 수 있다.

## 방법: 노출 일치 무작위 기준선 (exposure-matched random baseline)

진짜 전략의 포지션 시계열에서 **구조만 빌리고 타이밍만 섞는다**:

- 보유 구간(1이 이어진 덩어리)들의 길이 목록
- 현금 구간(0이 이어진 덩어리)들의 길이 목록

이 두 목록을 각각 섞어서 다시 번갈아 이어 붙이면, **시장 노출 비율·거래 횟수·
보유기간 분포가 진짜와 정확히 같으면서 타이밍만 무작위인** 전략이 나온다.
이걸 수천 개 만들어 같은 수수료·슬리피지로 백테스트하면 분포가 생긴다.

진짜 전략이 그 분포의 50번째 백분위에 있으면 → 타이밍 실력 없음. 시장 베타다.
95번째 이상이면 → 우연으로 보기 어려운 뭔가가 있다 (이 표본 크기에서).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import UPBIT_FEE, run_backtest
from .strategies.base import Strategy


class FixedSeries(Strategy):
    """미리 만들어둔 포지션 시계열을 그대로 내는 전략 — 무작위 기준선용."""

    name = "고정시계열"

    def __init__(self, positions: pd.Series):
        self.positions = positions

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        return self.positions.reindex(df.index).fillna(0).astype(int)

    def params(self) -> dict:
        return {}


def runs_of(positions: pd.Series) -> tuple[int, list[int]]:
    """0/1 시계열을 (첫 값, 연속 구간 길이들)로 압축한다. 예: 0011100 → (0, [2,3,2])"""
    values = positions.to_numpy().astype(int)
    if len(values) == 0:
        return 0, []
    change_points = np.flatnonzero(np.diff(values)) + 1
    boundaries = np.concatenate([[0], change_points, [len(values)]])
    lengths = np.diff(boundaries).tolist()
    return int(values[0]), lengths


def shuffle_timing(positions: pd.Series, rng: np.random.Generator) -> pd.Series:
    """보유/현금 구간의 길이는 그대로 두고 순서만 섞는다.

    노출 비율, 거래 횟수, 보유기간 분포가 원본과 정확히 같다. 타이밍만 다르다.
    """
    first, lengths = runs_of(positions)
    if not lengths:
        return positions.copy()

    hold_runs = lengths[(1 - first)::2]  # 값이 1인 구간들
    cash_runs = lengths[first::2]        # 값이 0인 구간들
    hold_runs = list(rng.permutation(hold_runs)) if hold_runs else []
    cash_runs = list(rng.permutation(cash_runs)) if cash_runs else []

    rebuilt: list[int] = []
    value = first
    hold_iter, cash_iter = iter(hold_runs), iter(cash_runs)
    while len(rebuilt) < len(positions):
        source = hold_iter if value == 1 else cash_iter
        length = next(source, None)
        if length is None:
            break
        rebuilt.extend([value] * length)
        value = 1 - value
    rebuilt = rebuilt[: len(positions)]
    rebuilt += [0] * (len(positions) - len(rebuilt))
    return pd.Series(rebuilt, index=positions.index, dtype=int)


@dataclass
class EdgeReport:
    strategy_name: str
    real_return: float
    random_returns: np.ndarray
    exposure: float
    num_trades: int

    @property
    def percentile(self) -> float:
        """진짜 전략이 무작위 분포에서 몇 번째 백분위인가 (0~100)."""
        return float((self.random_returns < self.real_return).mean() * 100)

    @property
    def random_mean(self) -> float:
        return float(self.random_returns.mean())

    @property
    def random_std(self) -> float:
        return float(self.random_returns.std(ddof=1)) if len(self.random_returns) > 1 else 0.0

    @property
    def z_score(self) -> float:
        return (self.real_return - self.random_mean) / self.random_std if self.random_std else 0.0

    @property
    def beats_random_by(self) -> float:
        """무작위 평균 대비 초과수익. 이게 '타이밍이 기여한 몫'의 추정치다."""
        return self.real_return - self.random_mean

    def verdict(self) -> str:
        p = self.percentile
        if p >= 97.5:
            return "우연으로 보기 어렵다 (상위 2.5%)"
        if p >= 90:
            return "약한 증거 (상위 10%)"
        if p <= 10:
            return "무작위보다 나쁘다 — 타이밍이 해를 끼친다"
        return "무작위와 구분되지 않는다 — 시장 베타다"


def measure_edge(
    df: pd.DataFrame,
    strategy: Strategy,
    n_random: int = 1000,
    seed: int = 0,
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
) -> EdgeReport:
    """진짜 전략 vs 노출 일치 무작위 기준선 n_random 개."""
    rng = np.random.default_rng(seed)
    positions = strategy.generate_positions(df).reindex(df.index).fillna(0).astype(int)
    real = run_backtest(df, FixedSeries(positions), fee=fee, slippage=slippage)

    randoms = np.empty(n_random)
    for k in range(n_random):
        shuffled = shuffle_timing(positions, rng)
        randoms[k] = run_backtest(df, FixedSeries(shuffled), fee=fee, slippage=slippage).total_return

    return EdgeReport(
        strategy_name=strategy.name,
        real_return=real.total_return,
        random_returns=randoms,
        exposure=float(positions.mean()),
        num_trades=real.num_trades,
    )
