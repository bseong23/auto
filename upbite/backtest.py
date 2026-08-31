"""백테스팅 엔진 — "이 전략을 과거에 돌렸으면 어땠을까".

정직한 백테스트를 위한 두 가지 원칙:

1. **미래를 보지 않는다.** 신호는 t봉 종가로 판단하고, 체결은 t+1봉 시가로 한다.
   같은 봉 종가에 사고파는 코드는 실제로는 불가능한 거래라 수익률이 부풀려진다.
2. **수수료·슬리피지를 뺀다.** 업비트 원화마켓 0.05%. 자주 사고팔수록 이게 수익을 갉아먹는다.

결과가 좋아 보여도 그건 '과거 이 구간에서 그랬다'는 뜻일 뿐, 미래 보장이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies.base import Strategy

#: 업비트 원화마켓 기본 수수료 (편도 0.05%)
UPBIT_FEE = 0.0005


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    entry_value: float = 0.0
    exit_value: float = 0.0
    bars_held: int = 0
    entry_i: int = -1

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def return_pct(self) -> float:
        """수수료까지 반영한 이 거래의 손익률."""
        if self.is_open or self.entry_value == 0:
            return float("nan")
        return self.exit_value / self.entry_value - 1


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    equity: pd.Series
    positions: pd.Series
    trades: list[Trade]
    initial_capital: float
    fee: float
    total_fees_paid: float
    benchmark: pd.Series = field(repr=False, default=None)

    # ---- 성과 지표 ----
    @property
    def total_return(self) -> float:
        return self.equity.iloc[-1] / self.initial_capital - 1

    @property
    def benchmark_return(self) -> float:
        if self.benchmark is None:
            return float("nan")
        return self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1

    @property
    def years(self) -> float:
        span = self.equity.index[-1] - self.equity.index[0]
        return max(span.total_seconds() / (365.25 * 24 * 3600), 1e-9)

    @property
    def cagr(self) -> float:
        """연평균 복리 수익률. 기간이 1년 미만이면 과대해석 금물."""
        growth = self.equity.iloc[-1] / self.initial_capital
        if growth <= 0:
            return -1.0
        return growth ** (1 / self.years) - 1

    @property
    def mdd(self) -> float:
        """최대낙폭 — 고점 대비 얼마나 깊게 꼬라박았나. 음수."""
        curve = self.equity
        return float((curve / curve.cummax() - 1).min())

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]

    @property
    def num_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        if not closed:
            return float("nan")
        return sum(t.return_pct > 0 for t in closed) / len(closed)

    @property
    def avg_trade_return(self) -> float:
        closed = self.closed_trades
        return float(np.mean([t.return_pct for t in closed])) if closed else float("nan")

    @property
    def profit_factor(self) -> float:
        """총이익 / 총손실. 1 미만이면 잃는 전략."""
        wins = [t.return_pct for t in self.closed_trades if t.return_pct > 0]
        losses = [-t.return_pct for t in self.closed_trades if t.return_pct <= 0]
        if not losses:
            return float("inf") if wins else float("nan")
        return sum(wins) / sum(losses)

    @property
    def sharpe(self) -> float:
        """샤프지수(무위험수익률 0 가정). 대략 1 이상이면 괜찮은 편."""
        rets = self.equity.pct_change().dropna()
        if len(rets) < 2 or rets.std() == 0:
            return float("nan")
        bars_per_year = len(rets) / self.years
        return float(rets.mean() / rets.std() * np.sqrt(bars_per_year))

    @property
    def exposure(self) -> float:
        """자산을 실제로 들고 있던 시간 비중."""
        return float(self.positions.mean())

    def summary(self) -> str:
        pct = lambda x: "  n/a " if pd.isna(x) else f"{x:>+7.2%}"
        lines = [
            f"전략        : {self.strategy_name}  {self.params}",
            f"기간        : {self.equity.index[0]:%Y-%m-%d} ~ {self.equity.index[-1]:%Y-%m-%d} ({self.years:.2f}년)",
            f"초기자본    : {self.initial_capital:,.0f}원  →  최종 {self.equity.iloc[-1]:,.0f}원",
            "-" * 58,
            f"총수익률    : {pct(self.total_return)}      (존버: {pct(self.benchmark_return)})",
            f"CAGR        : {pct(self.cagr)}",
            f"최대낙폭MDD : {pct(self.mdd)}",
            f"샤프지수    : {self.sharpe:>8.2f}",
            f"승률        : {pct(self.win_rate)}  ({self.num_trades}회 거래)",
            f"평균손익/거래: {pct(self.avg_trade_return)}",
            f"손익비(PF)  : {self.profit_factor:>8.2f}",
            f"보유시간비중: {pct(self.exposure)}",
            f"낸 수수료   : {self.total_fees_paid:,.0f}원 (자본의 {self.total_fees_paid / self.initial_capital:.2%})",
        ]
        return "\n".join(lines)

    def trades_frame(self) -> pd.DataFrame:
        rows = [
            {
                "진입": t.entry_time,
                "진입가": t.entry_price,
                "청산": t.exit_time,
                "청산가": t.exit_price,
                "손익률": t.return_pct,
                "보유봉수": t.bars_held,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows)


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 1_000_000,
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
) -> BacktestResult:
    """전략을 과거 데이터에 돌려 성과를 계산한다.

    slippage: 호가 미끄러짐. 시장가 주문은 원하는 가격에 정확히 안 붙는다.
    """
    if len(df) < 2:
        raise ValueError("캔들이 최소 2개는 있어야 백테스트가 가능하다.")

    positions = strategy.generate_positions(df).reindex(df.index).fillna(0).astype(int)
    if not positions.isin((0, 1)).all():
        raise ValueError(f"{strategy.name}: 포지션은 0 또는 1만 허용된다.")

    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    target = positions.to_numpy()
    index = df.index

    cash = float(initial_capital)
    coin = 0.0
    fees_paid = 0.0
    equity = np.empty(len(df), dtype=float)
    equity[0] = initial_capital
    trades: list[Trade] = []
    open_trade: Trade | None = None

    for i in range(1, len(df)):
        want = target[i - 1]           # 직전 봉 '종가'에서 내린 판단
        holding = coin > 0
        price = opens[i]               # 체결은 이번 봉 '시가'

        if want == 1 and not holding:
            fill = price * (1 + slippage)
            spend = cash
            fees_paid += spend * fee
            coin = spend * (1 - fee) / fill
            cash = 0.0
            open_trade = Trade(
                entry_time=index[i], entry_price=fill, entry_value=spend, entry_i=i
            )

        elif want == 0 and holding:
            fill = price * (1 - slippage)
            gross = coin * fill
            fees_paid += gross * fee
            cash = gross * (1 - fee)
            coin = 0.0
            if open_trade is not None:
                open_trade.exit_time = index[i]
                open_trade.exit_price = fill
                open_trade.exit_value = cash
                open_trade.bars_held = i - open_trade.entry_i
                trades.append(open_trade)
                open_trade = None

        equity[i] = cash + coin * closes[i]

    if open_trade is not None:  # 마지막까지 들고 있는 미청산 포지션
        open_trade.bars_held = len(df) - 1 - open_trade.entry_i
        trades.append(open_trade)

    benchmark = df["close"] / df["close"].iloc[0] * initial_capital

    return BacktestResult(
        strategy_name=strategy.name,
        params=strategy.params(),
        equity=pd.Series(equity, index=index),
        positions=positions,
        trades=trades,
        initial_capital=float(initial_capital),
        fee=fee,
        total_fees_paid=fees_paid,
        benchmark=benchmark,
    )


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """여러 전략 결과를 한 표로 — 총수익률 내림차순."""
    rows = [
        {
            "전략": r.strategy_name,
            "총수익률": r.total_return,
            "CAGR": r.cagr,
            "MDD": r.mdd,
            "샤프": r.sharpe,
            "승률": r.win_rate,
            "거래수": r.num_trades,
            "손익비": r.profit_factor,
            "수수료(원)": round(r.total_fees_paid),
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("총수익률", ascending=False).reset_index(drop=True)
