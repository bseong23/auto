"""다중 시간프레임 백테스트 — "손절을 얼마나 자주 확인해야 하나"를 재기 위한 엔진.

## 왜 필요한가

지금 백테스터는 일봉 하나로 모든 걸 판단한다. 손절은 그 봉의 **저가**가 손절선을
건드렸는지로 판정한다. 하지만 실전 봇은 하루에 한 번(봉 마감 때) 깨어나서
**그 순간의 현재가**만 본다. 둘은 다른 일이다:

- 백테스트: 장중에 손절선을 스쳤다가 회복해도 **손절된 것으로 친다**
- 실전: 스쳤다가 회복하면 **모른 채 지나간다**
- 백테스트: 갭하락해도 손절선 근처에서 잡는다(저가 판정)
- 실전: 다음 확인 시각까지 계속 물려 있다

이 차이가 얼마인지 재려면, **신호는 일봉으로 내고 체결·손절 감시는 분봉으로**
돌리는 엔진이 필요하다. 그게 이 파일이다.

## 미래참조 방지

일봉 D의 신호는 그 봉이 **마감된 시각**부터 유효하다. 업비트 일봉은 09:00 KST에
마감되므로, 일봉 D(인덱스 = D일 09:00)의 신호는 **D+1일 09:00부터** 쓸 수 있다.
그 이전 시간봉에 적용하면 미래를 본 것이 된다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import UPBIT_FEE, BacktestResult, Trade
from .risk import RiskRules
from .strategies.base import Strategy


def align_signals(
    positions: pd.Series, exec_index: pd.DatetimeIndex, bar_duration: pd.Timedelta
) -> pd.Series:
    """상위 시간프레임 신호를 하위 시간프레임 인덱스에 맞춘다.

    각 신호는 **그 봉이 마감된 뒤부터** 유효하다. 마감 시각 = 봉 시작 + 봉 길이.
    """
    effective_from = positions.index + bar_duration
    aligned = pd.Series(positions.to_numpy(), index=effective_from)
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()
    return aligned.reindex(exec_index, method="ffill").fillna(0).astype(int)


@dataclass
class StopCheck:
    """손절을 언제 확인할지."""

    #: 몇 개의 체결봉마다 확인할지. 1이면 매 봉, 24면 24봉마다(60분봉 기준 하루 한 번)
    every: int = 1
    #: 하루 한 번 확인이면 몇 시에 깨어날지 (KST 기준 시각). None이면 every만 사용
    at_hour: int | None = None

    def should_check(self, i: int, timestamp: pd.Timestamp) -> bool:
        if self.at_hour is not None:
            return timestamp.hour == self.at_hour
        return i % self.every == 0

    def describe(self) -> str:
        if self.at_hour is not None:
            return f"하루 1회 ({self.at_hour:02d}시)"
        return "매 봉" if self.every == 1 else f"{self.every}봉마다"


def run_multiframe_backtest(
    signal_df: pd.DataFrame,
    exec_df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskRules | None = None,
    stop_check: StopCheck | None = None,
    initial_capital: float = 1_000_000,
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
    signal_bar: pd.Timedelta = pd.Timedelta(days=1),
) -> BacktestResult:
    """신호는 `signal_df`(예: 일봉), 체결·손절 감시는 `exec_df`(예: 60분봉).

    실전 봇을 그대로 흉내낸다:
    - 정해진 시각에 깨어나 **그 시점 가격**을 읽는다 (봉 저가를 보지 않는다)
    - 손절선 아래면 즉시 시장가 매도
    - 손절 후엔 전략이 현금으로 돌아올 때까지 재진입 금지
    """
    if len(exec_df) < 2:
        raise ValueError("체결용 캔들이 최소 2개는 있어야 한다.")

    rules = risk or RiskRules()
    checker = stop_check or StopCheck(every=1)

    raw_positions = strategy.generate_positions(signal_df)
    target = align_signals(raw_positions, exec_df.index, signal_bar).to_numpy()

    atr_values = None
    if rules.needs_atr:
        from .indicators import atr

        # ATR은 **신호 시간프레임**에서 계산한다 — 일봉 전략의 손절폭은 일봉 변동성 기준
        daily_atr = atr(signal_df, rules.atr_window)
        atr_values = align_signals(
            daily_atr.fillna(-1), exec_df.index, signal_bar
        ).astype(float).replace(-1, np.nan).to_numpy()

    opens = exec_df["open"].to_numpy(dtype=float)
    closes = exec_df["close"].to_numpy(dtype=float)
    index = exec_df.index

    cash, coin, fees_paid = float(initial_capital), 0.0, 0.0
    equity = np.empty(len(exec_df), dtype=float)
    equity[0] = initial_capital
    trades: list[Trade] = []
    open_trade: Trade | None = None
    stop_price: float | None = None
    high_water: float | None = None
    blocked = False

    def sell(i: int, reason: str) -> None:
        nonlocal cash, coin, fees_paid, open_trade, stop_price, high_water
        fill = opens[i] * (1 - slippage)
        gross = coin * fill
        fees_paid += gross * fee
        cash = gross * (1 - fee)
        coin = 0.0
        if open_trade is not None:
            open_trade.exit_time = index[i]
            open_trade.exit_price = fill
            open_trade.exit_value = cash
            open_trade.bars_held = i - open_trade.entry_i
            open_trade.exit_reason = reason
            trades.append(open_trade)
            open_trade = None
        stop_price = high_water = None

    for i in range(1, len(exec_df)):
        price = opens[i]

        # 1) 손절 — 정해진 시각에 깨어나 '그 시점 가격'으로만 판단한다
        if coin > 0 and stop_price is not None and checker.should_check(i, index[i]):
            if price <= stop_price:
                sell(i, "손절")
                blocked = True
            elif rules.trailing:
                high_water = max(high_water or price, price)
                raised = rules.stop_price(
                    high_water, None if atr_values is None else atr_values[i - 1]
                )
                if raised is not None and raised > stop_price:
                    stop_price = raised

        want = target[i]
        if want == 0:
            blocked = False

        # 2) 전략 신호
        if want == 1 and coin == 0 and not blocked:
            fill = price * (1 + slippage)
            spend = cash
            fees_paid += spend * fee
            coin = spend * (1 - fee) / fill
            cash = 0.0
            open_trade = Trade(
                entry_time=index[i], entry_price=fill, entry_value=spend, entry_i=i
            )
            if rules.is_active:
                stop_price = rules.stop_price(
                    fill, None if atr_values is None else atr_values[i - 1]
                )
                high_water = fill
        elif want == 0 and coin > 0:
            sell(i, "신호")

        equity[i] = cash + coin * closes[i]

    if open_trade is not None:
        open_trade.bars_held = len(exec_df) - 1 - open_trade.entry_i
        trades.append(open_trade)

    return BacktestResult(
        strategy_name=f"{strategy.name} · 손절확인 {checker.describe()}",
        params={**strategy.params(), "손절확인": checker.describe()},
        equity=pd.Series(equity, index=index),
        positions=pd.Series(target, index=index),
        trades=trades,
        initial_capital=float(initial_capital),
        fee=fee,
        total_fees_paid=fees_paid,
        benchmark=exec_df["close"] / exec_df["close"].iloc[0] * initial_capital,
        risk=rules,
    )
