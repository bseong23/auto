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

from .risk import OpenPosition, RiskRules, atr_series
from .strategies.base import Strategy

#: 업비트 원화마켓 기본 수수료 (편도 0.05%)
UPBIT_FEE = 0.0005

#: 보수적 슬리피지 가정 (편도). 공식 문서의 숫자는 전부 이걸로 계산했다.
CONSERVATIVE_SLIPPAGE = 0.0005

#: 실측 슬리피지 (편도). 2026-09-01 업비트 호가창에서 BTC/ETH 1만~500만원 시장가 매수를
#: 시뮬레이션한 값 0.015~0.017% 를 반올림. 저가·넓은 스프레드 코인(DOGE 0.44%)엔 해당 없다.
#: 출처: scripts/11_research_aggressive.py → docs/조사-공격적전략.md
MEASURED_SLIPPAGE = 0.0002


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
    exit_reason: str = "신호"

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
    risk: RiskRules = field(default_factory=lambda: RiskRules())

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
            f"손절규칙    : {self.risk.describe()}",
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
                "청산사유": t.exit_reason if not t.is_open else "미청산",
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
    risk: RiskRules | None = None,
) -> BacktestResult:
    """전략을 과거 데이터에 돌려 성과를 계산한다.

    slippage: 호가 미끄러짐. 시장가 주문은 원하는 가격에 정확히 안 붙는다.
    risk:     손절/익절 규칙. 전략 신호보다 **우선** 적용된다.

    한 봉 안에서의 처리 순서 (이 순서가 정확성의 핵심):
      1. 보유 중이면 손절/익절부터 확인한다 — 전략이 "계속 보유"라고 해도 손절이 이긴다.
      2. 손절로 나갔으면 전략이 한 번 현금으로 돌아올 때까지 재진입을 막는다.
         (안 막으면 손절 다음 봉에 바로 다시 사서 계속 얻어맞는다.)
      3. 그 다음 전략 신호를 처리한다.
      4. 추적 손절선은 손절 판정이 **끝난 뒤에** 이번 봉 고점으로 갱신한다.
         판정 전에 올리면 이번 봉 고점을 미리 아는 셈이라 미래참조가 된다.
    """
    if len(df) < 2:
        raise ValueError("캔들이 최소 2개는 있어야 백테스트가 가능하다.")

    positions = strategy.generate_positions(df).reindex(df.index).fillna(0).astype(int)
    if not positions.isin((0, 1)).all():
        raise ValueError(f"{strategy.name}: 포지션은 0 또는 1만 허용된다.")

    rules = risk or RiskRules()
    atr_values = atr_series(df, rules)
    atr_array = None if atr_values is None else atr_values.to_numpy(dtype=float)

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
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
    position: OpenPosition | None = None
    blocked = False  # 손절 직후 재진입 차단

    def known_atr(i: int) -> float | None:
        """i번 봉 진입/판정 시점에 알 수 있는 ATR — 직전 확정봉 값."""
        if atr_array is None or i - 1 < 0:
            return None
        value = atr_array[i - 1]
        return None if np.isnan(value) else float(value)

    def close_position(i: int, fill: float, reason: str) -> None:
        nonlocal cash, coin, fees_paid, open_trade, position
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
        position = None

    for i in range(1, len(df)):
        want = target[i - 1]  # 직전 봉 '종가'에서 내린 판단

        # 1) 손절/익절이 전략보다 우선한다
        if coin > 0 and position is not None:
            hit = position.check_exit(opens[i], lows[i], highs[i])
            if hit is not None:
                raw_price, reason = hit
                close_position(i, raw_price * (1 - slippage), reason)
                blocked = True
            else:
                # 4) 판정이 끝난 뒤에만 추적 손절선을 올린다
                position.update_trailing(highs[i], known_atr(i), rules)

        # 2) 전략이 현금으로 돌아오면 재진입 차단 해제
        if want == 0:
            blocked = False

        # 3) 전략 신호 처리
        holding = coin > 0
        price = opens[i]  # 체결은 이번 봉 '시가'

        if want == 1 and not holding and not blocked:
            fill = price * (1 + slippage)
            spend = cash
            fees_paid += spend * fee
            coin = spend * (1 - fee) / fill
            cash = 0.0
            open_trade = Trade(
                entry_time=index[i], entry_price=fill, entry_value=spend, entry_i=i
            )
            if rules.is_active:
                position = OpenPosition(
                    entry_price=fill,
                    stop=rules.stop_price(fill, known_atr(i)),
                    target=rules.target_price(fill),
                    high_water=fill,
                )

        elif want == 0 and holding:
            close_position(i, price * (1 - slippage), "신호")

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
        risk=rules,
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
