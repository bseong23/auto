"""포트폴리오 백테스트 — 여러 종목 중 **고르기**의 성과를 잰다.

지금까지의 백테스터는 한 종목을 '언제' 사고 팔지(타이밍)만 다뤘다. 이 엔진은
다른 질문에 답한다: **"여러 코인 중 최근 제일 강한 것들을 들고 정기적으로 갈아타면
어떻게 되나"** (크로스섹션 모멘텀 / 상대강도). 학계에서 크립토에 가장 일관되게
확인된 현상이 타이밍이 아니라 이것이다.

## 규칙

- 리밸런싱 날 t 의 **시가**에 거래한다. 신호는 t-1 종가까지만 쓴다 (미래참조 없음).
- 모멘텀 = close[t-1] / close[t-1-lookback] − 1. 상장한 지 lookback 일이 안 된 종목은 후보 아님.
- 후보 중 상위 top_k 를 **동일 비중 1/top_k** 로 든다. `absolute=True` 면 모멘텀이 양수인
  종목만 후보다 — 다 음수면 현금. 이게 하락장에서 살아남는 장치(듀얼 모멘텀)다.
  후보가 top_k 보다 적으면 빈 슬롯은 현금으로 둔다(비중을 늘려 채우지 않는다).
- 거래비용 = |거래금액| × (수수료 + 그 종목의 슬리피지). 슬리피지는 종목별로 다를 수 있다
  (저가 코인은 호가 단위 때문에 구조적으로 비싸다 — 11 조사).
- 보유 중 데이터가 끊긴 종목(상장폐지)은 마지막 종가로 강제 청산한다.

## 무작위 기준선

`selector` 를 바꾸면 "상위 k 를 고르는 것"과 "후보 중 아무거나 k 개"를 같은 규칙·같은
비용으로 비교할 수 있다. 진짜가 무작위 분포의 어디에 있는지가 **선택 실력**의 증거다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import UPBIT_FEE

Selector = Callable[[pd.Series], list[str]]


def top_k_selector(k: int) -> Selector:
    def pick(momentum: pd.Series) -> list[str]:
        return list(momentum.sort_values(ascending=False).index[:k])
    return pick


def bottom_k_selector(k: int) -> Selector:
    """제일 약한 k 개 — 모멘텀이 아니라 **반전**에 베팅. 모멘텀 결과가 무작위보다 나쁘면
    그 반대가 무작위보다 좋은지로 '버그인지 진짜 반전인지'를 가른다."""
    def pick(momentum: pd.Series) -> list[str]:
        return list(momentum.sort_values(ascending=True).index[:k])
    return pick


def random_selector(k: int, rng: np.random.Generator) -> Selector:
    """후보 중 아무거나 k 개 — 순위 매기기에 실력이 있는지 재는 기준선."""
    def pick(momentum: pd.Series) -> list[str]:
        names = list(momentum.index)
        if len(names) <= k:
            return names
        return list(rng.choice(names, size=k, replace=False))
    return pick


@dataclass
class PortfolioResult:
    equity: pd.Series
    invested: pd.Series                      # 날마다 투자된 비중 (0~1)
    weights: pd.DataFrame = field(repr=False)  # 리밸런싱 날의 목표 비중
    turnover: pd.Series = field(repr=False)    # 리밸런싱 날의 회전율 (|Δ비중| 합)
    initial_capital: float = 1_000_000
    costs_paid: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_capital - 1)

    @property
    def years(self) -> float:
        span = self.equity.index[-1] - self.equity.index[0]
        return max(span.total_seconds() / (365.25 * 86_400), 1e-9)

    @property
    def cagr(self) -> float:
        growth = self.equity.iloc[-1] / self.initial_capital
        return float(growth ** (1 / self.years) - 1) if growth > 0 else -1.0

    @property
    def mdd(self) -> float:
        return float((self.equity / self.equity.cummax() - 1).min())

    @property
    def sharpe(self) -> float:
        rets = self.equity.pct_change().dropna()
        if len(rets) < 2 or rets.std() == 0:
            return float("nan")
        return float(rets.mean() / rets.std() * np.sqrt(len(rets) / self.years))

    @property
    def exposure(self) -> float:
        return float(self.invested.mean())

    @property
    def num_rebalances(self) -> int:
        return int(len(self.weights))

    @property
    def avg_turnover(self) -> float:
        return float(self.turnover.mean()) if len(self.turnover) else 0.0

    def summary(self) -> str:
        return (f"총수익 {self.total_return:+.1%} | CAGR {self.cagr:+.1%} | MDD {self.mdd:.1%} "
                f"| 샤프 {self.sharpe:.2f} | 노출 {self.exposure:.0%} "
                f"| 리밸런싱 {self.num_rebalances}회 · 회전율 {self.avg_turnover:.0%} "
                f"| 비용 {self.costs_paid:,.0f}원")


def _as_series(value, index: list[str], default: float) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index).fillna(default).astype(float)
    return pd.Series(float(value), index=index)


def run_portfolio(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    lookback: int = 28,
    top_k: int = 3,
    rebalance_every: int = 7,
    absolute: bool = True,
    fee: float = UPBIT_FEE,
    slippage: float | pd.Series = 0.0002,
    initial_capital: float = 1_000_000,
    selector: Selector | None = None,
) -> PortfolioResult:
    """opens/closes: 인덱스=날짜, 열=종목. 상장 전/폐지 후는 NaN."""
    if opens.shape != closes.shape or not opens.index.equals(closes.index) \
            or list(opens.columns) != list(closes.columns):
        raise ValueError("opens 와 closes 는 같은 인덱스·같은 열이어야 한다.")
    if top_k < 1 or lookback < 1 or rebalance_every < 1:
        raise ValueError("top_k, lookback, rebalance_every 는 1 이상이어야 한다.")

    tickers = list(closes.columns)
    slip = _as_series(slippage, tickers, 0.0002)
    pick = selector or top_k_selector(top_k)

    o_arr = opens.to_numpy(dtype=float)
    c_arr = closes.to_numpy(dtype=float)
    n, m = c_arr.shape

    units = np.zeros(m)
    cash = float(initial_capital)
    costs_paid = 0.0
    equity = np.empty(n)
    invested = np.empty(n)
    weight_rows: dict = {}
    turnover_rows: dict = {}
    last_valid_close = np.full(m, np.nan)

    first_rebalance = lookback + 1  # close[t-1] 과 close[t-1-lookback] 둘 다 필요

    for t in range(n):
        o = o_arr[t]
        c = c_arr[t]

        # --- 리밸런싱: t-1 까지의 정보로 판단, t 시가에 체결 ---
        if t >= first_rebalance and (t - first_rebalance) % rebalance_every == 0:
            prev, past = c_arr[t - 1], c_arr[t - 1 - lookback]
            with np.errstate(invalid="ignore", divide="ignore"):
                momentum = prev / past - 1
            tradable = ~np.isnan(momentum) & ~np.isnan(o) & (o > 0)
            if absolute:
                tradable &= momentum > 0
            candidates = pd.Series(momentum[tradable], index=np.array(tickers)[tradable])
            chosen = pick(candidates) if len(candidates) else []

            target_w = np.zeros(m)
            for name in chosen:
                target_w[tickers.index(name)] = 1.0 / top_k

            # 오늘 시가가 없는 보유 종목은 이번엔 못 판다 → 마지막 종가로 평가하고 그대로 둔다
            price_now = np.where(np.isnan(o), last_valid_close, o)
            frozen = (units > 0) & np.isnan(o)
            value = np.where(np.isnan(price_now), 0.0, units * price_now)
            total = cash + value.sum()
            if frozen.any():
                target_w[frozen] = value[frozen] / total if total > 0 else 0.0

            # 비용은 예산에서 나간다 — 수수료를 낸 만큼 덜 산다. 그래야 현금이 음수가 안 된다.
            rate = fee + slip.to_numpy()
            trade = target_w * total - value
            trade[frozen] = 0.0
            budget = total - float((np.abs(trade) * rate).sum())   # 1차 비용 추정
            target_value = target_w * budget
            trade = target_value - value
            trade[frozen] = 0.0
            cost = float((np.abs(trade) * rate).sum())

            new_units = units.copy()
            active = ~frozen
            with np.errstate(invalid="ignore", divide="ignore"):
                new_units[active] = np.where(target_w[active] > 0,
                                             target_value[active] / price_now[active], 0.0)
            new_units = np.nan_to_num(new_units, nan=0.0)
            spent = float((new_units * np.nan_to_num(price_now, nan=0.0)).sum())
            cash = total - spent - cost
            if cash < 0:  # 2차 오차(비용의 비용)는 매수를 그만큼 줄여 흡수한다
                scale = (total - cost) / spent if spent > 0 else 0.0
                new_units *= max(scale, 0.0)
                cash = 0.0
            units = new_units
            costs_paid += cost
            weight_rows[closes.index[t]] = pd.Series(target_w, index=tickers)
            turnover_rows[closes.index[t]] = float(np.abs(trade).sum() / total) if total > 0 else 0.0

        # --- 일말 평가; 데이터가 끊긴 보유 종목은 마지막 종가로 강제 청산 ---
        dead = (units > 0) & np.isnan(c)
        if dead.any():
            px = np.nan_to_num(last_valid_close, nan=0.0)
            proceeds = units[dead] * px[dead]
            cost = float((proceeds * (fee + slip.to_numpy()[dead])).sum())
            cash += float(proceeds.sum()) - cost
            costs_paid += cost
            units[dead] = 0.0
        last_valid_close = np.where(np.isnan(c), last_valid_close, c)

        held = float(np.nansum(units * c))
        equity[t] = cash + held
        invested[t] = held / equity[t] if equity[t] > 0 else 0.0

    return PortfolioResult(
        equity=pd.Series(equity, index=closes.index),
        invested=pd.Series(invested, index=closes.index),
        weights=pd.DataFrame(weight_rows).T if weight_rows else pd.DataFrame(columns=tickers),
        turnover=pd.Series(turnover_rows, dtype=float),
        initial_capital=float(initial_capital),
        costs_paid=costs_paid,
        params={"lookback": lookback, "top_k": top_k, "rebalance_every": rebalance_every,
                "absolute": absolute, "fee": fee},
    )


def windowed_portfolio_returns(
    opens: pd.DataFrame, closes: pd.DataFrame, n_splits: int = 6, warmup: int = 120, **params
) -> list[float]:
    """검증창 n_splits 개에서 각 창 동안 번 수익률. 창 앞에 warmup 을 붙여 모멘텀을 데운다."""
    usable = len(closes) - warmup
    if usable < n_splits * 30:
        return []
    window = usable // n_splits
    out = []
    for i in range(n_splits):
        start = warmup + i * window
        sl = slice(start - warmup, start + window)
        equity = run_portfolio(opens.iloc[sl], closes.iloc[sl], **params).equity
        base = equity.iloc[warmup]
        if base > 0:
            out.append(float(equity.iloc[-1] / base - 1))
    return out


def load_universe(folder) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """docs/data/universe/*.csv → (opens, closes) 넓은 표 + 선정 메타데이터."""
    import json
    from pathlib import Path

    folder = Path(folder)
    meta = json.loads((folder / "_selection.json").read_text(encoding="utf-8"))
    opens, closes = {}, {}
    for row in meta["rows"]:
        df = pd.read_csv(folder / f"{row['ticker']}.csv", index_col=0, parse_dates=True)
        opens[row["ticker"]] = df["open"]
        closes[row["ticker"]] = df["close"]
    return pd.DataFrame(opens).sort_index(), pd.DataFrame(closes).sort_index(), meta
