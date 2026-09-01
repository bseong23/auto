#!/usr/bin/env python
"""조사: '공격적으로' 갈 수 있는 곳과 없는 곳.

"더 자주, 더 세게" 가 돈이 되려면 두 가지가 맞아야 한다 — 비용이 생각보다 작고,
전략이 그 빈도에서 살아남아야 한다. 셋을 잰다.

  ① 실제 슬리피지 — 호가창을 직접 걸어가며 시장가 매수 비용을 계산한다 (돈 안 든다)
  ② 봉 한계선     — 60분/4시간/일봉 중 어디까지 수수료를 이기나
  ③ 변동성 돌파   — 국내 크립토 봇의 표준 '공격적' 전략을 워크포워드로

호가창은 초 단위로 변하므로 스냅샷을 docs/data 에 저장해 문서를 재현 가능하게 한다.

    python scripts/11_research_aggressive.py            # 스냅샷으로 재현
    python scripts/11_research_aggressive.py --refresh  # 호가창 다시 찍기
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from upbit.backtest import CONSERVATIVE_SLIPPAGE, MEASURED_SLIPPAGE, UPBIT_FEE, run_backtest
from upbit.exchange import tick_ratio
from upbit.indicators import sma
from upbit.optimize import summarize_windows, windowed_returns
from upbit.strategies import BuyAndHoldStrategy, MACrossStrategy

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "data"
ORDERBOOK = SNAPSHOT / "orderbook_snapshot.json"
REPORT = ROOT / "docs" / "조사-공격적전략.md"

COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
ORDER_SIZES = [5_500, 50_000, 500_000, 5_000_000, 50_000_000]
CAPITAL = 1_000_000


def pct(v, digits=1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{digits}%}"


# ---------------------------------------------------------------- ① 슬리피지

def fetch_orderbooks() -> dict:
    import pyupbit

    books = {}
    for ticker in COINS:
        ob = pyupbit.get_orderbook(ticker)
        books[ticker] = [
            {"ask_price": u["ask_price"], "ask_size": u["ask_size"],
             "bid_price": u["bid_price"], "bid_size": u["bid_size"]}
            for u in ob["orderbook_units"]
        ]
    return {"fetched_at": datetime.now().isoformat(timespec="seconds"), "books": books}


def walk_asks(units: list[dict], krw: float):
    """시장가 매수 krw 원어치가 매도호가를 얼마나 먹는지 → 평균 체결가. 호가가 모자라면 None."""
    remaining, cost, qty = krw, 0.0, 0.0
    for u in units:
        price, size = u["ask_price"], u["ask_size"]
        take = min(remaining, price * size)
        qty += take / price
        cost += take
        remaining -= take
        if remaining <= 1e-9:
            return cost / qty
    return None


def experiment_slippage(snapshot: dict) -> tuple[str, dict]:
    header = "| 종목 | 호가단위/가격 | 스프레드 | " + " | ".join(
        f"{s // 10_000}만원" for s in ORDER_SIZES) + " | 매도호가 15단 |"
    divider = "|---|---:|---:|" + "---:|" * len(ORDER_SIZES) + "---:|"
    rows, measured = [], {}
    for ticker, units in snapshot["books"].items():
        best_ask, best_bid = units[0]["ask_price"], units[0]["bid_price"]
        mid = (best_ask + best_bid) / 2
        spread = (best_ask - best_bid) / mid
        cells = []
        for size in ORDER_SIZES:
            avg = walk_asks(units, size)
            slip = None if avg is None else (avg - mid) / mid
            cells.append("호가 부족" if slip is None else f"{slip:.3%}")
            if size == 500_000:
                measured[ticker] = slip
        depth = sum(u["ask_price"] * u["ask_size"] for u in units)
        rows.append(f"| {ticker.replace('KRW-', '')} | {tick_ratio(mid):.3%} | {spread:.3%} | "
                    + " | ".join(cells) + f" | {depth / 1e8:.1f}억 |")
    return "\n".join([header, divider] + rows), measured


# ---------------------------------------------------------------- ② 봉 한계선

INTERVALS = [("60분봉", "KRW-BTC_minute60_20000.csv"),
             ("4시간봉", "KRW-BTC_minute240_11000.csv"),
             ("일봉", "KRW-BTC_day_1800.csv")]


def experiment_intervals() -> tuple[str, dict]:
    strategy = MACrossStrategy(5, 20)
    rows = ["| 봉 | 기간 | 거래/년 | 실측 슬리피지 (0.02%) | 보수적 (0.05%) |",
            "|---|---|---:|---:|---:|"]
    stats = {}
    for label, file in INTERVALS:
        df = pd.read_csv(SNAPSHOT / file, index_col=0, parse_dates=True)
        years = (df.index[-1] - df.index[0]).days / 365.25
        trades_per_year = run_backtest(df, strategy).num_trades / years
        cells = []
        for slip in (MEASURED_SLIPPAGE, CONSERVATIVE_SLIPPAGE):
            s = summarize_windows(windowed_returns(df, strategy, 6, 60, UPBIT_FEE, slip, CAPITAL))
            cells.append(f"**{pct(s['평균'])}** ({s['흑자']}흑자)")
            if slip == MEASURED_SLIPPAGE:
                stats[label] = {**s, "trades": trades_per_year, "years": years}
        rows.append(f"| {label} | {df.index[0]:%y-%m}~{df.index[-1]:%y-%m} ({years:.1f}년) "
                    f"| {trades_per_year:.0f} | {cells[0]} | {cells[1]} |")
    return "\n".join(rows), stats


# ---------------------------------------------------------------- ③ 변동성 돌파

def breakout_daily_returns(df: pd.DataFrame, k: float, slip: float, ma_filter: int | None):
    """Larry Williams 변동성 돌파 — 하루 단위 거래수익률. 거래 없는 날은 0.

    목표가 = 오늘 시가 + k × (어제 고가 − 어제 저가). 오늘 고가가 목표가에 닿으면
    목표가에 사서 **내일 시가**에 판다. 갭상승으로 시가가 이미 목표가 위면 시가에 체결(불리).
    이 전략은 '포지션 시계열' 모델(봉 마감 신호 → 다음 봉 시가 체결)에 맞지 않아
    백테스터를 쓰지 않고 여기서 직접 계산한다. 연구용이다.
    """
    o, h, l = (df[c].to_numpy(float) for c in ("open", "high", "low"))
    ma = sma(df["close"], ma_filter).shift(1).to_numpy(float) if ma_filter else None
    rets = np.zeros(len(df))
    hits = 0
    for t in range(1, len(df) - 1):  # 내일 시가가 있어야 청산 가능
        target = o[t] + k * (h[t - 1] - l[t - 1])
        if ma is not None and (np.isnan(ma[t]) or o[t] <= ma[t]):
            continue
        if h[t] < target:
            continue
        fill = max(target, o[t]) * (1 + slip)
        exit_price = o[t + 1] * (1 - slip)
        rets[t] = exit_price / fill * (1 - UPBIT_FEE) ** 2 - 1
        hits += 1
    return rets, hits


def breakout_windows(df, k, slip, ma_filter, n=6, warmup=60):
    usable = len(df) - warmup
    window = usable // n
    out, hits_total = [], 0
    for i in range(n):
        start = warmup + i * window
        rets, hits = breakout_daily_returns(df.iloc[start - 1 : start + window], k, slip, ma_filter)
        out.append(float(np.prod(1 + rets) - 1))
        hits_total += hits
    return out, hits_total


def experiment_breakout(frames: dict) -> tuple[str, dict]:
    btc = frames["KRW-BTC"]
    _, every_day = breakout_daily_returns(btc, 0.0, 0.0, None)
    _, never = breakout_daily_returns(btc, 99.0, 0.0, None)
    assert every_day == len(btc) - 2 and never == 0, "변동성 돌파 구현 정합성 실패"

    rows = ["| 설정 | 실측 슬리피지 (0.02%) | 보수적 (0.05%) | 거래/년 |", "|---|---:|---:|---:|"]
    best = {"label": None, "mean": -9e9}
    for k in (0.3, 0.5, 0.7):
        for ma in (None, 5):
            label = f"k={k}" + (f" + 시가>MA{ma}" if ma else "")
            cells, hits_all, years = [], 0, 0.0
            for slip in (MEASURED_SLIPPAGE, CONSERVATIVE_SLIPPAGE):
                pooled = []
                for df in frames.values():
                    rs, hits = breakout_windows(df, k, slip, ma)
                    pooled += rs
                    if slip == MEASURED_SLIPPAGE:
                        hits_all += hits
                        years += (df.index[-1] - df.index[0]).days / 365.25
                s = summarize_windows(pooled)
                cells.append(f"{pct(s['평균'])} ({s['흑자']}흑자)")
                if slip == MEASURED_SLIPPAGE and s["평균"] > best["mean"]:
                    best = {"label": label, "mean": s["평균"], "wins": s["흑자"]}
            rows.append(f"| {label} | {cells[0]} | {cells[1]} | {hits_all / years:.0f} |")

    baselines = {}
    for name, strategy in (("존버 (같은 창)", BuyAndHoldStrategy()), ("SMA 5/20 일봉 (같은 창)", MACrossStrategy(5, 20))):
        pooled = []
        for df in frames.values():
            pooled += windowed_returns(df, strategy, 6, 60, UPBIT_FEE, MEASURED_SLIPPAGE, CAPITAL)
        s = summarize_windows(pooled)
        baselines[name] = s
        rows.append(f"| **{name}** | **{pct(s['평균'])}** ({s['흑자']}흑자) | — | — |")
    return "\n".join(rows), {"best": best, "baselines": baselines}


def main() -> None:
    ap = argparse.ArgumentParser(description="공격적 전략 조사")
    ap.add_argument("--refresh", action="store_true", help="호가창 스냅샷 다시 찍기")
    args = ap.parse_args()

    if args.refresh or not ORDERBOOK.exists():
        ORDERBOOK.write_text(json.dumps(fetch_orderbooks(), ensure_ascii=False, indent=1), encoding="utf-8")
    snapshot = json.loads(ORDERBOOK.read_text(encoding="utf-8"))

    slip_table, measured = experiment_slippage(snapshot)
    print("① 슬리피지 완료")
    interval_table, interval_stats = experiment_intervals()
    print("② 봉 한계선 완료")
    frames = {t: pd.read_csv(SNAPSHOT / f"{t}_day_1800.csv", index_col=0, parse_dates=True)
              for t in COINS if t != "KRW-DOGE"}
    breakout_table, breakout_stats = experiment_breakout(frames)
    print("③ 변동성 돌파 완료")

    best_interval = max(interval_stats.items(), key=lambda kv: kv[1]["평균"])[0]
    hourly = interval_stats["60분봉"]
    fee_drag_hourly = hourly["trades"] * 2 * UPBIT_FEE
    liquid = {t: v for t, v in measured.items() if v is not None and t != "KRW-DOGE"}
    doge = measured.get("KRW-DOGE")
    vb_best = breakout_stats["best"]
    sma_base = breakout_stats["baselines"]["SMA 5/20 일봉 (같은 창)"]
    hold_base = breakout_stats["baselines"]["존버 (같은 창)"]

    REPORT.write_text(f"""# 조사: '공격적으로' 갈 수 있는 곳과 없는 곳

> 이 문서는 `scripts/11_research_aggressive.py` 가 생성한다. 손으로 고치지 말 것.

생성: {datetime.now():%Y-%m-%d %H:%M} · 호가창 스냅샷 {snapshot['fetched_at']}

## 질문

엣지가 약한 전략을 "더 자주, 더 세게" 돌리면 기대수익이 아니라 분산만 커진다.
그래도 공격적으로 갈 여지가 있는지, **비용**과 **빈도**의 실제 한계를 잰다.

---

## ① 실제 슬리피지 — 호가창을 직접 걸어가며 계산

백테스트는 슬리피지를 편도 0.05% 로 **가정**해 왔다. 실제 호가창에서 시장가 매수가
얼마나 밀리는지 주문 크기별로 계산했다 (중간가 대비 평균 체결가, 편도).

{slip_table}

**BTC·ETH 는 500만 원까지 {pct(min(liquid.values()), 3)}~{pct(max(liquid.values()), 3)}** — 가정의 3분의 1이다.
소액 봇에게 슬리피지는 사실상 스프레드의 절반이고, 호가 깊이는 문제가 안 된다.

**DOGE 는 {pct(doge, 3)}.** 가격 대비 호가 단위가 커서 스프레드가 구조적으로 넓다.
이 비용이면 연 66회 매매 전략은 슬리피지만으로 연 -{doge * 2 * 66:.0%} 를 낸다.
09의 DOGE +959% 는 이 비용을 0.05% 로 가정했기 때문에 나온 숫자다 — **환상이었다.**

> 규칙: **자주 매매할 종목은 호가단위/가격이 0.01% 이하인 것만.** 저가 코인은 제외.
>
> 이 스냅샷은 평시({snapshot['fetched_at'][11:16]}) 기준이다. 급락장에서는 호가가 얇아져
> 몇 배 나빠질 수 있다. 다만 소액이면 그래도 0.1% 안쪽일 가능성이 높다.

이 측정을 근거로 `upbit/backtest.py` 에 `MEASURED_SLIPPAGE = 0.02%` 를 추가했다.
공식 실험기록은 계속 보수적인 0.05% 로 계산한다 — 문서마다 가정이 다르면 비교가 안 되니까.

---

## ② 봉 한계선 — 어디까지 짧게 갈 수 있나

같은 전략(SMA 5/20)을 봉만 바꿔, 실측·보수적 슬리피지 양쪽에서 워크포워드(6창).

{interval_table}

- **{best_interval}이 최적점이다.** 실측 슬리피지 기준 {pct(interval_stats[best_interval]['평균'])}, {interval_stats[best_interval]['흑자']} 흑자.
- **60분봉은 죽는다.** 연 {hourly['trades']:.0f}회 거래 → 수수료만 연 **-{fee_drag_hourly:.0%}**.
  슬리피지가 0이어도 못 이긴다. 빈도의 한계는 슬리피지가 아니라 **수수료**가 정한다.
- 60분봉 데이터는 {hourly['years']:.1f}년치뿐이라 다른 봉과 기간이 다르다. 그래도 수수료 계산은 기간과 무관하다.

---

## ③ 변동성 돌파 — 국내 크립토 봇의 표준 '공격적' 전략

Larry Williams 변동성 돌파: 목표가 = 시가 + k × 어제 변동폭. 닿으면 사서 내일 시가에 판다.
국내 pyupbit 튜토리얼의 표준이라, 정직하게 재봤다. DOGE 는 ①의 이유로 제외 (4코인 × 6창 = 24창).

갭상승으로 시가가 이미 목표가 위면 시가에 체결(불리하게), 수수료는 왕복, 슬리피지 편도씩.

{breakout_table}

**최선이 {vb_best['label']} {pct(vb_best['mean'])} ({vb_best['wins']}흑자).**
같은 창에서 SMA 5/20 일봉은 {pct(sma_base['평균'])}, 존버는 {pct(hold_base['평균'])}.

**지금 수수료 구조에서 변동성 돌파는 죽은 전략이다.** 매일 사고팔아 왕복 0.1% + 슬리피지를
내니, 하루 움직임의 조각을 먹어도 남는 게 없다. 흑자 창 비율이 높은 건(꾸준히 조금씩 번다)
사실이지만, 그 '조금'이 수수료보다 작다. 2017~2021년 튜토리얼의 유산이다.

---

## 결론 — 공격적으로 가는 실제 방법

| | 판정 | 근거 |
|---|---|---|
| 4시간봉 전환 | ✅ **한다** | 일봉 대비 +{(interval_stats['4시간봉']['평균'] - interval_stats['일봉']['평균']) * 100:.0f}%p, 실측 비용에서 성립 |
| 저가 코인 제외 | ✅ **한다** | DOGE 슬리피지 {pct(doge, 2)} — 전략 수익을 통째로 먹음 |
| 60분봉 이하 | ❌ | 수수료만 연 -{fee_drag_hourly:.0%} |
| 변동성 돌파 | ❌ | 최선 {pct(vb_best['mean'])} vs SMA {pct(sma_base['평균'])} |
| 파라미터 더 조이기 | ❌ | 실험기록: 훈련 +93% → 검증 -4% |
| 레버리지 | ❌ | 현물엔 없고, 약한 엣지에 레버리지는 파산 확률만 키움 |

"더 자주"의 한계는 4시간봉이고, "더 세게"의 수단은 없다. 남은 방향은 **빈도가 아니라
폭** — 한 종목의 타이밍이 아니라 **여러 종목 중 고르기**(크로스섹션 모멘텀)다.
그건 별도 조사(12)로.

## 한계

1. 호가창 스냅샷 1장. 급락장 비용은 못 봤다.
2. 60분봉은 2.3년치라 다른 봉과 기간이 다르다.
3. 변동성 돌파는 '고가가 목표가에 닿으면 정확히 목표가 체결'을 가정 — 실제로는
   N분마다 확인하므로 더 불리하게 체결된다. 즉 실전은 표보다 **더 나쁘다.**
4. 전부 SMA 5/20 기준. 다른 전략이면 봉 한계선이 다를 수 있다.

## 재현

```bash
python scripts/11_research_aggressive.py
```
""", encoding="utf-8")
    print(f"작성: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
