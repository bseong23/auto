#!/usr/bin/env python
"""조사: 크로스섹션 모멘텀 — 한 종목의 타이밍이 아니라 여러 종목 중 고르기.

지금까지 본 건 전부 "BTC 를 언제 사고 팔까"였다. 학계에서 크립토에 가장 일관되게
확인된 현상은 그게 아니라 **상대강도**다: 여러 코인 중 최근 제일 강한 것들을 들고
정기적으로 갈아탄다. 구조가 다른 베팅이라 기존 전략과 합치면 분산 효과도 있다.

전부 워크포워드 검증창 성적으로 판단하고, 무작위 선택 기준선으로 '고르기에 실력이
있나'를 따로 잰다. 헤드라인 설정(lookback 28일, 상위 3개, 주 1회)은 **결과를 보기 전에**
교과서 기본값으로 정했다 — 표를 보고 제일 좋은 칸을 고르면 그게 과최적화다.

    python scripts/12_research_momentum.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from upbit.backtest import MEASURED_SLIPPAGE, UPBIT_FEE
from upbit.optimize import summarize_windows, windowed_returns
from upbit.portfolio import (
    bottom_k_selector,
    load_universe,
    random_selector,
    run_portfolio,
    windowed_portfolio_returns,
)
from upbit.strategies import MACrossStrategy

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "docs" / "data" / "universe"
REPORT = ROOT / "docs" / "조사-크로스섹션모멘텀.md"

SPLITS, WARMUP = 6, 120
HEADLINE = dict(lookback=28, top_k=3, rebalance_every=7, absolute=True)  # 결과 보기 전에 고정


def pct(v, d=1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}%}"


def per_ticker_slippage(meta: dict, stress: float = 1.0) -> pd.Series:
    """종목별 슬리피지 하한 = max(실측 0.02%, 호가단위/가격 의 절반). stress 로 배수를 준다.

    스프레드는 최소 1틱이므로 절반 틱이 시장가의 구조적 최소 비용이다. 실측(11)에서
    DOGE 는 이보다 5배 나빴다 — 그래서 stress 배수로도 본다.
    """
    return pd.Series({r["ticker"]: max(MEASURED_SLIPPAGE, 0.5 * r["tick_ratio"]) * stress
                      for r in meta["rows"]})


def windows(opens, closes, slip, **params) -> list[float]:
    return windowed_portfolio_returns(opens, closes, SPLITS, WARMUP, fee=UPBIT_FEE,
                                      slippage=slip, **params)


def main() -> None:
    ap = argparse.ArgumentParser(description="크로스섹션 모멘텀 조사")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    opens, closes, meta = load_universe(UNIVERSE)
    tickers = list(closes.columns)
    slip = per_ticker_slippage(meta)
    slip_x2 = per_ticker_slippage(meta, stress=2.0)
    print(f"유니버스 {len(tickers)}개, {closes.index[0]:%Y-%m-%d} ~ {closes.index[-1]:%Y-%m-%d}")

    # ---- 1. 파라미터 격자 (안정성을 보기 위한 것이지 고르기 위한 게 아니다) ----
    grid_rows = ["| lookback | 상위 k | 검증 평균 | 흑자 | 최악 | 슬리피지 ×2 |", "|---:|---:|---:|---:|---:|---:|"]
    grid = {}
    for lb in (7, 14, 28, 56, 90, 180):
        for k in (3, 5):
            params = dict(lookback=lb, top_k=k, rebalance_every=7, absolute=True)
            s = summarize_windows(windows(opens, closes, slip, **params))
            s2 = summarize_windows(windows(opens, closes, slip_x2, **params))
            grid[(lb, k)] = s
            mark = " ←" if (lb, k) == (HEADLINE["lookback"], HEADLINE["top_k"]) else ""
            grid_rows.append(f"| {lb} | {k} | **{pct(s['평균'])}** | {s['흑자']} | {pct(s['최악'])} | {pct(s2['평균'])}{mark} |")
    print("1. 격자 완료")

    # ---- 2. 헤드라인 vs 기준선 (같은 창) ----
    head = summarize_windows(windows(opens, closes, slip, **HEADLINE))
    reversal = summarize_windows(windows(opens, closes, slip, **HEADLINE,
                                         selector=bottom_k_selector(HEADLINE["top_k"])))
    no_abs = summarize_windows(windows(opens, closes, slip, **{**HEADLINE, "absolute": False}))

    n = len(tickers)
    ew = summarize_windows(windows(opens, closes, slip, lookback=1, top_k=n, rebalance_every=7,
                                   absolute=False, selector=lambda m: list(m.index)))
    btc = closes["KRW-BTC"]
    btc_hold = []
    usable = len(closes) - WARMUP
    w = usable // SPLITS
    for i in range(SPLITS):
        a, b = WARMUP + i * w, WARMUP + (i + 1) * w - 1
        btc_hold.append(float(btc.iloc[b] / btc.iloc[a] - 1))
    btc_hold_s = summarize_windows(btc_hold)
    btc_df = pd.read_csv(UNIVERSE / "KRW-BTC.csv", index_col=0, parse_dates=True)
    sma = summarize_windows(windowed_returns(btc_df, MACrossStrategy(5, 20), SPLITS, WARMUP,
                                             UPBIT_FEE, MEASURED_SLIPPAGE, 1_000_000))
    print("2. 기준선 완료")

    base_rows = ["| 전략 | 검증 평균 | 흑자 | 최악 |", "|---|---:|---:|---:|"]
    for name, s in (("**모멘텀 상위3 (헤드라인)**", head), ("모멘텀 상위3, 절대필터 없음", no_abs),
                    ("**반전: 하위3** (순위를 뒤집음)", reversal),
                    ("유니버스 동일비중 존버", ew), ("BTC 존버", btc_hold_s), ("BTC SMA 5/20", sma)):
        base_rows.append(f"| {name} | **{pct(s['평균'])}** | {s['흑자']} | {pct(s['최악'])} |")

    # ---- 3. 무작위 선택 기준선 (전 구간, 같은 규칙·같은 비용) ----
    sl = slice(WARMUP - HEADLINE["lookback"] - 1, None)  # 첫 창 시작 시 모멘텀이 데워져 있게
    real = run_portfolio(opens.iloc[sl], closes.iloc[sl], fee=UPBIT_FEE, slippage=slip, **HEADLINE)
    rng = np.random.default_rng(args.seed)
    randoms = np.array([
        run_portfolio(opens.iloc[sl], closes.iloc[sl], fee=UPBIT_FEE, slippage=slip, **HEADLINE,
                      selector=random_selector(HEADLINE["top_k"], rng)).total_return
        for _ in range(args.n_random)
    ])
    percentile = float((randoms < real.total_return).mean() * 100)
    rev = run_portfolio(opens.iloc[sl], closes.iloc[sl], fee=UPBIT_FEE, slippage=slip, **HEADLINE,
                        selector=bottom_k_selector(HEADLINE["top_k"]))
    rev_percentile = float((randoms < rev.total_return).mean() * 100)
    print("3. 무작위 기준선 완료")

    picks = (real.weights > 0).mean().sort_values(ascending=False)
    picks_rows = ["| 종목 | 보유된 리밸런싱 비율 |", "|---|---:|"]
    for t, v in picks.head(8).items():
        picks_rows.append(f"| {t.replace('KRW-', '')} | {v:.0%} |")

    verdict_sel = ("우연으로 보기 어렵다 (상위 2.5%)" if percentile >= 97.5 else
                   "약한 증거 (상위 10%)" if percentile >= 90 else
                   "무작위보다 나쁘다" if percentile <= 10 else "무작위와 구분되지 않는다")
    stable = sum(1 for s in grid.values() if s["평균"] > ew["평균"])
    beats_sma = sum(1 for s in grid.values() if s["평균"] >= sma["평균"])
    best_cell = max(grid.items(), key=lambda kv: kv[1]["평균"])
    if percentile < 20 and rev_percentile > 60:
        ranking_note = "모멘텀이 아니라 **반전**이 있다 — 최근 강한 알트는 그 뒤 약하다."
    elif percentile < 20:
        ranking_note = ("**상위 선택이 특이하게 해롭다** — 최근 가장 강한 알트를 고르는 건 펌핑의 꼭지를 사는 "
                        "쪽에 가깝다. 그런데 뒤집어도(하위 선택) 무작위 수준이라 **이용 가능한 반전도 아니다.** "
                        "순위를 어느 방향으로 써도 돈이 안 된다.")
    else:
        ranking_note = "순위 자체에 뚜렷한 정보가 없다."

    REPORT.write_text(f"""# 조사: 크로스섹션 모멘텀 — 여러 종목 중 고르기

> 이 문서는 `scripts/12_research_momentum.py` 가 생성한다. 손으로 고치지 말 것.

생성: {datetime.now():%Y-%m-%d %H:%M} · 무작위 기준선 {args.n_random}개 · seed {args.seed}

## 질문

지금까지 본 건 전부 "BTC 를 **언제** 사고 팔까"(타이밍)였고, 그 엣지는 약했다(10).
학계에서 크립토에 가장 일관되게 확인된 현상은 타이밍이 아니라 **상대강도**다 —
여러 코인 중 최근 제일 강한 것들을 들고 정기적으로 갈아탄다.
구조가 다른 베팅이라 기존 전략과 합치면 분산 효과도 있다. 업비트에서도 통하나?

## 방법

| 항목 | 값 |
|---|---|
| 유니버스 | 원화마켓 거래대금 상위 120개 중 **{meta['listed_before']} 이전 상장** {n}개 ({', '.join(t.replace('KRW-', '') for t in tickers)}) |
| 기간 | {closes.index[0]:%Y-%m-%d} ~ {closes.index[-1]:%Y-%m-%d} (일봉, 2022 하락장 포함) |
| 규칙 | 매주 리밸런싱, 직전 lookback 일 수익률 상위 k 를 동일비중. **모멘텀 양수인 종목만**(절대 필터) — 다 음수면 현금 |
| 체결 | 리밸런싱 날 시가. 신호는 전날 종가까지 (미래참조 없음) |
| 비용 | 수수료 {UPBIT_FEE:.2%} + 종목별 슬리피지 = max(실측 0.02%, 호가단위/가격 ÷ 2). SHIB {slip['KRW-SHIB']:.2%}, DOGE {slip['KRW-DOGE']:.3%}, BTC {slip['KRW-BTC']:.2%} |
| 검증 | 앞 {WARMUP}일 워밍업 후 {SPLITS}개 검증창. 각 창 안에서 번 수익률만 센다 |
| 헤드라인 | lookback {HEADLINE['lookback']}일 · 상위 {HEADLINE['top_k']} · 주 1회 — **결과를 보기 전에** 교과서 기본값으로 고정 |

**왜 유니버스를 상장일로 걸렀나**: 처음엔 오늘 거래대금 상위 25개를 그대로 썼는데
상장 21일·22일·190일짜리 코인이 즐비했다. 오늘 거래대금이 큰 코인 = 최근 펌핑한
코인이라, 그걸로 5년을 돌리면 "지금 뜬 코인"을 미리 알고 고른 셈이 된다.

**그래도 남는 생존 편향**: 지금 상장돼 있는 코인만 뽑혔다. 5년 사이 상장폐지된
코인은 처음부터 없다. 모멘텀 전략은 폭락 직전 코인을 들고 있을 수 있으므로,
**아래 숫자는 실제보다 후하다.** 얼마나 후한지는 이 데이터로 알 수 없다.

---

## 1. 파라미터 격자 — 고르려는 게 아니라 안정성을 보려는 것

{chr(10).join(grid_rows)}

12칸 중 **{stable}칸**이 유니버스 동일비중 존버({pct(ew['평균'])})를 이긴다.
슬리피지를 2배로 줘도(오른쪽 열) 순서가 크게 안 바뀌면 비용에 강건한 것이다.

---

## 2. 헤드라인 vs 기준선 — 같은 검증창에서

{chr(10).join(base_rows)}

- **모멘텀 상위3 {pct(head['평균'])}** vs 유니버스 동일비중 존버 {pct(ew['평균'])} → {pct(head['평균'] - ew['평균'])}p
- 절대 필터를 빼면 {pct(no_abs['평균'])} (최악 {pct(no_abs['최악'])}). 절대 필터의 몫은 {pct(head['평균'] - no_abs['평균'])}p, 최악 구간 {(head['최악'] - no_abs['최악']) * 100:+.1f}%p
- BTC 하나만 SMA 5/20 으로 돌린 것({pct(sma['평균'])})과 비교하면 {pct(head['평균'] - sma['평균'])}p

---

## 3. 고르기에 실력이 있나 — 무작위 선택 기준선

같은 규칙(매주, 상위 {HEADLINE['top_k']}, 절대 필터, 같은 비용)에서 **후보 중 아무거나 {HEADLINE['top_k']}개**를
고르는 전략 {args.n_random}개를 돌렸다. 진짜(모멘텀 순위)가 그 분포의 어디에 있나.

| | 전 구간 수익률 |
|---|---:|
| 모멘텀 순위로 고름 (상위 3) | **{pct(real.total_return)}** |
| 순위를 뒤집음 (하위 3) | {pct(rev.total_return)} (백분위 {rev_percentile:.0f}) |
| 무작위 {args.n_random}개 평균 | {pct(float(randoms.mean()))} |
| 무작위 상위 5% | {pct(float(np.percentile(randoms, 95)))} |
| **백분위** | **{percentile:.0f}** → {verdict_sel} |

'절대 필터 + 매주 리밸런싱' 이라는 **구조**의 몫은 무작위 기준선에도 들어 있다.
백분위가 말하는 건 그 위에 **순위 매기기**가 얹은 몫이다.

### 실제로 뭘 들고 있었나 (헤드라인, 전 구간)

{chr(10).join(picks_rows)}

전 구간 성적: {real.summary()}

---

## 결론

1. **헤드라인(사전 고정)은 실패했다.** 창당 {pct(head['평균'])}, 최악 {pct(head['최악'])}.
   BTC 하나를 SMA 5/20 으로 돌린 것({pct(sma['평균'])})에 {pct(head['평균'] - sma['평균'])}p 뒤진다.
2. **순위 매기기의 실력**: 상위 3 은 백분위 {percentile:.0f}, 하위 3 은 {rev_percentile:.0f}. {ranking_note}
   단기(1~8주) 상대강도가 크립토에서 통한다는 논문들은 2014~2020 대형주 표본이다.
   2021 이후 업비트 알트에서는 그 반대에 가깝다.
2-1. 예외처럼 보이는 건 **6개월(180일) lookback** 뿐이다 — 장기 모멘텀은 단기와 성질이 다르다는
   문헌과 일치한다. 하지만 12칸 중 사후 최선이고 흑자 3/6 이라, 가설로 남겨두되 결론으로 쓰지 않는다.
3. **절대 필터가 일을 한다**: 빼면 최악 구간이 {pct(no_abs['최악'])} 로 깊어진다.
   하락장에 현금이 되는 구조는 타이밍 전략(10)에서 본 것과 같은 원천이다.
4. 이 결과는 **생존 편향으로 부풀려져 있다.** 상장폐지 코인이 없다.
   부풀려진 상태에서도 졌으므로, 실전은 더 나쁘다.
5. **알트 유니버스 자체가 5년간 마이너스다** (동일비중 존버 {pct(ew['평균'])}/창). 그 안에서 고르는
   문제는 "누가 덜 잃나"였다. 같은 기간 BTC 존버는 {pct(btc_hold_s['평균'])}.
   이 유니버스에서 얻을 수 있었던 가장 확실한 알파는 **알트를 안 사는 것**이었다.

## 한계

1. 생존 편향 (위). 이게 제일 크다.
2. 유니버스 {n}개는 작다. 논문들은 수백 개를 쓴다.
3. 슬리피지는 하한(반 틱)이다. 실측 DOGE 는 5배 나빴다. ×2 열을 같이 볼 것.
4. 5년 중 상승 비중이 크다. 절대 필터 덕에 하락장은 현금이라 그 구간은 사실상 안 재진 것과 같다.
5. 거래대금 기준 선정은 **오늘** 기준이다. 5년 전엔 거래대금 하위였을 수 있다 (일종의 룩어헤드).

## 재현

```bash
python scripts/fetch_universe.py      # 유니버스 갱신 (선택)
python scripts/12_research_momentum.py
```
""", encoding="utf-8")
    print(f"작성: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
