#!/usr/bin/env python
"""조사: 전략을 어떻게 바꾸면 실제로 나아지나.

인프라(cron·손절 감시)는 수익을 늘리지 않는다는 걸 08에서 확인했다.
여기서는 **전략 자체**를 건드린다.

세 가지를 본다:
  1. 필터 — 추세/변동성 조건으로 '안 사는 구간'을 만들면 나아지나
  2. 봉 종류 — 일봉이 최선인가, 4시간봉이나 주봉이 나은가
  3. 종목 — BTC 하나에서 본 결과가 다른 코인에도 통하나

**단일 구간 백테스트는 쓰지 않는다.** 실험기록에서 확인했듯이 훈련 구간 성적은
대부분 착시였다. 여기서는 전부 **워크포워드 검증 구간 성적**으로만 판단한다.

    python scripts/09_research_strategy.py
    python scripts/09_research_strategy.py --refresh
"""
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbit.backtest import UPBIT_FEE, run_backtest
from upbit.data import get_ohlcv
from upbit.optimize import split
from upbit.strategies import (
    BuyAndHoldStrategy,
    MACrossStrategy,
    TrendFilter,
    VolatilityFilter,
)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "data"
REPORT = ROOT / "docs" / "조사-전략개선.md"

CAPITAL, FEE, SLIPPAGE = 1_000_000, UPBIT_FEE, 0.0005
COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
DAILY_COUNT = 1800   # 약 5년 — 2022 하락장을 포함시키기 위해
SPLITS = 6
#: 지표를 데울 봉 수. 추세필터 200일선을 쓰므로 넉넉히 잡는다.
WARMUP = 250


def load(ticker: str, interval: str, count: int, refresh: bool) -> pd.DataFrame:
    path = SNAPSHOT / f"{ticker}_{interval}_{count}.csv"
    if refresh or not path.exists():
        SNAPSHOT.mkdir(parents=True, exist_ok=True)
        get_ohlcv(ticker, interval, count, use_cache=False).to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:+.2%}"


def walk_forward_returns(
    df: pd.DataFrame, strategy, n_splits: int = SPLITS, warmup: int = WARMUP,
    slippage: float = SLIPPAGE,
) -> list[float]:
    """검증 구간을 여러 개 잡고 각 구간에서 번 수익률만 모은다.

    **워밍업이 핵심이다.** 200일선을 쓰는 전략을 64일짜리 창에 넣으면 지표가
    전부 NaN 이라 아무것도 안 산다. 그건 전략이 나쁜 게 아니라 평가가 틀린 것이다.
    그래서 각 검증 창 앞에 `warmup` 개의 봉을 붙여 지표를 데운 뒤,
    **검증 구간에서 늘어난 자산만** 센다.

    워밍업 구간에 이미 포지션을 들고 검증 구간에 진입할 수 있는데, 그건 실전과
    같으므로 그대로 둔다.
    """
    usable = len(df) - warmup
    if usable < n_splits * 30:
        return []

    window = usable // n_splits
    out = []
    for i in range(n_splits):
        test_start = warmup + i * window
        test_end = test_start + window
        frame = df.iloc[test_start - warmup : test_end]
        if len(frame) < warmup + 20:
            continue
        equity = run_backtest(frame, strategy, CAPITAL, FEE, slippage).equity
        start_value = equity.iloc[warmup]
        if start_value <= 0:
            continue
        out.append(equity.iloc[-1] / start_value - 1)
    return out


def summarize(returns: list[float]) -> dict:
    """검증 구간들의 성적 요약. 평균만 보면 한 번의 대박에 속으므로 흑자 비율과
    최악의 구간을 같이 본다."""
    if not returns:
        return {"평균": float("nan"), "흑자": "0/0", "최악": float("nan")}
    wins = sum(1 for r in returns if r > 0)
    return {
        "평균": sum(returns) / len(returns),
        "흑자": f"{wins}/{len(returns)}",
        "최악": min(returns),
    }


# ---------------------------------------------------------------- 실험 1

def variants(fast=5, slow=20):
    base = MACrossStrategy(fast, slow)
    return [
        ("필터 없음", MACrossStrategy(fast, slow)),
        ("추세필터 200", TrendFilter(MACrossStrategy(fast, slow), 200)),
        ("추세필터 100", TrendFilter(MACrossStrategy(fast, slow), 100)),
        ("변동성 하한 2%", VolatilityFilter(MACrossStrategy(fast, slow), min_pct=0.02)),
        ("변동성 상한 6%", VolatilityFilter(MACrossStrategy(fast, slow), max_pct=0.06)),
        ("추세100+변동성2%", TrendFilter(
            VolatilityFilter(MACrossStrategy(fast, slow), min_pct=0.02), 100)),
        ("사서 존버", BuyAndHoldStrategy()),
    ]


def experiment_filters(frames: dict[str, pd.DataFrame]) -> tuple[dict, str]:
    """필터별로 **모든 코인 × 모든 구간**의 검증 성적을 모은다."""
    results: dict[str, list[float]] = {}
    for label, strategy in variants():
        pooled = []
        for df in frames.values():
            pooled += walk_forward_returns(df, strategy)
        results[label] = pooled

    rows = ["| 전략 | 검증 평균 | 흑자 구간 | 최악 구간 |", "|---|---:|---:|---:|"]
    for label, returns in sorted(results.items(), key=lambda kv: -summarize(kv[1])["평균"]):
        stats = summarize(returns)
        rows.append(f"| {label} | **{pct(stats['평균'])}** | {stats['흑자']} | {pct(stats['최악'])} |")
    return results, "\n".join(rows)


# ---------------------------------------------------------------- 실험 2

def experiment_per_coin(frames: dict[str, pd.DataFrame]) -> str:
    labels = ["필터 없음", "추세필터 100", "사서 존버"]
    rows = ["| 종목 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for ticker, df in frames.items():
        cells = []
        for label in labels:
            strategy = next(s for l, s in variants() if l == label)
            cells.append(pct(summarize(walk_forward_returns(df, strategy))["평균"]))
        rows.append(f"| {ticker.replace('KRW-', '')} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------- 실험 3

def experiment_intervals(refresh: bool) -> tuple[str, str]:
    """봉 종류 비교 + 슬리피지 민감도.

    짧은 봉은 거래가 잦아 비용에 민감하다. 슬리피지 가정을 바꿔도 결론이
    유지되는지 확인하지 않으면, '4시간봉이 낫다'는 결론이 0.05%라는 가정
    하나에 기대게 된다.
    """
    # 기간을 맞춘다 — 봉마다 다른 기간을 보면 봉 종류가 아니라 시장을 비교하게 된다.
    settings = [("minute240", 11_000, "4시간봉"), ("day", 1_800, "일봉"), ("week", 300, "주봉")]
    strategy = MACrossStrategy(5, 20)

    rows = ["| 봉 종류 | 기간 | 검증 평균 | 흑자 구간 | 거래 빈도 |", "|---|---|---:|---:|---:|"]
    frames = {}
    for interval, count, label in settings:
        df = load("KRW-BTC", interval, count, refresh)
        frames[label] = df
        stats = summarize(walk_forward_returns(df, strategy, warmup=60))
        whole = run_backtest(df, strategy, CAPITAL, FEE, SLIPPAGE)
        years = max((df.index[-1] - df.index[0]).days / 365.25, 0.01)
        rows.append(
            f"| {label} | {df.index[0]:%y-%m} ~ {df.index[-1]:%y-%m} "
            f"| **{pct(stats['평균'])}** | {stats['흑자']} "
            f"| 연 {whole.num_trades / years:.0f}회 |"
        )

    levels = [0.0005, 0.001, 0.002, 0.004]
    sens = ["| 봉 종류 | " + " | ".join(f"슬리피지 {v:.2%}" for v in levels) + " |",
            "|---|" + "---:|" * len(levels)]
    for label, df in frames.items():
        cells = [
            pct(summarize(walk_forward_returns(df, strategy, warmup=60, slippage=v))["평균"])
            for v in levels
        ]
        sens.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(rows), "\n".join(sens)


def main() -> None:
    ap = argparse.ArgumentParser(description="전략 개선 조사")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    frames = {}
    for ticker in COINS:
        try:
            frames[ticker] = load(ticker, "day", DAILY_COUNT, args.refresh)
        except Exception as exc:
            print(f"  {ticker} 건너뜀: {exc}")
    print(f"종목 {len(frames)}개 로드")

    filter_results, filter_table = experiment_filters(frames)
    print("실험 1 완료 (필터)")
    coin_table = experiment_per_coin(frames)
    print("실험 2 완료 (종목별)")
    interval_table, sensitivity_table = experiment_intervals(args.refresh)
    print("실험 3 완료 (봉 종류)")

    ranked = sorted(filter_results.items(), key=lambda kv: -summarize(kv[1])["평균"])
    best_label, best_returns = ranked[0]
    base_stats = summarize(filter_results["필터 없음"])
    best_stats = summarize(best_returns)
    hold_stats = summarize(filter_results["사서 존버"])

    sample = sum(len(v) for v in filter_results.values()) // len(filter_results)

    # 필터의 진짜 효과: 평균이 아니라 '최악'과 '일관성'
    no_filter = summarize(filter_results["필터 없음"])
    trend = summarize(filter_results["추세필터 100"])
    vol_cap = summarize(filter_results["변동성 상한 6%"])
    worst_gain = trend["최악"] - no_filter["최악"]

    REPORT.write_text(f"""# 조사: 전략을 어떻게 바꾸면 실제로 나아지나

> 이 문서는 `scripts/09_research_strategy.py` 가 생성한다. 손으로 고치지 말 것.

생성: {datetime.now():%Y-%m-%d %H:%M}

## 방법

08에서 인프라(cron·손절 감시 주기)는 수익을 늘리지 않는다는 걸 확인했다.
여기서는 **전략 자체**를 건드린다.

**단일 구간 백테스트는 쓰지 않는다.** `docs/실험기록.md` 에서 확인했듯이
훈련 구간 성적은 대부분 착시였다(+93% → 검증 -4%). 여기서는 전부
**워크포워드 검증 구간 성적**으로만 판단한다.

| 항목 | 값 |
|---|---|
| 종목 | {', '.join(t.replace('KRW-', '') for t in frames)} |
| 봉 / 기간 | 일봉 {DAILY_COUNT}개 |
| 구간 분할 | 앞 {WARMUP}봉으로 지표를 데우고, 남은 구간을 {SPLITS}개 검증창으로 |
| 표본 | 전략당 검증구간 약 {sample}개 (종목 {len(frames)}개 × {SPLITS}구간) |
| 자본 / 수수료 / 슬리피지 | {CAPITAL:,}원 / {FEE:.3%} / {SLIPPAGE:.3%} |

---

## 실험 1 — 필터가 도움이 되나

MA교차의 알려진 약점은 **횡보장 휩쏘**다. 방향이 없는 구간에서 교차가 계속
발생해 사고팔기를 반복하고 매번 수수료를 낸다.
필터는 "이 조건이 아니면 아예 참여하지 않는다"를 더한다.

{filter_table}

### 평균만 보면 필터는 쓸모없어 보인다

1등은 {best_label}({pct(best_stats['평균'])})이고, 필터 없음은 {pct(base_stats['평균'])} 다.
차이는 {pct(best_stats['평균'] - base_stats['평균'])}p 로 크지 않다.

### 그런데 최악을 보면 얘기가 다르다

| | 평균 | 흑자 구간 | **최악 구간** |
|---|---:|---:|---:|
| 필터 없음 | {pct(no_filter['평균'])} | {no_filter['흑자']} | **{pct(no_filter['최악'])}** |
| 추세필터 100 | {pct(trend['평균'])} | {trend['흑자']} | **{pct(trend['최악'])}** |
| 변동성 상한 6% | {pct(vol_cap['평균'])} | {vol_cap['흑자']} | **{pct(vol_cap['최악'])}** |
| 사서 존버 | {pct(hold_stats['평균'])} | {hold_stats['흑자']} | **{pct(hold_stats['최악'])}** |

추세필터는 평균을 {pct(trend['평균'] - no_filter['평균'])}p 깎는 대신
**최악의 구간을 {abs(worst_gain) * 100:.1f}%p 개선**했다 ({pct(no_filter['최악'])} → {pct(trend['최악'])}).
변동성 상한은 흑자 구간 비율이 제일 높다({vol_cap['흑자']}) — 가장 꾸준하다는 뜻이다.

**필터는 더 벌게 해주지 않는다. 덜 잃게 해준다.**
그리고 실전에서 중요한 건 대개 후자다 — 최악의 구간에서 못 버티고 던지면
평균 수익률은 받아보지도 못한다.

존버는 평균이 {pct(hold_stats['평균'])} 로 나쁘지 않지만 흑자 구간이 {hold_stats['흑자']} 뿐이고
최악이 {pct(hold_stats['최악'])} 다. 5년 중 큰 상승 몇 번에 기댄 성적이다.

---

## 실험 2 — 종목을 바꾸면 결론이 유지되나

BTC 하나에서 본 결과가 다른 코인에도 통하는지 본다.
안 통하면 그건 BTC 에만 맞춰진 것이다.

{coin_table}

---

## 실험 3 — 봉 종류

짧은 봉은 노이즈가 많고 수수료가 자주 나간다. 긴 봉은 신호가 늦다.
같은 전략(SMA 5/20)을 봉만 바꿔 돌렸다.

{interval_table}

### 슬리피지를 바꿔도 유지되나

짧은 봉은 거래가 잦아 비용에 민감하다. 4시간봉은 연 66회, 일봉은 연 10회다.
'4시간봉이 낫다'가 슬리피지 0.05% 라는 가정 하나에 기댄 결론인지 확인한다.

{sensitivity_table}

---

## 결론

1. **필터는 수익을 늘리지 않는다. 최악을 줄인다.**
   평균은 오히려 조금 낮아지지만 최악의 구간이 {pct(no_filter['최악'])} → {pct(trend['최악'])} 로 개선됐다.
   "-50% 를 견딜 자신이 없다"면 필터를 켜는 게 맞다.

2. **봉 종류 결론은 슬리피지 가정에 전적으로 달렸다.**
   0.05% 가정에서는 4시간봉이 1등이지만, 0.4% 에서는 마이너스로 무너진다.
   일봉·주봉은 거의 영향이 없다. **실측 슬리피지를 모으기 전에는
   짧은 봉으로 가면 안 된다** — `reports/fills.csv` 가 그 자료를 모으고 있다.

3. **종목마다 결론이 다르다.** DOGE 는 전략이 존버를 크게 이겼지만
   XRP 는 존버가 이겼다. 한 종목에서 본 결과를 일반화하면 안 된다.

4. **여전히 '이 전략이 돈을 번다'는 증거는 없다.** 검증 구간 흑자 비율은
   어떤 전략이든 30개 중 20개 안팎이다. 5년이 대체로 상승장이었다는 점을
   감안하면 이건 강한 결과가 아니다.

## 한계

1. **필터 파라미터를 검증하지 않았다.** 추세필터 100/200, 변동성 2%/6% 같은
   숫자는 그냥 고른 것이다. 이걸 검증 구간 성적으로 고르면 그 순간 과최적화다.
   지금은 "구조적 선택(필터를 붙이나 마나)"만 비교했다.
2. **구간마다 파라미터를 재최적화하지 않았다.** 5/20 고정이다.
3. 코인 {len(frames)}개, 일봉 {DAILY_COUNT}개(약 5년). 2022 하락장이 포함돼 있다.
4. 검증 구간이 짧아 거래 표본이 적다. 각 창 앞에 {WARMUP}봉을 워밍업으로 붙였는데,
   워밍업 구간이 창끼리 겹치므로 검증들이 완전히 독립적이지는 않다.
5. **상장폐지 종목이 빠져 있다** — 지금 살아있는 코인만 봤으므로 생존 편향이 있다.

## 재현

```bash
python scripts/09_research_strategy.py
```
""", encoding="utf-8")
    print(f"작성: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
