#!/usr/bin/env python
"""조사: 실행 타이밍이 수익에 얼마나 영향을 주나.

두 가지를 실제로 측정한다.

  실험 A — **제때 실행하는 것의 가치** (cron/자동화)
    신호를 k봉 늦게 실행하면 얼마를 잃나. 봇이 하루 쉬면?

  실험 B — **손절을 자주 확인하는 것의 가치** (감시 주기 분리)
    일봉 전략의 손절을 하루 1회 vs 1시간마다 확인하면 얼마나 달라지나.
    지금 실전 봇은 하루 1회다.

    python scripts/08_research_timing.py
    python scripts/08_research_timing.py --refresh
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
from upbit.multiframe import StopCheck, run_multiframe_backtest
from upbit.risk import RiskRules
from upbit.strategies import MACrossStrategy
from upbit.strategies.base import Strategy

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "data"
REPORT = ROOT / "docs" / "조사-실행타이밍.md"

TICKER = "KRW-BTC"
CAPITAL, FEE, SLIPPAGE = 1_000_000, UPBIT_FEE, 0.0005
DAILY_COUNT, HOURLY_COUNT = 800, 20_000


class Delayed(Strategy):
    """전략 신호를 k봉 늦게 내보낸다 — 봇이 제때 안 돌았을 때를 흉내낸다."""

    def __init__(self, inner: Strategy, bars: int):
        self.inner = inner
        self.bars = bars
        self.name = f"{inner.name} · {bars}봉 지연" if bars else inner.name

    def generate_positions(self, df):
        return self.inner.generate_positions(df).shift(self.bars).fillna(0).astype(int)


def load(interval: str, count: int, refresh: bool) -> tuple[pd.DataFrame, str]:
    path = SNAPSHOT / f"{TICKER}_{interval}_{count}.csv"
    if refresh or not path.exists():
        SNAPSHOT.mkdir(parents=True, exist_ok=True)
        get_ohlcv(TICKER, interval, count, use_cache=False).to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df, hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:+.2%}"


def num(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.2f}"


# ---------------------------------------------------------------- 실험 A

def experiment_delay(daily: pd.DataFrame) -> tuple[list, str]:
    base = MACrossStrategy(5, 20)
    rows, results = [], []
    for bars in (0, 1, 2, 3, 5):
        result = run_backtest(daily, Delayed(base, bars), CAPITAL, FEE, SLIPPAGE)
        results.append((bars, result))
        label = "제때 실행" if bars == 0 else f"{bars}일 지연"
        rows.append(
            f"| {label} | {pct(result.total_return)} | {pct(result.mdd)} "
            f"| {num(result.sharpe)} | {pct(result.win_rate)} | {result.num_trades} |"
        )
    table = "\n".join(
        ["| 실행 시점 | 총수익률 | MDD | 샤프 | 승률 | 거래수 |",
         "|---|---:|---:|---:|---:|---:|"] + rows
    )
    return results, table


# ---------------------------------------------------------------- 실험 B

STOP_SETTINGS = [
    ("ATR×2", RiskRules(atr_multiple=2.0)),
    ("ATR×2 + 추적", RiskRules(atr_multiple=2.0, trailing=True)),
    ("고정 -5%", RiskRules(stop_loss_pct=0.05)),
]

CHECK_MODES = [
    ("하루 1회 (지금 실전)", StopCheck(at_hour=9)),
    ("4시간마다", StopCheck(every=4)),
    ("1시간마다", StopCheck(every=1)),
]


def experiment_stop_frequency(daily: pd.DataFrame, hourly: pd.DataFrame) -> tuple[dict, str]:
    strategy = MACrossStrategy(5, 20)
    outcomes: dict = {}

    no_stop = run_multiframe_backtest(
        daily, hourly, strategy, risk=None,
        initial_capital=CAPITAL, fee=FEE, slippage=SLIPPAGE)
    outcomes[("손절 없음", "—")] = no_stop

    for stop_label, rules in STOP_SETTINGS:
        for check_label, checker in CHECK_MODES:
            outcomes[(stop_label, check_label)] = run_multiframe_backtest(
                daily, hourly, strategy, risk=rules, stop_check=checker,
                initial_capital=CAPITAL, fee=FEE, slippage=SLIPPAGE)

    header = "| 손절 설정 | " + " | ".join(label for label, _ in CHECK_MODES) + " |"
    divider = "|---|" + "---:|" * len(CHECK_MODES)
    rows = [f"| **손절 없음** | " + " | ".join([pct(no_stop.total_return)] * len(CHECK_MODES)) + " |"]
    for stop_label, _ in STOP_SETTINGS:
        cells = []
        for check_label, _ in CHECK_MODES:
            r = outcomes[(stop_label, check_label)]
            stopped = sum(1 for t in r.closed_trades if t.exit_reason == "손절")
            cells.append(f"{pct(r.total_return)}<br><sub>MDD {pct(r.mdd)} · 손절 {stopped}회</sub>")
        rows.append(f"| **{stop_label}** | " + " | ".join(cells) + " |")

    return outcomes, "\n".join([header, divider] + rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="실행 타이밍이 수익에 미치는 영향 조사")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    daily, daily_hash = load("day", DAILY_COUNT, args.refresh)
    hourly, hourly_hash = load("minute60", HOURLY_COUNT, args.refresh)

    # 두 데이터의 겹치는 구간만 쓴다 — 비교가 성립하려면 같은 기간이어야 한다
    start = max(daily.index[0], hourly.index[0])
    daily = daily[daily.index >= start]
    hourly = hourly[hourly.index >= start]
    print(f"공통 구간: {start:%Y-%m-%d} ~ {hourly.index[-1]:%Y-%m-%d} "
          f"(일봉 {len(daily)}개 / 60분봉 {len(hourly)}개)")

    delay_results, delay_table = experiment_delay(daily)
    print("실험 A 완료")
    stop_outcomes, stop_table = experiment_stop_frequency(daily, hourly)
    print("실험 B 완료")

    on_time = delay_results[0][1]
    one_day_late = delay_results[1][1]

    def cell(stop_label, check_label):
        return stop_outcomes[(stop_label, check_label)]

    best_gain, best_desc = -9e9, ""
    return_gains, mdd_improved, mdd_lines = [], 0, []
    for stop_label, _ in STOP_SETTINGS:
        daily_check = cell(stop_label, "하루 1회 (지금 실전)")
        hourly_check = cell(stop_label, "1시간마다")
        gain = hourly_check.total_return - daily_check.total_return
        return_gains.append(gain)
        if gain > best_gain:
            best_gain, best_desc = gain, stop_label

        mdd_delta = hourly_check.mdd - daily_check.mdd  # 양수면 낙폭이 얕아짐
        if mdd_delta > 0:
            mdd_improved += 1
        mdd_lines.append(
            f"| {stop_label} | {pct(daily_check.mdd)} | {pct(hourly_check.mdd)} "
            f"| {'개선' if mdd_delta > 0 else '악화'} {abs(mdd_delta) * 100:.1f}%p "
            f"| {pct(gain)}p |"
        )

    mdd_table = "\n".join(
        ["| 손절 설정 | MDD (하루 1회) | MDD (1시간마다) | 낙폭 변화 | 수익 변화 |",
         "|---|---:|---:|---:|---:|"] + mdd_lines
    )
    worse_count = sum(1 for g in return_gains if g < 0)

    REPORT.write_text(f"""# 조사: 실행 타이밍이 수익에 얼마나 영향을 주나

> 이 문서는 `scripts/08_research_timing.py` 가 생성한다. 손으로 고치지 말 것.

생성: {datetime.now():%Y-%m-%d %H:%M}

## 왜 조사했나

실전 준비 작업이 두 갈래 남아 있었다.

1. **cron/launchd 등록** — 봇이 봉 마감마다 자동으로 돌게
2. **남은 한계 처리** — 백오프 재시도, **손절 감시 주기 분리**, 시계 동기화

"어느 쪽이 돈이 되나"를 추측 대신 측정으로 답한다.

먼저 분명히 할 것: **둘 다 수익을 늘리는 작업이 아니다.** 전략은 그대로고,
같은 전략을 얼마나 정확하게 실행하느냐의 문제다. 즉 **손실을 줄이는 작업**이다.

## 실험 조건

| 항목 | 값 |
|---|---|
| 종목 | `{TICKER}` |
| 전략 | SMA교차 5/20 (일봉 신호) |
| 기간 | {daily.index[0]:%Y-%m-%d} ~ {daily.index[-1]:%Y-%m-%d} ({len(daily)}일) |
| 체결 감시 | 60분봉 {len(hourly):,}개 |
| 자본 / 수수료 / 슬리피지 | {CAPITAL:,}원 / {FEE:.3%} / {SLIPPAGE:.3%} |
| 데이터 | 일봉 `{daily_hash}` · 60분봉 `{hourly_hash}` |

---

## 실험 A — 제때 실행하는 것의 가치 (cron)

신호를 k일 늦게 실행하면 얼마를 잃나. 봇이 안 돌았거나, 오류로 주기를 건너뛰었거나,
봉 마감과 무관한 시각에 도는 상황이다.

{delay_table}

**하루만 늦어도 {pct(one_day_late.total_return - on_time.total_return)}p.**
{pct(on_time.total_return)} → {pct(one_day_late.total_return)} 로 떨어진다.

추세추종 전략은 방향이 바뀌는 그 시점에 들어가고 나오는 게 전부다.
하루 늦으면 그 하루의 움직임을 통째로 놓친다. 5일 지연은
{pct(delay_results[-1][1].total_return)} — 전략이 사실상 무너진다.

> ⚠️ 다만 이건 **매번 늦는** 경우다. cron 을 안 걸어도 사람이 매일 같은 시각에
> 직접 돌리면 지연은 0이다. cron 의 진짜 가치는 "잊지 않는 것"이지
> "빠른 것"이 아니다.

---

## 실험 B — 손절을 자주 확인하는 것의 가치

지금 실전 봇은 **하루 한 번**(봉 마감 09:00)만 깨어나 손절선을 확인한다.
장중에 급락하면 다음 날까지 모른다. 손절 감시만 따로 자주 돌리면 얼마나 달라지나.

{stop_table}

### 읽는 법

**손절 확인을 자주 한다고 수익이 늘지 않는다.**
손절 설정 {len(STOP_SETTINGS)}개 중 {worse_count}개에서 수익이 **줄었고**,
제일 좋은 경우({best_desc})조차 **{pct(best_gain)}p** 에 그친다.

이유: 자주 확인할수록 **장중에 스쳤다가 회복하는 급락**에 더 자주 털린다.
하루 1회 확인은 그런 흔들림을 못 보고 지나가서 오히려 살아남는다.
손절 발동 횟수를 보면 하루 1회 대비 1시간마다가 눈에 띄게 많다.

낙폭은 어떤가 — **일관되게 좋아지지도 않았다** ({len(STOP_SETTINGS)}개 중 {mdd_improved}개만 개선):

{mdd_table}

추적손절에서만 낙폭이 뚜렷이 개선된다. 추적손절은 손절선이 계속 올라가므로
자주 확인할수록 더 촘촘히 따라붙기 때문이다. 나머지는 손절선이 고정이라
자주 본다고 더 잘 잡히지 않고, 오히려 회복할 급락에 털려 낙폭이 나빠지기도 한다.

---

## 결론

| 작업 | 수익 영향 | 판단 |
|---|---|---|
| **cron/launchd 등록** | 매번 하루씩 늦는 것을 막으면 **{pct(on_time.total_return - one_day_late.total_return)}p** | **이쪽이 훨씬 크다** |
| 손절 감시 주기 분리 | 최선이 **{pct(best_gain)}p**, {worse_count}/{len(STOP_SETTINGS)}은 마이너스 | 수익 관점에선 할 이유가 없다 |

**cron 을 먼저 걸어야 한다.** 다만 그 이유는 "빨라서"가 아니라
**실행을 빠뜨리지 않기 위해서**다. 지연 비용이 큰 이유도 같다 — 하루를 통째로
놓치는 게 비싸지, 몇 시간 빠른 게 이득인 게 아니다.

손절 감시 분리는 **수익이 아니라 낙폭을 위해**, 그것도 **추적손절을 쓸 때만**
의미가 있다. 고정 손절선이면 자주 봐도 낙폭이 나아지지 않았다.
수익을 늘리려는 목적이라면 근거가 없다.

### 그리고 더 중요한 것

두 작업 다 **전략을 건드리지 않는다.** 실행을 아무리 정확히 해도
`docs/실험기록.md` 의 워크포워드 결과(구간마다 최적 파라미터가 다르고 검증 성적이
들쭉날쭉)는 그대로다. **수익을 진짜로 늘리려면 인프라가 아니라 전략을 봐야 한다.**

## 한계

1. 종목 1개(BTC), 기간 1개, 전략 1개(SMA 5/20). 다른 조합이면 뒤집힐 수 있다.
2. 60분봉 안에서의 움직임은 여전히 못 본다 — 1시간마다 확인도 근사치다.
3. 지연 실험은 '매번 k일 늦음'을 가정한다. 실제로는 가끔 건너뛰는 형태에 가깝다.
4. 거래 표본이 20건 안팎이라 통계적으로 강한 주장은 어렵다.

## 재현

```bash
python scripts/08_research_timing.py
```
""", encoding="utf-8")
    print(f"작성: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
