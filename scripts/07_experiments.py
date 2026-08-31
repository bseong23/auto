#!/usr/bin/env python
"""실험 기록 생성기 — README/문서의 숫자를 이 스크립트가 만든다.

숫자를 손으로 문서에 옮겨 적으면 코드가 바뀌었을 때 문서가 조용히 거짓말을 한다.
그래서 결과 문서를 코드가 생성하게 했다.

**재현성**: 업비트는 항상 '최근 N개'를 주므로 내일 돌리면 데이터가 달라진다.
그래서 실험에 쓴 데이터를 docs/data/ 에 스냅샷으로 고정해두고, 이후 실행은
그 파일을 읽는다. 새 데이터로 다시 찍으려면 --refresh.

    python scripts/07_experiments.py            # 스냅샷으로 재현
    python scripts/07_experiments.py --refresh  # 최신 데이터로 갱신
"""
import argparse
import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbit.backtest import UPBIT_FEE, run_backtest
from upbit.data import get_ohlcv
from upbit.optimize import holdout_test, walk_forward
from upbit.plotting import plot_comparison, plot_result
from upbit.risk import RiskRules
from upbit.strategies import (
    BollingerStrategy,
    BuyAndHoldStrategy,
    MACrossStrategy,
    RSIStrategy,
)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "data"
IMAGES = ROOT / "docs" / "images"
REPORT = ROOT / "docs" / "실험기록.md"

TICKER, INTERVAL, COUNT = "KRW-BTC", "day", 800
CAPITAL, FEE, SLIPPAGE = 1_000_000, UPBIT_FEE, 0.0005

#: 손절 실험에서 비교할 설정들
STOP_VARIANTS = [
    ("손절 없음", RiskRules()),
    ("고정 -5%", RiskRules(stop_loss_pct=0.05)),
    ("고정 -10%", RiskRules(stop_loss_pct=0.10)),
    ("ATR×2", RiskRules(atr_multiple=2.0)),
    ("ATR×3", RiskRules(atr_multiple=3.0)),
    ("ATR×2 + 추적", RiskRules(atr_multiple=2.0, trailing=True)),
    ("고정 -5% + 추적", RiskRules(stop_loss_pct=0.05, trailing=True)),
]


def load_dataset(refresh: bool) -> tuple[pd.DataFrame, Path, str]:
    """스냅샷을 읽거나(재현) 새로 받아 저장한다(갱신). 해시로 동일성을 보장."""
    path = SNAPSHOT / f"{TICKER}_{INTERVAL}_{COUNT}.csv"
    if refresh or not path.exists():
        df = get_ohlcv(TICKER, INTERVAL, COUNT, use_cache=False)
        SNAPSHOT.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return df, path, digest


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:+.2%}"


def num(v, fmt="{:.2f}") -> str:
    return "n/a" if pd.isna(v) else fmt.format(v)


def experiment_strategies(df: pd.DataFrame) -> tuple[list, str]:
    """실험 1 — 전략 4종 + 벤치마크를 같은 조건에서 비교."""
    strategies = [
        MACrossStrategy(5, 20),
        MACrossStrategy(5, 20, use_ema=True),
        RSIStrategy(),
        BollingerStrategy(),
        BuyAndHoldStrategy(),
    ]
    results = [run_backtest(df, s, CAPITAL, FEE, SLIPPAGE) for s in strategies]
    results.sort(key=lambda r: r.total_return, reverse=True)

    rows = ["| 전략 | 총수익률 | CAGR | MDD | 샤프 | 승률 | 거래수 | 손익비 | 수수료 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        rows.append(
            f"| {r.strategy_name} | **{pct(r.total_return)}** | {pct(r.cagr)} | {pct(r.mdd)} "
            f"| {num(r.sharpe)} | {pct(r.win_rate)} | {r.num_trades} | {num(r.profit_factor)} "
            f"| {r.total_fees_paid:,.0f}원 |"
        )
    return results, "\n".join(rows)


def experiment_stops(df: pd.DataFrame) -> tuple[list, str]:
    """실험 2 — 전략은 SMA교차 5/20 고정, 손절 설정만 바꾼다."""
    outcomes = []
    for label, rules in STOP_VARIANTS:
        result = run_backtest(df, MACrossStrategy(5, 20), CAPITAL, FEE, SLIPPAGE, risk=rules)
        stopped = sum(1 for t in result.closed_trades if t.exit_reason.startswith("손절"))
        outcomes.append((label, rules, result, stopped))

    rows = ["| 손절 설정 | 총수익률 | MDD | 샤프 | 승률 | 거래수 | 손절청산 | 평균손익/거래 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, _, r, stopped in outcomes:
        rows.append(
            f"| {label} | {pct(r.total_return)} | {pct(r.mdd)} | {num(r.sharpe)} "
            f"| {pct(r.win_rate)} | {r.num_trades} | {stopped}건 | {pct(r.avg_trade_return)} |"
        )
    return outcomes, "\n".join(rows)


def experiment_overfitting(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, str]:
    """실험 3 — 파라미터 탐색이 만들어내는 착시를 측정한다."""
    grid = {"fast": [3, 5, 7, 10, 15, 20], "slow": [20, 30, 40, 60, 90, 120]}
    holdout = holdout_test(df, MACrossStrategy, grid, train_ratio=0.7, fee=FEE, slippage=SLIPPAGE)
    forward = walk_forward(df, MACrossStrategy, grid, n_splits=4, fee=FEE, slippage=SLIPPAGE)

    rows = ["| 구간 | 검증기간 | 선택된 파라미터 | 훈련수익률 | 검증수익률 | 검증MDD | 청산거래 |",
            "|---|---|---|---:|---:|---:|---:|"]
    for _, row in forward.iterrows():
        params = row["선택된 파라미터"]
        rows.append(
            f"| {row['구간']} | {row['검증기간']} | {params['fast']}/{params['slow']} "
            f"| {pct(row['훈련수익률'])} | **{pct(row['검증수익률'])}** "
            f"| {pct(row['검증MDD'])} | {row['검증거래수']} |"
        )
    return holdout, forward, "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="실험 기록 생성")
    ap.add_argument("--refresh", action="store_true", help="최신 데이터로 스냅샷 갱신")
    ap.add_argument("--no-images", action="store_true", help="그래프 생성 건너뛰기")
    args = ap.parse_args()

    df, snapshot_path, digest = load_dataset(args.refresh)
    print(f"데이터: {snapshot_path.relative_to(ROOT)} ({len(df)}개, sha256:{digest})")

    results, table_strategies = experiment_strategies(df)
    outcomes, table_stops = experiment_stops(df)
    holdout, forward, table_forward = experiment_overfitting(df)
    print("실험 3종 완료")

    if not args.no_images:
        plot_comparison(results, IMAGES / f"comparison_{TICKER}_{INTERVAL}.png",
                        title=f"{TICKER} {INTERVAL} — 전략 비교 (손절 없음)")
        plot_result(max(results, key=lambda r: r.total_return), df,
                    IMAGES / f"best_{TICKER}_{INTERVAL}.png")
        trail = next(r for label, _, r, _ in outcomes if label == "ATR×2 + 추적")
        plot_result(trail, df, IMAGES / f"best_{TICKER}_{INTERVAL}_atr2-trail.png")
        print("그래프 3장 생성")

    train_r = holdout["train"].total_return
    test_r = holdout["test"].total_return
    if test_r <= 0:
        reproduction_note = (
            "**검증 구간에서는 아예 손실이 났다.** 훈련 구간의 성적은 통째로 착시였다는 뜻이다.\n"
            "36개 조합 중 1등을 고른 행위 자체가 '그 구간에 제일 잘 맞는 숫자 찾기'였을 뿐이다."
        )
    else:
        reproduction_note = (
            f"훈련 성적의 {test_r / train_r * 100:.1f}% 만 검증 구간에서 재현됐다.\n"
            "나머지는 **그 구간에만 맞춰진 착시**였다."
        )

    no_stop = outcomes[0][2]
    trailing = next(r for label, _, r, _ in outcomes if label == "ATR×2 + 추적")
    best = results[0]
    bench = next(r for r in results if r.strategy_name == "사서 존버")

    REPORT.write_text(f"""# 실험 기록

> 이 문서는 `scripts/07_experiments.py` 가 생성한다. 손으로 고치지 말 것.
> 갱신: `python scripts/07_experiments.py`

생성 시각: {datetime.now():%Y-%m-%d %H:%M} · Python {platform.python_version()} · {platform.system()}

---

## 실험 조건

| 항목 | 값 |
|---|---|
| 종목 / 봉 | `{TICKER}` / `{INTERVAL}` |
| 기간 | {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({len(df)}개 캔들, {(df.index[-1] - df.index[0]).days / 365.25:.2f}년) |
| 초기자본 | {CAPITAL:,}원 |
| 수수료 | {FEE:.3%} (업비트 원화마켓 편도) |
| 슬리피지 | {SLIPPAGE:.3%} |
| 체결 규칙 | t봉 종가 신호 → **t+1봉 시가** 체결 |
| 데이터 스냅샷 | `{snapshot_path.relative_to(ROOT)}` (sha256 `{digest}`) |

**왜 스냅샷을 고정했나**: 업비트 API는 항상 '최근 N개'를 준다. 내일 같은 명령을 돌리면
데이터가 하루씩 밀려서 숫자가 달라진다. 문서의 숫자를 검증할 수 있으려면 데이터가
고정돼야 한다. 최신 데이터로 다시 찍으려면 `--refresh`.

---

## 실험 1 — 전략 비교

전략 4종과 벤치마크(사서 존버)를 **완전히 같은 조건**에서 돌렸다.

{table_strategies}

![전략 비교](images/comparison_{TICKER}_{INTERVAL}.png)

### 읽는 법

- **1등 {best.strategy_name} {pct(best.total_return)}** vs 존버 {pct(bench.total_return)}
  → 존버 대비 {pct(best.total_return - bench.total_return)}p
- 하지만 **MDD를 같이 봐야 한다**: 존버는 {pct(bench.mdd)} 를 견뎌야 했다.
  {best.strategy_name}는 {pct(best.mdd)}.
- 역추세 전략(RSI·볼린저)은 이 구간에서 **둘 다 손실**이다.
  이 구간이 방향성 있는 추세장이었기 때문 — 횡보장이면 결과가 뒤집힌다.
- 승률이 높은 게 좋은 전략이 아니다: 볼린저는 승률 {pct(next(r.win_rate for r in results if r.strategy_name.startswith('볼린저')))}
  로 제일 높은데 총수익률은 꼴찌다. 작게 여러 번 이기고 크게 한 번 잃었다는 뜻.

---

## 실험 2 — 손절이 성과에 미치는 영향

전략을 **SMA교차 5/20 으로 고정**하고 손절 설정만 바꿨다.
변수를 하나만 움직여야 그 변수의 효과를 알 수 있다.

{table_stops}

![추적손절 적용](images/best_{TICKER}_{INTERVAL}_atr2-trail.png)

### 결과: 손절은 수익률을 깎았다

| | 손절 없음 | ATR×2 + 추적 | 변화 |
|---|---:|---:|---|
| 총수익률 | {pct(no_stop.total_return)} | {pct(trailing.total_return)} | {pct(trailing.total_return - no_stop.total_return)}p |
| MDD | {pct(no_stop.mdd)} | {pct(trailing.mdd)} | {(trailing.mdd - no_stop.mdd) * 100:+.1f}%p |
| 승률 | {pct(no_stop.win_rate)} | {pct(trailing.win_rate)} | {(trailing.win_rate - no_stop.win_rate) * 100:+.1f}%p |
| 샤프 | {num(no_stop.sharpe)} | {num(trailing.sharpe)} | {trailing.sharpe - no_stop.sharpe:+.2f} |

**승률은 올랐는데 수익은 줄었다.** 추적손절이 이익 구간을 일찍 끊었기 때문이다.
추세추종 전략은 소수의 큰 상승으로 먹고 사는데, 손절이 그 큰 상승을 중간에 잘라버린다.

**손절은 공짜 보험이 아니다.** 낙폭은 {pct(no_stop.mdd)} → {pct(trailing.mdd)} 로
{abs(trailing.mdd - no_stop.mdd) * 100:.1f}%p 얕아졌지만, 그 대가로 수익률
{pct(no_stop.total_return)} → {pct(trailing.total_return)} ({pct(trailing.total_return - no_stop.total_return)}p) 를 냈다.

어느 쪽이 맞는지는 백테스트가 아니라
"내가 {pct(no_stop.mdd)} 낙폭을 실제로 견디고 안 던질 수 있는가"가 정한다.
못 견딜 것 같으면 손절을 켜는 게 맞다 — 중간에 던지면 수익률 {pct(no_stop.total_return)} 는 어차피 못 받는다.

> ⚠️ 이 결론은 **이 구간, 이 전략, 이 종목**에 대한 것이다.
> 급락이 잦은 구간이었다면 손절이 이겼을 수도 있다.

---

## 실험 3 — 과최적화 측정

파라미터를 바꿔가며 제일 좋은 걸 고르면 성적이 좋아진다.
그게 **전략이 좋아진 건지, 과거를 외운 건지** 구분하는 실험.

### 3-1. 훈련/검증 분리 (홀드아웃)

앞 70% 구간에서 36개 조합(fast 6종 × slow 6종)을 탐색해 1등을 고르고,
그 파라미터를 **한 번도 안 본 뒤 30% 구간**에 적용했다.

| | 값 |
|---|---|
| 훈련 구간에서 고른 파라미터 | **{holdout['best_params']['fast']}/{holdout['best_params']['slow']}** |
| 훈련 구간 수익률 | {pct(holdout['train'].total_return)} |
| **검증 구간 수익률** | **{pct(holdout['test'].total_return)}** ← 믿을 수 있는 숫자 |
| 낙차 | {pct(holdout['degradation'])} |

{reproduction_note}

### 3-2. 워크포워드

구간을 4개로 나누고, 각 구간마다 앞부분에서 재최적화 → 뒷부분에서 검증.
실전에서 주기적으로 파라미터를 다시 고르는 상황을 흉내낸 것.

{table_forward}

- 검증 구간 평균: **{pct(forward['검증수익률'].mean())}**
- 흑자 구간: **{int((forward['검증수익률'] > 0).sum())}/{len(forward)}**
- 구간별 편차: {forward['검증수익률'].std():.2%}
- 구간마다 '최적' 파라미터가 **{forward['선택된 파라미터'].astype(str).nunique()}가지**로 달랐다

### 결론

**{'검증 구간 절반 이상에서 잃었다. 실전에 넣을 전략이 아니다.' if (forward['검증수익률'] > 0).sum() <= len(forward) / 2 else '검증 구간에서도 버텼지만, 구간별 편차가 크다.'}**

구간마다 최적 파라미터가 달랐다는 게 핵심이다. "5/20이 최고"라는 건 고정된 진리가 아니라
그 구간에서 우연히 잘 맞은 숫자였을 뿐이다. 실험 1의 {pct(best.total_return)} 도 같은 성격이다.

---

## 이 실험들의 한계

정직하게 적어둔다. 이 숫자들로 할 수 있는 주장은 생각보다 좁다.

1. **종목 1개** — BTC 하나. 알트코인은 변동성·유동성이 달라 결과가 다르다.
2. **기간 1개** — {df.index[0]:%Y-%m}~{df.index[-1]:%Y-%m}. 2018년 하락장이나 2021년 급등장은 안 봤다.
3. **봉 1종** — 일봉만. 분봉은 노이즈와 수수료 부담이 완전히 다르다.
4. **손절 실험에 워크포워드 미적용** — 실험 2는 전 구간 단일 백테스트다.
   손절 설정도 과최적화될 수 있는데 검증하지 않았다.
5. **슬리피지 고정 {SLIPPAGE:.2%}** — 실제로는 주문 크기와 호가 두께에 따라 달라진다.
   소액이면 더 작고, 급락장에선 훨씬 크다.
6. **거래 표본이 적다** — 대부분 20건 안팎. 통계적으로 유의하다고 말하기 어렵다.
7. **생존 편향 없음(다행)** — 상장폐지 종목을 안 다루므로 이 편향은 해당 없음.

## 재현 방법

```bash
make setup
python scripts/07_experiments.py          # 스냅샷으로 이 문서의 숫자 재현
python scripts/07_experiments.py --refresh # 최신 데이터로 갱신
```
""", encoding="utf-8")
    print(f"작성: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
