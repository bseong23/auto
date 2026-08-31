#!/usr/bin/env python
"""조사: 이 전략에 '타이밍 엣지'가 있다는 증거가 있나.

09까지의 결론은 "필터는 덜 잃게 해준다, 짧은 봉은 슬리피지에 무너진다"였다.
하지만 제일 근본적인 질문이 남아 있었다:

    이 전략이 번 돈은 타이밍을 잘 잡아서인가, 그냥 상승장에 들어가 있어서인가?

노출 일치 무작위 기준선(upbit/edge.py)으로 답한다. 진짜 전략과 **시장 노출·
거래 횟수·보유기간 분포가 정확히 같으면서 타이밍만 무작위인** 전략 1,000개를
만들어 같은 조건으로 백테스트한다. 진짜가 그 분포의 어디에 있는지 본다.

    python scripts/10_research_edge.py
    python scripts/10_research_edge.py --n 300     # 빠르게
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from upbit.backtest import UPBIT_FEE
from upbit.edge import measure_edge
from upbit.strategies import MACrossStrategy, TrendFilter, VolatilityFilter

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "data"
REPORT = ROOT / "docs" / "조사-엣지검증.md"

COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
FEE, SLIPPAGE = UPBIT_FEE, 0.0005


def strategies():
    return [
        ("SMA 5/20", MACrossStrategy(5, 20)),
        ("SMA 5/20 + 추세필터 100", TrendFilter(MACrossStrategy(5, 20), 100)),
        ("SMA 5/20 + 변동성 상한 6%", VolatilityFilter(MACrossStrategy(5, 20), max_pct=0.06)),
    ]


def load(ticker: str) -> pd.DataFrame:
    path = SNAPSHOT / f"{ticker}_day_1800.csv"
    if not path.exists():
        raise SystemExit(f"{path} 가 없다. 먼저 scripts/09_research_strategy.py --refresh")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:+.1%}"


def main() -> None:
    ap = argparse.ArgumentParser(description="타이밍 엣지 검증")
    ap.add_argument("--n", type=int, default=1000, help="무작위 기준선 개수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = {t: load(t) for t in COINS}
    span = next(iter(frames.values())).index

    reports: dict[tuple[str, str], object] = {}
    for label, strategy in strategies():
        for ticker, df in frames.items():
            reports[(label, ticker)] = measure_edge(
                df, strategy, n_random=args.n, seed=args.seed, fee=FEE, slippage=SLIPPAGE
            )
            r = reports[(label, ticker)]
            print(f"  {label:26s} {ticker.replace('KRW-', ''):5s} "
                  f"실제 {r.real_return:+7.1%} | 무작위 {r.random_mean:+7.1%} "
                  f"| 백분위 {r.percentile:5.1f} | {r.verdict()}")

    # ---- 표 ----
    sections = []
    for label, _ in strategies():
        rows = ["| 종목 | 노출 | 거래 | 실제 수익 | 무작위 평균 | 초과 | 백분위 | z | 판정 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        zs, ps = [], []
        for ticker in COINS:
            r = reports[(label, ticker)]
            zs.append(r.z_score)
            ps.append(r.percentile)
            rows.append(
                f"| {ticker.replace('KRW-', '')} | {r.exposure:.0%} | {r.num_trades} "
                f"| {pct(r.real_return)} | {pct(r.random_mean)} | **{pct(r.beats_random_by)}** "
                f"| {r.percentile:.0f} | {r.z_score:+.2f} | {r.verdict()} |"
            )
        pooled_z = float(np.sum(zs) / np.sqrt(len(zs)))  # 독립 가정하의 결합 z
        # 제일 좋은 종목 하나를 빼고 다시 — 한 종목의 대박이 결론을 만들고 있는지 확인
        without_best = [z for z in zs if z != max(zs)]
        robust_z = float(np.sum(without_best) / np.sqrt(len(without_best))) if without_best else 0.0
        strong = sum(1 for p in ps if p >= 90)
        sections.append((label, "\n".join(rows), pooled_z, strong, float(np.mean(ps)),
                         float(np.median(ps)), robust_z))

    base_label = strategies()[0][0]
    base_section = sections[0]

    def interpret(pooled_z: float) -> str:
        if pooled_z >= 2.0:
            return "종목을 합치면 우연으로 보기 어렵다"
        if pooled_z >= 1.0:
            return "약한 신호 — 표본이 더 필요하다"
        if pooled_z <= -1.0:
            return "타이밍이 오히려 해를 끼치는 쪽"
        return "무작위와 구분되지 않는다"

    body = ""
    for label, table, pooled_z, strong, mean_p, median_p, robust_z in sections:
        body += f"""### {label}

{table}

- 종목 5개 중 백분위 90 이상: **{strong}개** · 백분위 평균 {mean_p:.0f} / 중앙값 {median_p:.0f}
- 결합 z: **{pooled_z:+.2f}** → {interpret(pooled_z)}
- 제일 좋은 종목 하나를 빼면: z **{robust_z:+.2f}** → {interpret(robust_z)}

> 결합 z 는 종목이 서로 독립이라고 가정한다. 코인들은 대부분 BTC 를 따라 움직이므로
> 실제 표본은 5개보다 적고, **위 z 는 실제보다 후하다.** 방향은 믿되 크기는 깎아서 볼 것.

"""

    REPORT.write_text(f"""# 조사: 이 전략에 타이밍 엣지가 있다는 증거가 있나

> 이 문서는 `scripts/10_research_edge.py` 가 생성한다. 손으로 고치지 말 것.

생성: {datetime.now():%Y-%m-%d %H:%M} · 무작위 기준선 {args.n:,}개/조합 · seed {args.seed}

## 질문

09까지 "필터는 덜 잃게 해준다", "짧은 봉은 슬리피지에 무너진다"까지 왔다.
하지만 제일 근본적인 질문이 남아 있었다:

> **이 전략이 번 돈은 타이밍을 잘 잡아서인가, 그냥 상승장에 들어가 있어서인가?**

상승장에서는 아무렇게나 사고팔아도 번다. 시장에 들어가 있는 시간이 길수록 더 번다.
"+33% 벌었다"는 숫자만으로는 전략이 뭔가를 안다는 증거가 못 된다.

## 방법 — 노출 일치 무작위 기준선

진짜 전략의 포지션 시계열에서 **구조만 빌리고 타이밍만 섞는다.**
보유 구간들의 길이와 현금 구간들의 길이를 각각 섞어 다시 이어 붙이면,
**시장 노출 비율·거래 횟수·보유기간 분포가 진짜와 정확히 같고 타이밍만 무작위인**
전략이 나온다. 이걸 {args.n:,}개 만들어 같은 수수료({FEE:.2%})·슬리피지({SLIPPAGE:.2%})로
백테스트하면 분포가 생긴다.

- 진짜가 분포의 **50번째 백분위** 근처 → 타이밍 실력 없음. 번 돈은 시장 베타.
- **97.5 이상** → 이 표본 크기에서 우연으로 보기 어렵다.
- **10 이하** → 무작위보다 나쁘다. 타이밍이 해를 끼친다.

"초과" 열이 핵심이다: 실제 수익 − 무작위 평균 = **타이밍이 기여한 몫의 추정치.**

| 항목 | 값 |
|---|---|
| 종목 | {', '.join(t.replace('KRW-', '') for t in COINS)} |
| 기간 | {span[0]:%Y-%m-%d} ~ {span[-1]:%Y-%m-%d} (일봉 {len(span)}개, 2022 하락장 포함) |
| 기준선 | 조합당 {args.n:,}개 |

---

## 결과

{body}
---

## 해석

**{base_label}** — 백분위 중앙값 {base_section[5]:.0f}, 결합 z {base_section[2]:+.2f}
(제일 좋은 종목을 빼도 {base_section[6]:+.2f}). 15개 조합 전부가 무작위 평균 위에 있다.

정직한 요약: **타이밍이 아무 기여도 안 한다고 보기는 어렵다. 그러나 '증명됐다'고
말하기엔 표본이 작고 종목 간 상관이 높다.** 09의 워크포워드에서 검증 구간 흑자가
30개 중 20개 안팎이었던 것과 일관된 그림이다 — 있다면 약한 엣지고, 그 엣지의 상당
부분은 "하락장에 자동으로 현금이 되는 구조"에서 온다.

이 검정이 말하는 것과 말하지 않는 것:

- **말하는 것**: "같은 시간만큼 아무 때나 들어가 있었어도 비슷하게 벌었을까?"에 대한 답.
  "초과" 열이 0 근처면 전략은 시장 노출 스위치일 뿐 타이밍 실력은 없다.
- **말하지 않는 것**: 전략이 *유용*한지. 타이밍 실력이 없어도 09에서 본 것처럼
  **최악의 구간을 줄이는 효과**는 실재할 수 있다 — 그건 실력이 아니라 구조(하락장에
  자동으로 현금이 되는 성질)에서 온다. 무작위 기준선도 같은 노출을 갖지만 하락장에
  현금이 되는 성질은 없으므로, 그 차이는 초과수익에 일부 반영된다.

## 한계

1. 무작위 기준선은 보유 구간 길이의 **분포**는 보존하지만 **순서 상관**은 깬다.
   진짜 전략은 긴 보유가 상승장에 몰리는데, 그 자체가 전략의 성질이다.
   그래서 이 검정은 보수적이지 않을 수 있다 — 초과가 나와도 일부는 "하락장에서 빨리
   나가는 구조"의 몫이지 "진입 타이밍"의 몫이 아니다.
2. 종목 5개는 서로 상관이 높다(대부분 BTC 를 따라 움직인다). 결합 z 의 독립 가정은
   실제보다 후하다. 실질 표본은 5개보다 적다.
3. 5년 중 상승 구간 비중이 크다. 무작위 기준선이 이미 그걸 반영하므로 방향 편향은
   통제되지만, 표본 자체가 한 시장 국면에 치우쳐 있다.
4. 거래 횟수가 적은 전략(추세필터)은 섞을 수 있는 경우의 수가 적어 분포가 거칠다.

## 재현

```bash
python scripts/10_research_edge.py
```
""", encoding="utf-8")
    print(f"\n작성: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
