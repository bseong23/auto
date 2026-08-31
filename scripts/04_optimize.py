#!/usr/bin/env python
"""4.5단계: 파라미터 탐색 + 과최적화 검증.

"5/20보다 좋은 조합이 있나?" 를 찾되, **찾은 게 진짜인지** 검증한다.

    python scripts/04_optimize.py
    python scripts/04_optimize.py --ticker KRW-ETH --count 1200 --splits 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbite.backtest import UPBIT_FEE
from upbite.data import VALID_INTERVALS, describe, get_ohlcv
from upbite.optimize import holdout_test, walk_forward
from upbite.strategies import MACrossStrategy


def pct(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:+.2%}"


def main() -> None:
    ap = argparse.ArgumentParser(description="파라미터 탐색과 과최적화 검증")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--fee", type=float, default=UPBIT_FEE)
    ap.add_argument("--slippage", type=float, default=0.0005)
    args = ap.parse_args()

    pd.set_option("display.width", 170, "display.max_columns", 15)

    df = get_ohlcv(args.ticker, args.interval, args.count)
    print(f"[{args.ticker} / {args.interval}] {describe(df)}\n")

    grid = {"fast": [3, 5, 7, 10, 15, 20], "slow": [20, 30, 40, 60, 90, 120]}
    combos = len(grid["fast"]) * len(grid["slow"])

    # ---- 1) 훈련 구간에서 최적 파라미터 고르고, 검증 구간에서 채점 ----
    print(f"■ 1단계: 훈련(앞 70%)에서 {combos}개 조합 탐색 → 검증(뒤 30%)에서 채점\n")
    out = holdout_test(df, MACrossStrategy, grid, train_ratio=0.7,
                       fee=args.fee, slippage=args.slippage)

    top = out["ranking"].head(5).copy()
    for col in ("총수익률", "CAGR", "MDD", "승률"):
        top[col] = top[col].map(pct)
    top["샤프"] = top["샤프"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    print("훈련 구간 상위 5개 조합:")
    print(top.drop(columns=["_metric"]).to_string(index=False))

    train_r, test_r = out["train"].total_return, out["test"].total_return
    print(f"\n선택: {out['strategy'].name}")
    print(f"  훈련 구간 수익률 : {pct(train_r)}")
    print(f"  검증 구간 수익률 : {pct(test_r)}   ← 실제로 믿을 수 있는 숫자")
    print(f"  낙차             : {pct(out['degradation'])}")
    if test_r < 0 < train_r:
        print("  판정: ❌ 과최적화. 훈련에선 벌었는데 처음 보는 데이터에선 잃었다.")
    elif train_r > 0 and test_r < train_r * 0.3:
        print("  판정: ⚠️  성적 대부분이 훈련 구간에만 있다. 신뢰도 낮음.")
    else:
        print("  판정: ✅ 검증 구간에서도 버텼다. (그래도 미래 보장은 아님)")

    # ---- 2) 워크포워드: 구간을 밀어가며 반복 ----
    print(f"\n■ 2단계: 워크포워드 {args.splits}회 — 구간마다 재최적화하고 검증\n")
    table = walk_forward(df, MACrossStrategy, grid, n_splits=args.splits,
                         fee=args.fee, slippage=args.slippage)
    if table.empty:
        print("데이터가 부족해 워크포워드를 못 돌렸다. --count 를 늘릴 것.")
        return

    shown = table.copy()
    for col in ("훈련수익률", "검증수익률", "검증MDD"):
        shown[col] = shown[col].map(pct)
    print(shown.to_string(index=False))

    oos = table["검증수익률"]
    wins = int((oos > 0).sum())
    print(f"\n검증 구간 성적: 평균 {pct(oos.mean())} | 흑자 {wins}/{len(oos)}구간 "
          f"| 편차 {oos.std():.2%}")

    chosen = table["선택된 파라미터"].astype(str).nunique()
    if chosen > 1:
        print(f"→ 구간마다 '최적' 파라미터가 {chosen}가지로 달랐다. "
              "이게 바로 최적값이 고정된 진리가 아니라는 증거다.")
    if wins <= len(oos) / 2:
        print("→ 검증 구간 절반 이상에서 잃었다. 실전에 넣을 전략이 아니다.")

    print(
        "\n💡 파라미터를 더 잘게 쪼개 탐색할수록 훈련 성적은 무조건 좋아진다.\n"
        "   좋아지는 건 전략이 아니라 '과거를 외운 정도'다. 검증 구간 숫자만 믿을 것."
    )


if __name__ == "__main__":
    main()
