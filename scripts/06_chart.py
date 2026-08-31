#!/usr/bin/env python
"""백테스트 결과를 그래프로 — 숫자만으로는 안 보이는 걸 본다.

    python scripts/06_chart.py
    python scripts/06_chart.py --ticker KRW-ETH --stop-atr 2.0 --trailing
    python scripts/06_chart.py --out docs/images    # README용 이미지 갱신
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbit.backtest import UPBIT_FEE, compare, run_backtest
from upbit.data import VALID_INTERVALS, describe, get_ohlcv
from upbit.plotting import plot_comparison, plot_result
from upbit.risk import RiskRules
from upbit.strategies import (
    BollingerStrategy,
    BuyAndHoldStrategy,
    MACrossStrategy,
    RSIStrategy,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="백테스트 결과 그래프 생성")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=20)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--fee", type=float, default=UPBIT_FEE)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--stop-atr", type=float, default=None, help="ATR 배수 손절 (예: 2.0)")
    ap.add_argument("--stop-pct", type=float, default=None, help="고정 %% 손절 (예: 0.05)")
    ap.add_argument("--take-profit", type=float, default=None, help="고정 %% 익절")
    ap.add_argument("--trailing", action="store_true", help="추적 손절")
    ap.add_argument("--out", default="reports", help="이미지 저장 폴더")
    args = ap.parse_args()

    pd.set_option("display.width", 160, "display.max_columns", 15)

    risk = RiskRules(
        stop_loss_pct=args.stop_pct,
        atr_multiple=args.stop_atr,
        take_profit_pct=args.take_profit,
        trailing=args.trailing,
    )

    df = get_ohlcv(args.ticker, args.interval, args.count)
    print(f"[{args.ticker} / {args.interval}] {describe(df)}")
    print(f"손절규칙: {risk.describe()}\n")

    strategies = [
        MACrossStrategy(args.fast, args.slow),
        MACrossStrategy(args.fast, args.slow, use_ema=True),
        RSIStrategy(),
        BollingerStrategy(),
        BuyAndHoldStrategy(),
    ]
    results = [
        run_backtest(df, s, args.capital, args.fee, args.slippage,
                     risk=None if isinstance(s, BuyAndHoldStrategy) else risk)
        for s in strategies
    ]

    table = compare(results)
    display = table.copy()
    for col in ("총수익률", "CAGR", "MDD", "승률"):
        display[col] = display[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.2%}")
    display["샤프"] = display["샤프"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    display["손익비"] = display["손익비"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    print(display.to_string(index=False))

    out_dir = (ROOT / args.out).resolve()
    # 손절 설정이 다르면 파일도 달라야 한다 (안 그러면 서로 덮어쓴다)
    suffix = ""
    if risk.is_active:
        parts = []
        if args.stop_atr:
            parts.append(f"atr{args.stop_atr:g}")
        if args.stop_pct:
            parts.append(f"pct{args.stop_pct:g}")
        if args.take_profit:
            parts.append(f"tp{args.take_profit:g}")
        if args.trailing:
            parts.append("trail")
        suffix = "_" + "-".join(parts)
    tag = f"{args.ticker}_{args.interval}{suffix}"

    best = max(results, key=lambda r: r.total_return)
    detail_path = plot_comparison(results, out_dir / f"comparison_{tag}.png",
                                  title=f"{args.ticker} {args.interval} — 전략 비교"
                                        f"  ({risk.describe()})")
    single_path = plot_result(df=df, result=best, fast=args.fast, slow=args.slow,
                              path=out_dir / f"best_{tag}.png")

    print(f"\n저장:\n  {detail_path.relative_to(ROOT)}\n  {single_path.relative_to(ROOT)}")

    if risk.is_active:
        stopped = sum(
            1 for r in results for t in r.closed_trades if t.exit_reason.startswith("손절")
        )
        total = sum(r.num_trades for r in results)
        print(f"\n손절로 청산된 거래: {stopped}/{total}건")


if __name__ == "__main__":
    main()
