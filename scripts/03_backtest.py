#!/usr/bin/env python
"""4단계: 백테스팅 — 실제 돈 넣기 전에 무조건 이거 먼저.

    python scripts/03_backtest.py
    python scripts/03_backtest.py --ticker KRW-ETH --interval minute60 --count 1000
    python scripts/03_backtest.py --strategy ma --fast 10 --slow 30 --detail
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbite.backtest import UPBIT_FEE, compare, run_backtest
from upbite.data import VALID_INTERVALS, describe, get_ohlcv
from upbite.strategies import (
    BollingerStrategy,
    BuyAndHoldStrategy,
    MACrossStrategy,
    RSIStrategy,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def build_strategies(args) -> list:
    catalog = {
        "ma": lambda: [MACrossStrategy(args.fast, args.slow)],
        "ema": lambda: [MACrossStrategy(args.fast, args.slow, use_ema=True)],
        "rsi": lambda: [RSIStrategy()],
        "bb": lambda: [BollingerStrategy()],
        "hold": lambda: [BuyAndHoldStrategy()],
        "all": lambda: [
            MACrossStrategy(args.fast, args.slow),
            MACrossStrategy(args.fast, args.slow, use_ema=True),
            RSIStrategy(),
            BollingerStrategy(),
            BuyAndHoldStrategy(),
        ],
    }
    return catalog[args.strategy]()


def main() -> None:
    ap = argparse.ArgumentParser(description="전략 백테스팅")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--strategy", default="all", choices=["all", "ma", "ema", "rsi", "bb", "hold"])
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=20)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--fee", type=float, default=UPBIT_FEE)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--detail", action="store_true", help="개별 거래 내역까지 출력")
    ap.add_argument("--save", action="store_true", help="reports/ 에 CSV 저장")
    args = ap.parse_args()

    pd.set_option("display.width", 160, "display.max_columns", 15)

    df = get_ohlcv(args.ticker, args.interval, args.count)
    print(f"[{args.ticker} / {args.interval}] {describe(df)}")
    print(f"수수료 {args.fee:.3%} + 슬리피지 {args.slippage:.3%} 반영 | 자본 {args.capital:,.0f}원\n")

    results = [
        run_backtest(df, s, args.capital, args.fee, args.slippage)
        for s in build_strategies(args)
    ]

    table = compare(results)
    display = table.copy()
    for col in ("총수익률", "CAGR", "MDD", "승률"):
        display[col] = display[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.2%}")
    display["샤프"] = display["샤프"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    display["손익비"] = display["손익비"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    print(display.to_string(index=False))

    if args.detail:
        for res in results:
            print("\n" + "=" * 58)
            print(res.summary())
            trades = res.trades_frame()
            if not trades.empty:
                shown = trades.copy()
                shown["손익률"] = shown["손익률"].map(
                    lambda v: "미청산" if pd.isna(v) else f"{v:+.2%}"
                )
                for col in ("진입가", "청산가"):
                    shown[col] = shown[col].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
                print("\n거래 내역:")
                print(shown.to_string(index=False))

    if args.save:
        REPORTS.mkdir(exist_ok=True)
        tag = f"{args.ticker}_{args.interval}_{args.count}"
        table.to_csv(REPORTS / f"compare_{tag}.csv", index=False)
        for res in results:
            safe = res.strategy_name.replace("/", "-").replace(" ", "")
            res.trades_frame().to_csv(REPORTS / f"trades_{tag}_{safe}.csv", index=False)
        print(f"\n저장: {REPORTS}")

    best = max(results, key=lambda r: r.total_return)
    bench = next((r for r in results if r.strategy_name == "사서 존버"), None)
    print("\n" + "-" * 58)
    print(f"이 구간 1등: {best.strategy_name} ({best.total_return:+.2%}, MDD {best.mdd:.2%})")
    if bench is not None and best is not bench:
        gap = best.total_return - bench.total_return
        verdict = "존버보다 나음" if gap > 0 else "존버보다 못함 — 수수료만 낸 셈"
        print(f"존버 대비: {gap:+.2%}p → {verdict}")
    print(
        "\n⚠️  이건 '과거 이 구간에서 그랬다'는 뜻일 뿐 미래 보장이 아니다.\n"
        "    기간·코인을 바꿔서 여러 번 돌려보고, 어디서 깨지는지 확인할 것."
    )


if __name__ == "__main__":
    main()
