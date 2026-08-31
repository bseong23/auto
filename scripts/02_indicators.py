#!/usr/bin/env python
"""2단계: 지표 계산 — 숫자에서 규칙 뽑아내기.

    python scripts/02_indicators.py
    python scripts/02_indicators.py --ticker KRW-ETH --fast 10 --slow 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbite import indicators as ind
from upbite.data import VALID_INTERVALS, describe, get_ohlcv


def main() -> None:
    ap = argparse.ArgumentParser(description="지표 계산해서 눈으로 확인")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=20)
    args = ap.parse_args()

    pd.set_option("display.width", 160, "display.max_columns", 12)

    df = get_ohlcv(args.ticker, args.interval, args.count)
    close = df["close"]

    table = pd.DataFrame({"종가": close})
    table[f"MA{args.fast}"] = ind.sma(close, args.fast)
    table[f"MA{args.slow}"] = ind.sma(close, args.slow)
    table["RSI14"] = ind.rsi(close, 14)
    lower, mid, upper = ind.bollinger(close, 20, 2.0)
    table["BB하단"], table["BB상단"] = lower, upper

    fast_ma = table[f"MA{args.fast}"]
    slow_ma = table[f"MA{args.slow}"]
    golden = ind.crossover(fast_ma, slow_ma)
    dead = ind.crossunder(fast_ma, slow_ma)
    table["신호"] = ""
    table.loc[golden, "신호"] = "골든크로스↑"
    table.loc[dead, "신호"] = "데드크로스↓"

    print(f"[{args.ticker} / {args.interval}] {describe(df)}\n")
    print(table.tail(15).round(0).to_string())

    print(f"\n교차 발생 이력 (최근 10건):")
    events = table[table["신호"] != ""].tail(10)
    for when, row in events.iterrows():
        print(f"  {when:%Y-%m-%d %H:%M}  {row['신호']}   종가 {row['종가']:>14,.0f}원")

    print(f"\n총 골든크로스 {int(golden.sum())}회 / 데드크로스 {int(dead.sum())}회")
    latest = table.iloc[-1]
    state = "보유(단기>장기)" if latest[f"MA{args.fast}"] > latest[f"MA{args.slow}"] else "현금(단기<장기)"
    print(f"현재 상태: {state} | RSI {latest['RSI14']:.1f}")
    print("\n→ 교차가 잦으면 그만큼 수수료를 낸다. 3단계 백테스팅에서 확인할 것.")


if __name__ == "__main__":
    main()
