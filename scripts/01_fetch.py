#!/usr/bin/env python
"""1단계: 데이터 가져오기 — API 키 없이, 돈 없이, 지금 바로.

    python scripts/01_fetch.py
    python scripts/01_fetch.py --ticker KRW-ETH --interval minute60 --count 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from upbite.data import VALID_INTERVALS, describe, get_ohlcv


def main() -> None:
    ap = argparse.ArgumentParser(description="업비트 캔들 데이터 조회")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--compare", action="store_true", help="봉 종류별로 비교해서 보기")
    args = ap.parse_args()

    pd.set_option("display.width", 140, "display.max_columns", 10)

    if args.compare:
        print(f"[{args.ticker}] 봉 종류별 비교 — 같은 개수라도 담는 기간이 다르다\n")
        for interval in ("minute5", "minute60", "day", "week"):
            df = get_ohlcv(args.ticker, interval, 100)
            print(f"  {interval:9s} {describe(df)}")
        print("\n→ 짧은 봉일수록 노이즈 많고 수수료 자주 나감. 초보는 day / minute60 권장.")
        return

    df = get_ohlcv(args.ticker, args.interval, args.count)
    print(f"[{args.ticker} / {args.interval}] {describe(df)}\n")
    print(df.tail(10).to_string())
    print("\n컬럼 = OHLCV:")
    print("  open   시가   — 이 봉이 시작할 때 가격")
    print("  high   고가   — 이 봉에서 제일 비쌌던 가격")
    print("  low    저가   — 이 봉에서 제일 쌌던 가격")
    print("  close  종가   — 이 봉이 끝날 때 가격")
    print("  volume 거래량 — 거래된 코인 수량")
    print("  value  거래대금 — 거래된 원화 금액")
    print("\n캔들 하나 = 이 값들의 묶음. '차트'는 결국 이 숫자들의 나열이다.")


if __name__ == "__main__":
    main()
