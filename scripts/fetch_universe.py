#!/usr/bin/env python
"""크로스섹션 모멘텀용 유니버스 수집 — 업비트 원화마켓 거래대금 상위 N개, 일봉 5년.

    python scripts/fetch_universe.py            # 상위 25개
    python scripts/fetch_universe.py --top 40

선정 기준과 시점을 _selection.json 에 남긴다. **생존 편향 주의**: 지금 상장돼 있는
코인만 뽑히므로, 과거에 상장폐지된 코인은 처음부터 빠진다. 결과는 실제보다 후하다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyupbit
import requests

from upbit.data import get_ohlcv
from upbit.exchange import tick_ratio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "universe"
STABLE = ("USDT", "USDC", "DAI", "TUSD", "USD1", "USDS", "FDUSD", "PYUSD")


def rank_by_trade_value(tickers: list[str]) -> list[dict]:
    rows = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i : i + 100]
        resp = requests.get("https://api.upbit.com/v1/ticker",
                            params={"markets": ",".join(chunk)}, timeout=(5, 15))
        resp.raise_for_status()
        for r in resp.json():
            price = float(r["trade_price"])
            rows.append({
                "ticker": r["market"],
                "trade_value_24h": float(r["acc_trade_price_24h"]),
                "price": price,
                "tick_ratio": tick_ratio(price),
            })
    return sorted(rows, key=lambda r: -r["trade_value_24h"])


def fetch_with_retry(ticker: str, count: int, tries: int = 4) -> "pd.DataFrame":
    """레이트리밋(429)이면 pyupbit 이 None 을 돌려준다 → 잠시 쉬고 재시도."""
    for attempt in range(tries):
        try:
            return get_ohlcv(ticker, "day", count, use_cache=False)
        except RuntimeError:
            wait = 2.0 * (attempt + 1)
            print(f"    {ticker}: 응답 없음(레이트리밋?) — {wait:.0f}초 후 재시도")
            time.sleep(wait)
    raise RuntimeError(f"{ticker}: {tries}번 재시도 후에도 실패")


def listed_before(ticker: str, date: str) -> bool:
    """그 날짜 이전에 상장돼 있었나 — 일봉 1개만 요청해 확인 (저렴)."""
    time.sleep(0.15)  # 초당 요청 제한
    resp = requests.get("https://api.upbit.com/v1/candles/days",
                        params={"market": ticker, "count": 1, "to": f"{date}T00:00:00Z"},
                        timeout=(5, 15))
    if resp.status_code != 200:
        return False
    rows = resp.json()
    return bool(rows) and rows[0]["candle_date_time_utc"][:10] <= date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--count", type=int, default=1800)
    ap.add_argument("--listed-before", default="2023-06-01",
                    help="이 날짜 이전 상장 코인만. 오늘 거래대금 상위는 최근 상장·최근 펌핑 코인이 많아 "
                         "그대로 쓰면 '지금 뜬 코인'을 미리 알고 고른 룩어헤드가 된다")
    ap.add_argument("--scan", type=int, default=120, help="거래대금 상위 몇 개까지 훑을지")
    args = ap.parse_args()

    all_krw = pyupbit.get_tickers(fiat="KRW")
    ranked = [r for r in rank_by_trade_value(all_krw) if r["ticker"].split("-")[1] not in STABLE]
    chosen = []
    for r in ranked[: args.scan]:
        if listed_before(r["ticker"], args.listed_before):
            chosen.append(r)
        if len(chosen) >= args.top:
            break
    print(f"거래대금 상위 {args.scan}개 중 {args.listed_before} 이전 상장 {len(chosen)}개 선정\n")

    OUT.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        df = fetch_with_retry(r["ticker"], args.count)
        df.to_csv(OUT / f"{r['ticker']}.csv")
        time.sleep(0.5)
        r["bars"] = len(df)
        r["first_date"] = df.index[0].strftime("%Y-%m-%d")
        print(f"  {r['ticker']:10s} 거래대금 {r['trade_value_24h']/1e8:>8,.0f}억  "
              f"호가/가격 {r['tick_ratio']:.3%}  {len(df)}일 ({r['first_date']}~)")

    (OUT / "_selection.json").write_text(json.dumps({
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "criterion": f"KRW 마켓 24h 거래대금 상위 {args.scan}개 중 {args.listed_before} 이전 상장, 스테이블코인 제외",
        "listed_before": args.listed_before,
        "universe_size_total": len(all_krw),
        "rows": chosen,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(chosen)}개 저장 → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
