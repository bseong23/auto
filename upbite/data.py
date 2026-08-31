"""OHLCV 데이터 수집 + 로컬 캐시.

업비트 시세 조회는 인증이 필요 없다(공개 데이터). 주문만 API 키가 필요하다.
백테스팅을 반복하면서 매번 API를 때리면 느리고 실례이므로 CSV로 캐싱한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyupbit

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

VALID_INTERVALS = (
    "minute1", "minute3", "minute5", "minute10", "minute15",
    "minute30", "minute60", "minute240", "day", "week", "month",
)


def _cache_path(ticker: str, interval: str, count: int) -> Path:
    return CACHE_DIR / f"{ticker}_{interval}_{count}.csv"


def get_ohlcv(
    ticker: str = "KRW-BTC",
    interval: str = "day",
    count: int = 200,
    use_cache: bool = True,
    max_age_sec: int = 3600,
) -> pd.DataFrame:
    """캔들 데이터를 DataFrame으로 반환한다.

    컬럼: open, high, low, close, volume, value / 인덱스: 캔들 시작 시각(KST).
    count > 200 이면 pyupbit가 내부적으로 나눠서 받아온다.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval은 {VALID_INTERVALS} 중 하나여야 함. 받은 값: {interval!r}")
    if count < 1:
        raise ValueError(f"count는 1 이상이어야 함. 받은 값: {count}")

    path = _cache_path(ticker, interval, count)
    if use_cache and path.exists() and (time.time() - path.stat().st_mtime) < max_age_sec:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _validate(df, ticker)

    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    if df is None or df.empty:
        raise RuntimeError(
            f"{ticker} 데이터를 못 받았다. 티커 오타이거나 API 호출 제한일 수 있음."
        )

    df = _validate(df, ticker)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def _validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """시간순 정렬, 중복 캔들 제거, 결측 확인."""
    df = df[~df.index.duplicated(keep="last")].sort_index()
    missing = df[["open", "high", "low", "close"]].isna().any(axis=1)
    if missing.any():
        df = df[~missing]
    if df.empty:
        raise RuntimeError(f"{ticker}: 유효한 캔들이 하나도 없다.")
    return df


def get_tickers(fiat: str = "KRW") -> list[str]:
    """원화 마켓 티커 목록 (예: KRW-BTC, KRW-ETH ...)."""
    return pyupbit.get_tickers(fiat=fiat)


def describe(df: pd.DataFrame) -> str:
    """데이터 요약 한 줄 — 스크립트에서 사람이 눈으로 확인할 용도."""
    first, last = df.index[0], df.index[-1]
    return (
        f"{len(df)}개 캔들 | {first:%Y-%m-%d %H:%M} ~ {last:%Y-%m-%d %H:%M} | "
        f"종가 {df['close'].iloc[0]:,.0f} → {df['close'].iloc[-1]:,.0f}"
    )
