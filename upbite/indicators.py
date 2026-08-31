"""기술적 지표 계산.

모든 함수는 pandas Series를 받아 Series를 돌려준다.
중요: 각 지표는 "그 시점까지의 데이터만" 사용한다 (미래 참조 없음).
"""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """단순이동평균 — 최근 window개 종가의 평균."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """지수이동평균 — 최근 값에 더 큰 가중치."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI (0~100). 70 이상 과매수 / 30 이하 과매도로 통상 해석.

    Wilder 방식(지수평활)을 쓴다 — 원 논문 정의이자 대부분의 차트 프로그램 기본값.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    # 하락이 전혀 없던 구간(avg_loss == 0)은 RSI 100으로 정의된다.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저밴드 (하단, 중심선, 상단)."""
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return mid - num_std * std, mid, mid + num_std * std


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (macd선, 시그널선, 히스토그램)."""
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR — 변동성 크기. 손절 폭을 가격 대비 자동 조절할 때 쓴다."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """골든크로스 시점만 True — 직전 봉엔 아래, 이번 봉엔 위."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """데드크로스 시점만 True."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))
