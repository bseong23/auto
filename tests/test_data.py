"""데이터 레이어 — 네트워크 없이 검증 가능한 부분만."""
import pandas as pd
import pytest

from upbite import data


def test_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval"):
        data.get_ohlcv("KRW-BTC", interval="minute7")


def test_rejects_bad_count():
    with pytest.raises(ValueError, match="count"):
        data.get_ohlcv("KRW-BTC", count=0)


def test_validate_sorts_and_drops_duplicates():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"])
    df = pd.DataFrame(
        {"open": [3, 1, 2, 99], "high": [3, 1, 2, 99],
         "low": [3, 1, 2, 99], "close": [3, 1, 2, 99]},
        index=idx,
    )
    out = data._validate(df, "TEST")
    assert out.index.is_monotonic_increasing
    assert len(out) == 3
    assert out.loc["2024-01-02", "close"] == 99  # 중복은 마지막 값 유지


def test_validate_raises_when_everything_is_missing():
    df = pd.DataFrame(
        {"open": [None], "high": [None], "low": [None], "close": [None]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    with pytest.raises(RuntimeError):
        data._validate(df, "TEST")
