"""지표 계산이 손으로 계산한 값과 맞는지."""
import numpy as np
import pandas as pd
import pytest

from upbite import indicators as ind


def test_sma_matches_hand_calculation():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])  # 창이 안 찼으면 NaN
    assert out.iloc[2] == pytest.approx(2.0)                 # (1+2+3)/3
    assert out.iloc[4] == pytest.approx(4.0)                 # (3+4+5)/3


def test_sma_never_uses_future_data():
    """t 시점 값은 t 이후 데이터를 바꿔도 변하지 않아야 한다."""
    s = pd.Series(np.arange(50, dtype=float))
    base = ind.sma(s, 10)
    tampered = s.copy()
    tampered.iloc[30:] = 9999.0
    after = ind.sma(tampered, 10)
    pd.testing.assert_series_equal(base.iloc[:30], after.iloc[:30])


def test_rsi_bounds_and_extremes():
    up_only = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up_only, 14).iloc[-1] == pytest.approx(100.0)

    down_only = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down_only, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    noisy = pd.Series(np.random.default_rng(0).normal(100, 5, 500).cumsum())
    r = ind.rsi(noisy).dropna()
    assert r.between(0, 100).all()


def test_bollinger_band_ordering():
    s = pd.Series(np.random.default_rng(1).normal(100, 3, 200))
    lo, mid, up = ind.bollinger(s, 20, 2)
    valid = mid.notna()
    assert (lo[valid] <= mid[valid]).all()
    assert (mid[valid] <= up[valid]).all()


def test_crossover_fires_only_on_the_crossing_bar():
    fast = pd.Series([1.0, 1.0, 3.0, 4.0, 5.0])
    slow = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cross = ind.crossover(fast, slow)
    assert cross.tolist() == [False, False, True, False, False]


def test_crossunder_is_mirror_of_crossover():
    fast = pd.Series([5.0, 5.0, 1.0, 0.5])
    slow = pd.Series([2.0, 2.0, 2.0, 2.0])
    assert ind.crossunder(fast, slow).tolist() == [False, False, True, False]


def test_atr_is_positive():
    rng = np.random.default_rng(2)
    close = pd.Series(rng.normal(100, 2, 300).cumsum() + 1000)
    df = pd.DataFrame({
        "high": close + 5, "low": close - 5, "close": close, "open": close,
    })
    assert (ind.atr(df, 14).dropna() > 0).all()
