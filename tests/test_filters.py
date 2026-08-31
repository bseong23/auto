"""필터 검증 — 필터가 신호를 잘못 바꾸면 전략 자체가 달라진다."""
import numpy as np
import pandas as pd
import pytest

from upbit.strategies import (
    BuyAndHoldStrategy,
    MACrossStrategy,
    TrendFilter,
    VolatilityFilter,
)
from upbit.strategies.base import Strategy


class AlwaysHold(Strategy):
    name = "항상보유"

    def generate_positions(self, df):
        return pd.Series(1, index=df.index, dtype=int)


def frame(closes, high_mult=1.01, low_mult=0.99):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": closes * high_mult,
         "low": closes * low_mult, "close": closes, "volume": np.ones(len(closes))},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


# ---------- 공통 규약 ----------

def test_filter_only_removes_never_adds():
    """필터는 안 사게 만들 뿐, 없던 매수를 만들면 안 된다."""
    rng = np.random.default_rng(4)
    df = frame(100 + rng.normal(0, 2, 400).cumsum())
    base = MACrossStrategy(5, 20)

    original = base.generate_positions(df)
    filtered = TrendFilter(base, 50).generate_positions(df)

    assert (filtered <= original).all()


def test_filtered_positions_stay_binary():
    rng = np.random.default_rng(9)
    df = frame(100 + rng.normal(0, 2, 300).cumsum())
    positions = TrendFilter(MACrossStrategy(5, 20)).generate_positions(df)
    assert positions.isin((0, 1)).all()


def test_no_lookahead():
    rng = np.random.default_rng(1)
    df = frame(100 + rng.normal(0, 2, 400).cumsum())
    strategy = TrendFilter(VolatilityFilter(MACrossStrategy(5, 20), min_pct=0.01), 100)

    base = strategy.generate_positions(df)
    tampered = df.copy()
    tampered.iloc[250:] *= 4.0

    pd.testing.assert_series_equal(base.iloc[:250],
                                   strategy.generate_positions(tampered).iloc[:250])


def test_params_show_the_wrapped_strategy():
    strategy = TrendFilter(MACrossStrategy(5, 20), 100)
    assert strategy.params()["window"] == 100
    assert "MACrossStrategy" in strategy.params()["inner"]


# ---------- 추세 필터 ----------

def test_trend_filter_blocks_below_the_long_average():
    """꾸준히 내려가면 종가가 장기선 아래라 아예 안 산다."""
    df = frame(np.linspace(200, 100, 300))
    assert TrendFilter(AlwaysHold(), 100).generate_positions(df).sum() == 0


def test_trend_filter_allows_above_the_long_average():
    df = frame(np.linspace(100, 300, 300))
    positions = TrendFilter(AlwaysHold(), 100).generate_positions(df)
    assert positions.iloc[-1] == 1
    assert positions.iloc[150:].mean() > 0.9


def test_trend_filter_blocks_during_warmup():
    """장기선이 아직 계산 안 된 초반엔 참여하지 않는다."""
    df = frame(np.linspace(100, 300, 300))
    assert TrendFilter(AlwaysHold(), 100).generate_positions(df).iloc[:99].sum() == 0


# ---------- 변동성 필터 ----------

def test_volatility_filter_blocks_a_dead_flat_market():
    """변동성이 거의 0인 횡보장 — 교차만 잦고 수수료만 나가는 구간."""
    df = frame([100.0] * 200, high_mult=1.0, low_mult=1.0)
    assert VolatilityFilter(AlwaysHold(), min_pct=0.01).generate_positions(df).sum() == 0


def test_volatility_filter_allows_a_moving_market():
    rng = np.random.default_rng(3)
    df = frame(100 + rng.normal(0, 3, 300).cumsum(), high_mult=1.05, low_mult=0.95)
    assert VolatilityFilter(AlwaysHold(), min_pct=0.005).generate_positions(df).iloc[-1] == 1


def test_volatility_filter_blocks_a_panic_market():
    """변동성 상한 — 패닉 구간엔 손절이 계속 털린다."""
    rng = np.random.default_rng(7)
    df = frame(100 + rng.normal(0, 3, 300).cumsum(), high_mult=1.30, low_mult=0.70)
    assert VolatilityFilter(AlwaysHold(), max_pct=0.05).generate_positions(df).sum() == 0


def test_volatility_filter_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="작아야"):
        VolatilityFilter(AlwaysHold(), min_pct=0.10, max_pct=0.02)


def test_volatility_filter_needs_at_least_one_bound():
    with pytest.raises(ValueError, match="하나는 있어야"):
        VolatilityFilter(AlwaysHold(), min_pct=None, max_pct=None)


# ---------- 겹쳐 쓰기 ----------

def test_filters_compose_and_both_must_agree():
    rng = np.random.default_rng(2)
    df = frame(100 + rng.normal(0.3, 2, 400).cumsum(), high_mult=1.02, low_mult=0.98)

    trend_only = TrendFilter(AlwaysHold(), 100).generate_positions(df)
    vol_only = VolatilityFilter(AlwaysHold(), min_pct=0.015).generate_positions(df)
    both = TrendFilter(VolatilityFilter(AlwaysHold(), min_pct=0.015), 100).generate_positions(df)

    pd.testing.assert_series_equal(both, (trend_only.astype(bool) & vol_only.astype(bool)).astype(int),
                                   check_names=False)


def test_filter_can_wrap_buy_and_hold():
    df = frame(np.linspace(100, 300, 300))
    filtered = TrendFilter(BuyAndHoldStrategy(), 100).generate_positions(df)
    assert 0 < filtered.mean() < 1
