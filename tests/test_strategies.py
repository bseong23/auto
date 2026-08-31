"""전략 공통 규약 검증 — 새 전략을 추가하면 여기 STRATEGIES에만 넣으면 된다."""
import numpy as np
import pandas as pd
import pytest

from upbit.strategies import (
    BollingerStrategy,
    BuyAndHoldStrategy,
    MACrossStrategy,
    RSIStrategy,
)

STRATEGIES = [
    MACrossStrategy(5, 20),
    MACrossStrategy(10, 30, use_ema=True),
    RSIStrategy(),
    BollingerStrategy(),
    BuyAndHoldStrategy(),
]


@pytest.fixture
def price_df():
    rng = np.random.default_rng(42)
    close = 100_000_000 + rng.normal(0, 1_000_000, 400).cumsum()
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": np.ones(400),
        },
        index=idx,
    )


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_positions_are_binary_and_aligned(strategy, price_df):
    pos = strategy.generate_positions(price_df)
    assert len(pos) == len(price_df)
    assert pos.index.equals(price_df.index)
    assert pos.isin((0, 1)).all(), "포지션은 0(현금) 또는 1(보유)만"


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_no_lookahead(strategy, price_df):
    """미래 가격을 조작해도 과거 포지션은 그대로여야 한다."""
    cut = 250
    base = strategy.generate_positions(price_df)

    tampered = price_df.copy()
    tampered.iloc[cut:] *= 3.0
    after = strategy.generate_positions(tampered)

    pd.testing.assert_series_equal(base.iloc[:cut], after.iloc[:cut])


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_warmup_period_stays_flat(strategy, price_df):
    """지표가 아직 계산되지 않은 초반엔 포지션을 잡으면 안 된다."""
    if isinstance(strategy, BuyAndHoldStrategy):
        pytest.skip("존버는 첫 봉부터 보유가 정상")
    assert strategy.generate_positions(price_df).iloc[0] == 0


def test_ma_cross_enters_on_golden_cross():
    """하락 후 급반등 → 골든크로스에서 진입해야 한다."""
    prices = np.concatenate([np.linspace(200, 100, 40), np.linspace(100, 300, 40)])
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=pd.date_range("2024-01-01", periods=80, freq="D"),
    )
    pos = MACrossStrategy(5, 20).generate_positions(df)
    assert pos.iloc[:40].sum() == 0     # 하락 구간엔 현금
    assert pos.iloc[-1] == 1            # 상승 추세엔 보유


def test_rsi_strategy_buys_oversold_sells_overbought():
    down = np.linspace(200, 100, 30)     # RSI 바닥으로
    up = np.linspace(100, 250, 30)       # RSI 천장으로
    prices = np.concatenate([down, up])
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=pd.date_range("2024-01-01", periods=60, freq="D"),
    )
    pos = RSIStrategy(14, 30, 70).generate_positions(df)
    assert pos.iloc[29] == 1   # 과매도 구간에서 매수 상태
    assert pos.iloc[-1] == 0   # 과매수 구간에선 청산


def test_ma_cross_rejects_invalid_windows():
    with pytest.raises(ValueError, match="작아야"):
        MACrossStrategy(fast=20, slow=5)


def test_rsi_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        RSIStrategy(buy_below=80, sell_above=20)


def test_params_and_repr_are_readable():
    s = MACrossStrategy(5, 20)
    assert s.params() == {"fast": 5, "slow": 20, "use_ema": False}
    assert "fast=5" in repr(s)
