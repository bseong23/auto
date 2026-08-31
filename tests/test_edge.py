"""엣지 검증 도구 — 기준선이 '노출·거래 횟수가 같으면서 타이밍만 다른지' 확인한다.
이게 틀리면 비교 자체가 무의미하다."""
import numpy as np
import pandas as pd
import pytest

from upbit.edge import EdgeReport, FixedSeries, measure_edge, runs_of, shuffle_timing
from upbit.strategies import MACrossStrategy


def series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"), dtype=int)


# ---------- 구간 압축 ----------

def test_runs_of_compresses_alternating_blocks():
    assert runs_of(series([0, 0, 1, 1, 1, 0, 0])) == (0, [2, 3, 2])
    assert runs_of(series([1, 1, 0, 1])) == (1, [2, 1, 1])


def test_runs_of_single_block_and_empty():
    assert runs_of(series([1, 1, 1])) == (1, [3])
    assert runs_of(series([])) == (0, [])


# ---------- 섞기가 보존해야 하는 것 ----------

@pytest.fixture
def real_positions():
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 2, 600).cumsum()
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                      index=pd.date_range("2024-01-01", periods=600, freq="D"))
    return MACrossStrategy(5, 20).generate_positions(df)


def test_shuffle_preserves_exposure_exactly(real_positions):
    shuffled = shuffle_timing(real_positions, np.random.default_rng(0))
    assert shuffled.sum() == real_positions.sum()
    assert len(shuffled) == len(real_positions)


def test_shuffle_preserves_trade_count_and_hold_length_distribution(real_positions):
    shuffled = shuffle_timing(real_positions, np.random.default_rng(0))
    first_a, runs_a = runs_of(real_positions)
    first_b, runs_b = runs_of(shuffled)

    holds_a = sorted(runs_a[(1 - first_a)::2])
    holds_b = sorted(runs_b[(1 - first_b)::2])
    assert holds_a == holds_b, "보유 구간 길이의 분포가 달라졌다"
    assert first_a == first_b


def test_shuffle_actually_changes_timing(real_positions):
    shuffled = shuffle_timing(real_positions, np.random.default_rng(0))
    assert not shuffled.equals(real_positions)


def test_shuffle_is_deterministic_per_seed(real_positions):
    a = shuffle_timing(real_positions, np.random.default_rng(7))
    b = shuffle_timing(real_positions, np.random.default_rng(7))
    pd.testing.assert_series_equal(a, b)


def test_shuffle_of_constant_series_is_unchanged():
    flat = series([1] * 50)
    pd.testing.assert_series_equal(shuffle_timing(flat, np.random.default_rng(0)), flat)


# ---------- 판정 ----------

def test_percentile_and_verdict():
    randoms = np.linspace(-0.2, 0.2, 1000)
    strong = EdgeReport("s", real_return=0.5, random_returns=randoms, exposure=0.5, num_trades=10)
    assert strong.percentile == 100.0 and "우연으로 보기 어렵다" in strong.verdict()

    average = EdgeReport("a", real_return=0.0, random_returns=randoms, exposure=0.5, num_trades=10)
    assert 49 <= average.percentile <= 51 and "베타" in average.verdict()

    bad = EdgeReport("b", real_return=-0.5, random_returns=randoms, exposure=0.5, num_trades=10)
    assert bad.percentile == 0.0 and "해를" in bad.verdict()


def test_measure_edge_end_to_end():
    """하락 후 반등 → 골든크로스로 진입해 끝까지 보유. 무작위 기준선이 만들어져야 한다."""
    close = np.concatenate([np.linspace(200, 100, 60), np.linspace(100, 300, 340)])
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                      index=pd.date_range("2024-01-01", periods=400, freq="D"))
    report = measure_edge(df, MACrossStrategy(5, 20), n_random=50, seed=0, fee=0.0, slippage=0.0)

    assert report.num_trades == 0 and report.exposure > 0.7  # 진입 후 미청산 보유
    assert len(report.random_returns) == 50
    assert 0.0 <= report.percentile <= 100.0
    # 무작위 기준선들도 노출이 같으므로 하나같이 양수 수익이어야 한다 (상승장)
    assert (report.random_returns > 0).all()


def test_fixed_series_strategy_reindexes_to_frame():
    df = pd.DataFrame({"close": [1.0] * 5}, index=pd.date_range("2024-01-01", periods=5, freq="D"))
    positions = series([1, 0, 1])
    out = FixedSeries(positions).generate_positions(df)
    assert list(out) == [1, 0, 1, 0, 0]
