import numpy as np
import pandas as pd
import pytest

from upbite import optimize
from upbite.strategies import MACrossStrategy


@pytest.fixture
def trending_df():
    rng = np.random.default_rng(11)
    close = 100 + rng.normal(0.15, 2.0, 600).cumsum()
    idx = pd.date_range("2023-01-01", periods=600, freq="D")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.ones(600)},
        index=idx,
    )


def test_split_preserves_time_order_and_covers_everything(trending_df):
    train, test = optimize.split(trending_df, 0.7)
    assert len(train) + len(test) == len(trending_df)
    assert train.index[-1] < test.index[0]          # 시간이 섞이면 안 된다
    assert len(train) == pytest.approx(420, abs=1)


def test_split_rejects_bad_ratio(trending_df):
    with pytest.raises(ValueError):
        optimize.split(trending_df, 1.5)


def test_split_rejects_too_short_data():
    tiny = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    with pytest.raises(ValueError, match="너무 짧"):
        optimize.split(tiny, 0.7)


def test_grid_search_ranks_by_metric_and_skips_invalid_combos(trending_df):
    grid = {"fast": [3, 5, 20], "slow": [10, 20]}
    table = optimize.grid_search(trending_df, MACrossStrategy, grid, min_trades=0)

    assert list(table["_metric"]) == sorted(table["_metric"], reverse=True)
    # fast >= slow 조합(20/10, 20/20, 5/... 유효)은 제외되어야 한다
    assert ((table["fast"] < table["slow"]).all())


def test_grid_search_filters_low_trade_counts(trending_df):
    grid = {"fast": [3, 5], "slow": [10, 20]}
    table = optimize.grid_search(trending_df, MACrossStrategy, grid, min_trades=5)
    assert (table["거래수"] >= 5).all()


def test_holdout_evaluates_on_unseen_data(trending_df):
    grid = {"fast": [3, 5, 10], "slow": [20, 40]}
    out = optimize.holdout_test(trending_df, MACrossStrategy, grid, train_ratio=0.7)

    assert set(out["best_params"]) == {"fast", "slow"}
    # 검증 결과는 훈련 구간 이후 데이터로만 계산돼야 한다
    train_end = trending_df.index[int(len(trending_df) * 0.7) - 1]
    assert out["test"].equity.index[0] > train_end
    assert out["degradation"] == pytest.approx(
        out["train"].total_return - out["test"].total_return
    )


def test_walk_forward_produces_one_row_per_split(trending_df):
    grid = {"fast": [3, 5], "slow": [20, 40]}
    table = optimize.walk_forward(trending_df, MACrossStrategy, grid, n_splits=4)
    assert len(table) == 4
    assert table["검증기간"].is_unique


def test_walk_forward_rejects_windows_that_are_too_small(trending_df):
    with pytest.raises(ValueError, match="의미가 없다"):
        optimize.walk_forward(trending_df, MACrossStrategy, {"fast": [3], "slow": [20]}, n_splits=30)
