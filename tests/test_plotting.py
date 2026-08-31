"""그래프 생성 — 그림 내용까지는 검증할 수 없으니, 안 죽고 파일이 나오는지 본다."""
import numpy as np
import pandas as pd
import pytest

from upbit.backtest import run_backtest
from upbit.plotting import plot_comparison, plot_result, use_korean_font
from upbit.risk import RiskRules
from upbit.strategies import BuyAndHoldStrategy, MACrossStrategy


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    close = 100_000_000 + rng.normal(0, 1_500_000, 300).cumsum()
    return pd.DataFrame(
        {"open": close, "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": np.ones(300)},
        index=pd.date_range("2024-01-01", periods=300, freq="D"),
    )


def test_single_result_chart_is_written(tmp_path, df):
    result = run_backtest(df, MACrossStrategy(5, 20))
    out = plot_result(result, df, tmp_path / "one.png")
    assert out.exists() and out.stat().st_size > 10_000  # 빈 파일이 아님


def test_comparison_chart_is_written(tmp_path, df):
    results = [
        run_backtest(df, MACrossStrategy(5, 20)),
        run_backtest(df, BuyAndHoldStrategy()),
    ]
    out = plot_comparison(results, tmp_path / "compare.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_creates_missing_directories(tmp_path, df):
    result = run_backtest(df, BuyAndHoldStrategy())
    out = plot_result(result, df, tmp_path / "a" / "b" / "deep.png")
    assert out.exists()


def test_handles_a_strategy_that_never_trades(tmp_path, df):
    """거래가 0건이어도 그래프는 그려져야 한다 (마커만 없음)."""
    result = run_backtest(df, MACrossStrategy(5, 20), risk=RiskRules(stop_loss_pct=0.001))
    out = plot_result(result, df, tmp_path / "notrade.png")
    assert out.exists()


def test_handles_an_open_position_without_exit_marker(tmp_path, df):
    result = run_backtest(df, BuyAndHoldStrategy())
    assert any(t.is_open for t in result.trades)
    assert plot_result(result, df, tmp_path / "open.png").exists()


def test_korean_font_lookup_does_not_raise():
    use_korean_font()  # 폰트가 없는 환경이면 None을 돌려주고 넘어가야 한다
