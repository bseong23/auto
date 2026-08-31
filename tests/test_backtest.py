"""백테스터가 거짓말을 안 하는지 검증한다.

여기서 제일 중요한 건 '미래참조 없음'과 '수수료 반영'.
이 두 개가 틀리면 백테스트 결과 전체가 무의미하다.
"""
import numpy as np
import pandas as pd
import pytest

from upbite.backtest import compare, run_backtest
from upbite.strategies import BuyAndHoldStrategy
from upbite.strategies.base import Strategy


class FixedPositions(Strategy):
    """테스트용 — 미리 정한 포지션 시퀀스를 그대로 낸다."""

    name = "고정"

    def __init__(self, seq):
        self.seq = list(seq)

    def generate_positions(self, df):
        return pd.Series(self.seq, index=df.index, dtype=int)


def make_df(opens, closes=None, index=None):
    closes = opens if closes is None else closes
    n = len(opens)
    idx = index if index is not None else pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": np.array(opens, dtype=float),
            "high": np.maximum(opens, closes) * 1.01,
            "low": np.minimum(opens, closes) * 0.99,
            "close": np.array(closes, dtype=float),
            "volume": np.ones(n),
        },
        index=idx,
    )


# ---------- 체결 타이밍 (미래참조 방지) ----------

def test_entry_fills_at_next_bar_open_not_signal_bar_close():
    """t봉 종가에 신호 → t+1봉 시가에 체결. 같은 봉 종가 체결이면 안 된다."""
    df = make_df(opens=[100, 100, 200, 200, 400], closes=[100, 150, 250, 300, 500])
    #                   i=0  i=1  i=2  i=3  i=4
    res = run_backtest(df, FixedPositions([0, 1, 1, 0, 0]), fee=0.0, slippage=0.0)

    assert len(res.closed_trades) == 1
    trade = res.closed_trades[0]
    assert trade.entry_price == pytest.approx(200.0)   # open[2], close[1]=150 아님
    assert trade.exit_price == pytest.approx(400.0)    # open[4], close[3]=300 아님
    assert trade.return_pct == pytest.approx(1.0)      # 200 → 400


def test_signal_on_final_bar_never_executes():
    """마지막 봉에서 신호가 떠도 체결할 다음 봉이 없으므로 거래 0건."""
    df = make_df(opens=[100, 100, 100, 100])
    res = run_backtest(df, FixedPositions([0, 0, 0, 1]), fee=0.0, slippage=0.0)
    assert res.trades == []
    assert res.total_return == pytest.approx(0.0)


def test_future_price_changes_cannot_alter_past_equity():
    """뒤쪽 가격을 바꿔도 앞쪽 자산곡선은 그대로여야 한다."""
    rng = np.random.default_rng(7)
    prices = 100 + rng.normal(0, 1, 80).cumsum()
    df = make_df(prices, prices)
    seq = [i % 7 < 3 for i in range(80)]

    base = run_backtest(df, FixedPositions(seq), fee=0.0, slippage=0.0)

    tampered = df.copy()
    tampered.iloc[50:] *= 5.0
    after = run_backtest(tampered, FixedPositions(seq), fee=0.0, slippage=0.0)

    pd.testing.assert_series_equal(base.equity.iloc[:50], after.equity.iloc[:50])


# ---------- 수수료 / 슬리피지 ----------

def test_fees_and_slippage_reduce_return_exactly():
    df = make_df(opens=[100, 100, 200, 200, 400], closes=[100, 150, 250, 300, 500])
    fee, slip = 0.0005, 0.0005
    res = run_backtest(df, FixedPositions([0, 1, 1, 0, 0]), fee=fee, slippage=slip)

    buy_fill = 200 * (1 + slip)
    sell_fill = 400 * (1 - slip)
    expected = (1 - fee) ** 2 * sell_fill / buy_fill - 1
    assert res.closed_trades[0].return_pct == pytest.approx(expected)
    assert res.total_fees_paid > 0


def test_churning_on_flat_market_bleeds_money_through_fees():
    """가격이 전혀 안 움직여도 매매를 반복하면 수수료로 잃는다."""
    df = make_df(opens=[100.0] * 21)
    seq = [i % 2 for i in range(21)]
    res = run_backtest(df, FixedPositions(seq), fee=0.0005, slippage=0.0)

    assert res.total_return < 0
    assert res.num_trades >= 4
    assert res.win_rate == pytest.approx(0.0)  # 수수료 때문에 전부 손실


def test_zero_fee_flat_market_is_break_even():
    df = make_df(opens=[100.0] * 21)
    res = run_backtest(df, FixedPositions([i % 2 for i in range(21)]), fee=0.0, slippage=0.0)
    assert res.total_return == pytest.approx(0.0)


# ---------- 지표 계산 ----------

def test_buy_and_hold_equals_price_move_from_first_open():
    df = make_df(opens=[100, 110, 120, 130], closes=[105, 115, 125, 135])
    res = run_backtest(df, BuyAndHoldStrategy(), fee=0.0, slippage=0.0)
    # open[1]에 매수 → close[-1]까지 보유
    assert res.total_return == pytest.approx(135 / 110 - 1)
    assert res.exposure == pytest.approx(1.0)


def test_mdd_matches_hand_calculation():
    df = make_df(opens=[100, 100, 200, 100, 150], closes=[100, 100, 200, 100, 150])
    res = run_backtest(df, BuyAndHoldStrategy(), fee=0.0, slippage=0.0)
    # 진입가 100 기준 자산: 100 → 200 → 100 → 150. 고점200 대비 저점100 = -50%
    assert res.mdd == pytest.approx(-0.5)


def test_win_rate_and_profit_factor():
    #        i:   0    1    2    3    4    5    6
    df = make_df(opens=[100, 100, 120, 120, 100, 100, 110],
                 closes=[100, 100, 120, 120, 100, 100, 110])
    # 거래1: open[2]=120 매수 → open[4]=100 매도 (손실)
    # 거래2: open[5]=100 매수 → open[6]=110 매도 (이익)
    res = run_backtest(df, FixedPositions([0, 1, 1, 0, 1, 0, 0]), fee=0.0, slippage=0.0)

    assert res.num_trades == 2
    assert res.win_rate == pytest.approx(0.5)
    returns = sorted(t.return_pct for t in res.closed_trades)
    assert returns[0] == pytest.approx(100 / 120 - 1)
    assert returns[1] == pytest.approx(110 / 100 - 1)
    assert res.profit_factor == pytest.approx(0.1 / (1 / 6))


def test_open_position_is_marked_to_market_but_not_counted_as_closed_trade():
    df = make_df(opens=[100, 100, 100], closes=[100, 100, 150])
    res = run_backtest(df, FixedPositions([1, 1, 1]), fee=0.0, slippage=0.0)
    assert res.num_trades == 0                    # 청산된 거래는 없고
    assert len(res.trades) == 1                   # 미청산 포지션은 기록되며
    assert res.trades[0].is_open
    assert res.total_return == pytest.approx(0.5)  # 평가익은 자산곡선에 반영


def test_cagr_is_consistent_with_total_return_over_one_year():
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    prices = np.linspace(100, 200, len(idx))
    df = make_df(prices, prices, index=idx)
    res = run_backtest(df, BuyAndHoldStrategy(), fee=0.0, slippage=0.0)
    assert res.cagr == pytest.approx(res.total_return, rel=0.02)


# ---------- 방어 ----------

def test_rejects_positions_outside_zero_one():
    df = make_df([100, 100, 100])
    with pytest.raises(ValueError, match="0 또는 1"):
        run_backtest(df, FixedPositions([0, 2, 1]))


def test_rejects_too_short_data():
    with pytest.raises(ValueError, match="최소 2개"):
        run_backtest(make_df([100]), BuyAndHoldStrategy())


def test_compare_sorts_by_return_descending():
    df = make_df([100, 110, 120, 130], [105, 115, 125, 135])
    results = [
        run_backtest(df, BuyAndHoldStrategy(), fee=0.0, slippage=0.0),
        run_backtest(df, FixedPositions([0, 0, 0, 0]), fee=0.0, slippage=0.0),
    ]
    table = compare(results)
    assert list(table["총수익률"]) == sorted(table["총수익률"], reverse=True)
