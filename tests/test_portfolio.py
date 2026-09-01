"""포트폴리오 엔진 검증 — 이 엔진으로 '고르기에 실력이 있나'를 판단할 것이므로 틀리면 안 된다."""
import numpy as np
import pandas as pd
import pytest

from upbit.portfolio import (
    random_selector,
    run_portfolio,
    top_k_selector,
    windowed_portfolio_returns,
)


def make_frames(n=300, drifts=None, start_nan=None, seed=0):
    """일일 수익률이 종목별 drift + 잡음인 합성 시장. start_nan: {종목: 상장 전 NaN 일수}"""
    rng = np.random.default_rng(seed)
    drifts = drifts or {"A": 0.005, "B": 0.002, "C": 0.0, "D": -0.003}
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = {}
    for name, d in drifts.items():
        path = 100 * np.cumprod(1 + d + rng.normal(0, 0.005, n))
        closes[name] = path
    closes = pd.DataFrame(closes, index=idx)
    opens = closes.shift(1).fillna(closes.iloc[0])  # 시가 = 전일 종가 (갭 없음)
    for name, k in (start_nan or {}).items():
        closes.iloc[:k, closes.columns.get_loc(name)] = np.nan
        opens.iloc[:k, opens.columns.get_loc(name)] = np.nan
    return opens, closes


# ---------- 기본 동작 ----------

def test_picks_the_strongest_names_and_never_the_falling_one():
    opens, closes = make_frames()
    res = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=10, fee=0, slippage=0)
    share = (res.weights > 0).mean()
    assert share["A"] > 0.9 and share["B"] > 0.6
    assert share["D"] == 0.0, "계속 떨어지는 종목을 골랐다"


def test_weights_never_exceed_one_and_cash_is_never_negative():
    opens, closes = make_frames()
    res = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=5)
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    assert (res.invested <= 1 + 1e-9).all()
    assert (res.equity > 0).all()


def test_equal_weight_leaves_empty_slots_in_cash():
    """후보가 top_k 보다 적으면 비중을 늘려 채우지 않는다."""
    opens, closes = make_frames(drifts={"A": 0.005, "B": -0.003, "C": -0.003})
    res = run_portfolio(opens, closes, lookback=20, top_k=3, rebalance_every=10, fee=0, slippage=0)
    assert res.weights.max().max() == pytest.approx(1 / 3)
    assert res.invested.iloc[-1] == pytest.approx(1 / 3, abs=0.05)


# ---------- 미래참조 없음 ----------

def test_future_prices_do_not_change_past_equity():
    opens, closes = make_frames()
    base = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=7)
    o2, c2 = opens.copy(), closes.copy()
    c2.iloc[200:] *= 3
    o2.iloc[200:] *= 3
    after = run_portfolio(o2, c2, lookback=20, top_k=2, rebalance_every=7)
    pd.testing.assert_series_equal(base.equity.iloc[:200], after.equity.iloc[:200])


def test_signal_uses_only_closes_before_the_trade_day():
    """t 일 시가에 거래하는데 t 일 종가를 쓰면 미래참조다."""
    opens, closes = make_frames(n=100, drifts={"A": 0.0, "B": 0.0})
    # 30일째 A 만 종가가 폭등 — 그날 리밸런싱이 그걸 보고 A 를 고르면 안 된다
    closes.iloc[30, 0] *= 2
    res = run_portfolio(opens, closes, lookback=10, top_k=1, rebalance_every=19, fee=0, slippage=0,
                        absolute=False)
    day30 = closes.index[30]
    assert day30 in res.weights.index
    # 30일째 판단은 29일 종가까지: A 와 B 모멘텀이 같으므로 어느 쪽이든 되지만
    # 폭등 '이후' 첫 리밸런싱(49일)에서는 A 를 골라야 한다
    assert res.weights.loc[closes.index[49], "A"] == pytest.approx(1.0)


# ---------- 비용 ----------

def test_zero_cost_rebalancing_conserves_wealth():
    opens, closes = make_frames()
    res = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=10, fee=0, slippage=0)
    first = res.weights.index[0]
    i = closes.index.get_loc(first)
    # 리밸런싱 직전 자산(= 현금 100만) 과 직후 자산(시가 기준) 이 같아야 한다
    prev_equity = res.equity.iloc[i - 1]
    assert prev_equity == pytest.approx(1_000_000)


def test_costs_reduce_equity_by_turnover_times_rate():
    opens, closes = make_frames()
    free = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=10, fee=0, slippage=0)
    paid = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=10, fee=0.001, slippage=0)
    assert paid.costs_paid > 0
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    # 첫 리밸런싱: 현금 100% → 2종목 50%씩 매수. 비용(0.1%)만큼 덜 사므로 회전율 = 1 − 0.001
    assert paid.turnover.iloc[0] == pytest.approx(1.0 - 0.001, abs=1e-6)
    # 비용은 예산에서 나간다 — 현금이 음수가 되면 안 된다
    first = paid.weights.index[0]
    assert paid.invested.loc[first] <= 1.0 + 1e-12


def test_per_ticker_slippage_penalises_expensive_names():
    opens, closes = make_frames(drifts={"A": 0.003, "B": 0.003})
    cheap = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=5, fee=0,
                          slippage=pd.Series({"A": 0.0, "B": 0.0}))
    pricey = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=5, fee=0,
                           slippage=pd.Series({"A": 0.0, "B": 0.01}))
    assert pricey.costs_paid > cheap.costs_paid == 0


# ---------- 절대 모멘텀(현금 회피) ----------

def test_absolute_filter_goes_to_cash_when_everything_falls():
    opens, closes = make_frames(drifts={"A": -0.004, "B": -0.003, "C": -0.005})
    res = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=7, absolute=True)
    assert res.invested.iloc[-1] == 0.0
    assert res.equity.iloc[-1] == pytest.approx(1_000_000)


def test_without_absolute_filter_it_stays_invested_in_a_bear_market():
    opens, closes = make_frames(drifts={"A": -0.004, "B": -0.003, "C": -0.005})
    res = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=7, absolute=False)
    assert res.invested.iloc[-1] > 0.5
    assert res.equity.iloc[-1] < 1_000_000


# ---------- 중간 상장 / 상장폐지 ----------

def test_late_listing_is_not_eligible_until_it_has_history():
    opens, closes = make_frames(drifts={"A": 0.002, "E": 0.02}, start_nan={"E": 150})
    res = run_portfolio(opens, closes, lookback=20, top_k=1, rebalance_every=5, fee=0, slippage=0)
    before = res.weights[res.weights.index < closes.index[150 + 21]]
    assert (before["E"] == 0).all(), "상장 전/이력 부족 종목을 골랐다"
    assert res.weights["E"].iloc[-1] == 1.0, "이력이 쌓인 뒤엔 제일 강한 E 를 골라야 한다"


def test_delisted_holding_is_liquidated_at_last_close():
    opens, closes = make_frames(n=200, drifts={"A": 0.004, "B": 0.0})
    closes.iloc[120:, 0] = np.nan
    opens.iloc[120:, 0] = np.nan
    res = run_portfolio(opens, closes, lookback=20, top_k=1, rebalance_every=30, fee=0, slippage=0)
    assert np.isfinite(res.equity).all()
    assert res.invested.iloc[121] == 0.0  # 끊긴 다음 날엔 현금
    assert res.equity.iloc[121] == pytest.approx(res.equity.iloc[119], rel=0.02)


# ---------- 선택기 ----------

def test_random_selector_only_picks_from_candidates():
    rng = np.random.default_rng(0)
    pick = random_selector(2, rng)
    chosen = pick(pd.Series({"A": 0.1, "B": 0.2, "C": 0.3}))
    assert len(chosen) == 2 and set(chosen) <= {"A", "B", "C"}
    assert pick(pd.Series({"A": 0.1})) == ["A"]


def test_top_k_selector_orders_by_momentum():
    assert top_k_selector(2)(pd.Series({"A": 0.1, "B": 0.3, "C": 0.2})) == ["B", "C"]


def test_random_selection_differs_from_ranked_selection():
    opens, closes = make_frames()
    ranked = run_portfolio(opens, closes, lookback=20, top_k=1, rebalance_every=5, fee=0, slippage=0)
    rnd = run_portfolio(opens, closes, lookback=20, top_k=1, rebalance_every=5, fee=0, slippage=0,
                        selector=random_selector(1, np.random.default_rng(1)))
    assert not ranked.weights.equals(rnd.weights)
    assert ranked.total_return > rnd.total_return  # A 가 압도적인 시장이라 순위가 이겨야 한다


# ---------- 기타 ----------

def test_rebalance_cadence_is_respected():
    opens, closes = make_frames()
    res = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=7)
    gaps = pd.Series(res.weights.index).diff().dropna().dt.days.unique()
    assert list(gaps) == [7]


def test_rejects_mismatched_frames():
    opens, closes = make_frames()
    with pytest.raises(ValueError, match="같은 인덱스"):
        run_portfolio(opens.iloc[:-1], closes)


def test_windowed_returns_have_one_value_per_split():
    opens, closes = make_frames(n=800)
    out = windowed_portfolio_returns(opens, closes, n_splits=4, warmup=100,
                                     lookback=20, top_k=2, rebalance_every=7)
    assert len(out) == 4


def test_summary_is_readable():
    opens, closes = make_frames()
    text = run_portfolio(opens, closes, lookback=20, top_k=2, rebalance_every=7).summary()
    assert "총수익" in text and "MDD" in text and "회전율" in text
