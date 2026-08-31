"""손절/익절 검증.

손절은 백테스트에서 결과를 제일 쉽게 부풀릴 수 있는 지점이다:
갭하락을 손절가에 체결했다고 치면 손실이 마법처럼 줄어든다. 여기서 그걸 막는다.
"""
import numpy as np
import pandas as pd
import pytest

from upbit.backtest import run_backtest
from upbit.risk import OpenPosition, RiskRules
from upbit.strategies.base import Strategy


class FixedPositions(Strategy):
    name = "고정"

    def __init__(self, seq):
        self.seq = list(seq)

    def generate_positions(self, df):
        return pd.Series(self.seq, index=df.index, dtype=int)


def bars(rows):
    """rows: (open, high, low, close) 리스트 — 봉을 직접 그린다."""
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
         "close": arr[:, 3], "volume": np.ones(len(arr))},
        index=pd.date_range("2024-01-01", periods=len(arr), freq="D"),
    )


# ---------- 설정 검증 ----------

def test_rejects_out_of_range_percentages():
    with pytest.raises(ValueError, match="0과 1 사이"):
        RiskRules(stop_loss_pct=1.5)
    with pytest.raises(ValueError, match="0과 1 사이"):
        RiskRules(take_profit_pct=0)


def test_rejects_non_positive_atr_multiple():
    with pytest.raises(ValueError, match="양수"):
        RiskRules(atr_multiple=0)


def test_trailing_requires_a_stop():
    with pytest.raises(ValueError, match="trailing"):
        RiskRules(trailing=True, take_profit_pct=0.1)


def test_inactive_rules_change_nothing():
    rules = RiskRules()
    assert not rules.is_active and not rules.needs_atr
    assert rules.stop_price(100, 5) is None
    assert rules.describe() == "규칙 없음"


def test_tighter_stop_wins_when_both_rules_are_set():
    """% 손절과 ATR 손절이 둘 다 있으면 더 가까운(높은) 쪽을 쓴다."""
    rules = RiskRules(stop_loss_pct=0.10, atr_multiple=2.0)   # 90 vs 100-2*3=94
    assert rules.stop_price(100, 3.0) == pytest.approx(94.0)
    rules = RiskRules(stop_loss_pct=0.02, atr_multiple=2.0)   # 98 vs 94
    assert rules.stop_price(100, 3.0) == pytest.approx(98.0)


def test_atr_rule_falls_back_to_pct_when_atr_is_nan():
    rules = RiskRules(stop_loss_pct=0.05, atr_multiple=2.0)
    assert rules.stop_price(100, float("nan")) == pytest.approx(95.0)


# ---------- 체결가 정직성 ----------

def test_intrabar_touch_fills_at_the_stop_price():
    pos = OpenPosition(entry_price=100, stop=95, target=None, high_water=100)
    assert pos.check_exit(bar_open=100, bar_low=94, bar_high=101) == (95, "손절")


def test_gap_down_fills_at_the_open_not_the_stop():
    """갭하락은 손절선을 건너뛴다. 손절가에 체결했다고 치면 손실이 축소된다."""
    pos = OpenPosition(entry_price=100, stop=95, target=None, high_water=100)
    fill, reason = pos.check_exit(bar_open=80, bar_low=78, bar_high=85)
    assert fill == 80 and reason == "손절(갭)"


def test_stop_wins_when_stop_and_target_hit_in_the_same_bar():
    """봉 안에서 어느 쪽이 먼저였는지 모르므로 보수적으로 손절 처리."""
    pos = OpenPosition(entry_price=100, stop=95, target=110, high_water=100)
    fill, reason = pos.check_exit(bar_open=100, bar_low=94, bar_high=111)
    assert reason == "손절" and fill == 95


def test_no_exit_when_neither_level_is_touched():
    pos = OpenPosition(entry_price=100, stop=95, target=110, high_water=100)
    assert pos.check_exit(bar_open=100, bar_low=96, bar_high=109) is None


# ---------- 추적 손절 ----------

def test_trailing_stop_rises_but_never_falls():
    rules = RiskRules(stop_loss_pct=0.05, trailing=True)
    pos = OpenPosition(entry_price=100, stop=95, target=None, high_water=100)

    pos.update_trailing(bar_high=120, atr_value=None, rules=rules)
    assert pos.stop == pytest.approx(114.0)   # 120 * 0.95

    pos.update_trailing(bar_high=105, atr_value=None, rules=rules)
    assert pos.stop == pytest.approx(114.0)   # 내려가지 않는다


def test_trailing_does_not_use_the_current_bar_high_for_the_current_decision():
    """이번 봉 고점으로 손절선을 올린 뒤 같은 봉에서 판정하면 미래참조다.

    진입 100, 손절 95. 봉이 (시가100, 고가120, 저가96) 이면
    - 올바름: 손절선 95 vs 저가 96 → 청산 없음. 다음 봉부터 손절선 114.
    - 미래참조: 고가120으로 손절선을 114로 먼저 올리면 저가 96에 걸려 청산 판정.
    """
    df = bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),   # 여기서 진입
        (100, 120, 96, 118),    # 크게 올랐다 내려온 봉
        (118, 120, 118, 119),
    ])
    result = run_backtest(
        df, FixedPositions([1, 1, 1, 1]), fee=0.0, slippage=0.0,
        risk=RiskRules(stop_loss_pct=0.05, trailing=True),
    )
    assert result.num_trades == 0, "이번 봉 고점으로 올린 손절선에 같은 봉에서 걸리면 안 된다"


def test_trailing_stop_locks_in_profit_on_the_next_bar():
    df = bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),   # 진입
        (100, 120, 100, 120),   # 급등 → 손절선 114로 상향
        (119, 119, 110, 112),   # 다음 봉에서 114 하향 이탈 → 청산
    ])
    result = run_backtest(
        df, FixedPositions([1, 1, 1, 1]), fee=0.0, slippage=0.0,
        risk=RiskRules(stop_loss_pct=0.05, trailing=True),
    )
    assert result.num_trades == 1
    trade = result.closed_trades[0]
    assert trade.exit_price == pytest.approx(114.0)
    assert trade.return_pct == pytest.approx(0.14)


# ---------- 백테스터 통합 ----------

def test_stop_beats_the_strategy_signal():
    """전략이 '계속 보유'라고 해도 손절이 이긴다."""
    df = bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),   # 진입
        (100, 101, 90, 92),     # 저가 90 → -5% 손절선 95 이탈
    ])
    result = run_backtest(
        df, FixedPositions([1, 1, 1]), fee=0.0, slippage=0.0,
        risk=RiskRules(stop_loss_pct=0.05),
    )
    assert result.num_trades == 1
    assert result.closed_trades[0].exit_reason == "손절"
    assert result.closed_trades[0].return_pct == pytest.approx(-0.05)


def test_no_reentry_until_the_strategy_goes_flat_first():
    """손절 직후 다음 봉에 바로 다시 사면 계속 얻어맞는다."""
    df = bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),   # 진입
        (100, 101, 90, 92),     # 손절
        (92, 95, 91, 94),       # 전략은 여전히 보유 → 재진입 막혀야 함
        (94, 96, 93, 95),
    ])
    result = run_backtest(
        df, FixedPositions([1, 1, 1, 1, 1]), fee=0.0, slippage=0.0,
        risk=RiskRules(stop_loss_pct=0.05),
    )
    assert result.num_trades == 1
    assert len(result.trades) == 1, "손절 후 재진입이 발생했다"


def test_reentry_allowed_after_strategy_returns_to_cash():
    df = bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),   # 진입
        (100, 101, 90, 92),     # 손절
        (92, 95, 91, 94),       # 전략 현금 전환 → 차단 해제
        (94, 96, 93, 95),       # 재진입
        (95, 97, 94, 96),
    ])
    result = run_backtest(
        df, FixedPositions([1, 1, 0, 1, 1, 1]), fee=0.0, slippage=0.0,
        risk=RiskRules(stop_loss_pct=0.05),
    )
    assert len(result.trades) == 2, "현금으로 돌아온 뒤에는 재진입이 가능해야 한다"


def test_take_profit_exit_is_recorded():
    df = bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),   # 진입
        (100, 115, 99, 114),    # 고가 115 → +10% 익절선 110 도달
    ])
    result = run_backtest(
        df, FixedPositions([1, 1, 1]), fee=0.0, slippage=0.0,
        risk=RiskRules(take_profit_pct=0.10),
    )
    assert result.closed_trades[0].exit_reason == "익절"
    assert result.closed_trades[0].return_pct == pytest.approx(0.10)


def test_atr_stop_adapts_to_volatility():
    """변동성이 크면 손절선이 멀어져 쉽게 안 털린다."""
    calm = [(100, 101, 99, 100)] * 30           # ATR ≈ 2 로 수렴
    df = bars(calm + [(100, 100, 100, 100),     # 여기서 진입 (시가 100)
                      (100, 101, 96, 97)])      # 저가 96 — 손절선에 걸리나?
    seq = [0] * 29 + [1] * 3

    tight = run_backtest(df, FixedPositions(seq), fee=0.0, slippage=0.0,
                         risk=RiskRules(atr_multiple=1.0))
    loose = run_backtest(df, FixedPositions(seq), fee=0.0, slippage=0.0,
                         risk=RiskRules(atr_multiple=10.0))
    assert tight.num_trades == 1, "좁은 ATR 손절은 걸려야 한다"
    assert loose.num_trades == 0, "넓은 ATR 손절은 안 걸려야 한다"


def test_result_records_the_rules_used():
    df = bars([(100, 100, 100, 100)] * 3)
    result = run_backtest(df, FixedPositions([0, 0, 0]),
                          risk=RiskRules(atr_multiple=2.0, trailing=True))
    assert "ATR14×2" in result.risk.describe()
    assert "손절규칙" in result.summary()


def test_backtest_without_risk_is_unchanged():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 120, 80, 110)])
    plain = run_backtest(df, FixedPositions([1, 1, 1]), fee=0.0, slippage=0.0)
    explicit = run_backtest(df, FixedPositions([1, 1, 1]), fee=0.0, slippage=0.0,
                            risk=RiskRules())
    pd.testing.assert_series_equal(plain.equity, explicit.equity)
