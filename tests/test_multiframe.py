"""다중 시간프레임 엔진 검증.

이 엔진의 결과로 "손절을 자주 확인할 가치가 있나"를 판단할 것이므로,
엔진이 틀리면 결론도 틀린다.
"""
import numpy as np
import pandas as pd
import pytest

from upbit.backtest import run_backtest
from upbit.multiframe import StopCheck, align_signals, run_multiframe_backtest
from upbit.risk import RiskRules
from upbit.strategies.base import Strategy


class FixedPositions(Strategy):
    name = "고정"

    def __init__(self, seq):
        self.seq = list(seq)

    def generate_positions(self, df):
        return pd.Series(self.seq, index=df.index, dtype=int)


def hourly_frame(rows, start="2024-01-01 09:00"):
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
         "close": arr[:, 3], "volume": np.ones(len(arr))},
        index=pd.date_range(start, periods=len(arr), freq="h"),
    )


def daily_from(hourly: pd.DataFrame) -> pd.DataFrame:
    """60분봉을 09:00 기준 일봉으로 묶는다 (업비트와 같은 경계)."""
    grouped = hourly.resample("24h", origin=hourly.index[0])
    return pd.DataFrame({
        "open": grouped["open"].first(), "high": grouped["high"].max(),
        "low": grouped["low"].min(), "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(),
    }).dropna()


# ---------- 신호 정렬 ----------

def test_signal_becomes_effective_only_after_its_bar_closes():
    """일봉 D의 신호를 D일 장중에 쓰면 미래참조다."""
    daily = pd.Series([0, 1], index=pd.to_datetime(["2024-01-01 09:00", "2024-01-02 09:00"]))
    hourly = pd.date_range("2024-01-01 09:00", "2024-01-03 09:00", freq="h")

    aligned = align_signals(daily, hourly, pd.Timedelta(days=1))

    assert aligned.loc["2024-01-03 08:00"] == 0   # 아직 그 일봉이 안 끝났다
    assert aligned.loc["2024-01-03 09:00"] == 1   # 마감된 뒤부터 유효


def test_alignment_never_looks_ahead_when_later_signals_change():
    daily_index = pd.date_range("2024-01-01 09:00", periods=6, freq="D")
    hourly = pd.date_range("2024-01-01 09:00", periods=6 * 24, freq="h")

    base = align_signals(pd.Series([0, 1, 0, 1, 0, 1], index=daily_index),
                         hourly, pd.Timedelta(days=1))
    changed = align_signals(pd.Series([0, 1, 0, 0, 1, 0], index=daily_index),
                            hourly, pd.Timedelta(days=1))

    cut = hourly.get_loc(pd.Timestamp("2024-01-04 09:00"))
    pd.testing.assert_series_equal(base.iloc[:cut], changed.iloc[:cut])


# ---------- 손절 없으면 일봉 백테스트와 같아야 한다 ----------

def test_without_stops_it_matches_the_plain_daily_backtest():
    """손절이 없으면 체결 시점이 일봉 시가와 같으므로 결과가 일치해야 한다."""
    rng = np.random.default_rng(5)
    n = 24 * 40
    close = 100.0 + rng.normal(0, 0.5, n).cumsum()
    hourly = hourly_frame(np.column_stack([close, close * 1.002, close * 0.998, close]))
    daily = daily_from(hourly)

    seq = [i % 6 < 3 for i in range(len(daily))]
    plain = run_backtest(daily, FixedPositions(seq), fee=0.0, slippage=0.0)
    multi = run_multiframe_backtest(daily, hourly, FixedPositions(seq),
                                    fee=0.0, slippage=0.0)

    assert multi.num_trades == plain.num_trades
    assert multi.total_return == pytest.approx(plain.total_return, rel=1e-9)


# ---------- 손절 확인 주기 ----------

def test_stop_is_only_evaluated_at_check_times():
    """확인 시각이 아니면 손절선 아래로 내려가도 모른 채 지나간다."""
    # 09:00 진입, 10~11시에 급락했다가 12시에 회복
    rows = [(100, 100, 100, 100)] * 24            # 1일차
    rows += [(100, 100, 100, 100)]                # 2일차 09:00 — 진입
    rows += [(80, 80, 80, 80), (80, 80, 80, 80)]  # 10,11시 급락
    rows += [(100, 100, 100, 100)] * 21           # 회복
    rows += [(100, 100, 100, 100)] * 24           # 3일차
    hourly = hourly_frame(rows)
    daily = daily_from(hourly)
    seq = [1] * len(daily)

    hourly_check = run_multiframe_backtest(
        daily, hourly, FixedPositions(seq), risk=RiskRules(stop_loss_pct=0.05),
        stop_check=StopCheck(every=1), fee=0.0, slippage=0.0)
    daily_check = run_multiframe_backtest(
        daily, hourly, FixedPositions(seq), risk=RiskRules(stop_loss_pct=0.05),
        stop_check=StopCheck(at_hour=9), fee=0.0, slippage=0.0)

    assert hourly_check.num_trades == 1, "매시간 확인이면 급락을 잡아야 한다"
    assert daily_check.num_trades == 0, "하루 1회 확인이면 회복한 급락을 못 본다"


def test_daily_check_only_fires_at_the_configured_hour():
    rows = [(100, 100, 100, 100)] * 25 + [(80, 80, 80, 80)] * 47
    hourly = hourly_frame(rows)
    daily = daily_from(hourly)

    result = run_multiframe_backtest(
        daily, hourly, FixedPositions([1] * len(daily)),
        risk=RiskRules(stop_loss_pct=0.05), stop_check=StopCheck(at_hour=9),
        fee=0.0, slippage=0.0)

    assert result.num_trades == 1
    assert result.closed_trades[0].exit_time.hour == 9


def test_stop_fills_at_the_price_seen_not_at_the_stop_line():
    """실전은 손절선 가격을 보장받지 못한다 — 확인 시점 가격에 판다."""
    rows = [(100, 100, 100, 100)] * 25 + [(70, 70, 70, 70)] * 47
    hourly = hourly_frame(rows)
    daily = daily_from(hourly)

    result = run_multiframe_backtest(
        daily, hourly, FixedPositions([1] * len(daily)),
        risk=RiskRules(stop_loss_pct=0.05), stop_check=StopCheck(every=1),
        fee=0.0, slippage=0.0)

    assert result.closed_trades[0].exit_price == pytest.approx(70.0)  # 95가 아니다


def test_no_reentry_until_the_strategy_goes_flat():
    rows = [(100, 100, 100, 100)] * 25 + [(80, 80, 80, 80)] * 71
    hourly = hourly_frame(rows)
    daily = daily_from(hourly)

    result = run_multiframe_backtest(
        daily, hourly, FixedPositions([1] * len(daily)),
        risk=RiskRules(stop_loss_pct=0.05), stop_check=StopCheck(every=1),
        fee=0.0, slippage=0.0)

    assert len(result.trades) == 1, "손절 후 재진입했다"


def test_trailing_stop_rises_between_checks():
    rows = [(100, 100, 100, 100)] * 25          # 진입
    rows += [(150, 150, 150, 150)] * 24         # 급등 → 손절선 상향
    rows += [(140, 140, 140, 140)] * 24         # 142.5 아래로 하락 → 청산
    hourly = hourly_frame(rows)
    daily = daily_from(hourly)

    result = run_multiframe_backtest(
        daily, hourly, FixedPositions([1] * len(daily)),
        risk=RiskRules(stop_loss_pct=0.05, trailing=True),
        stop_check=StopCheck(every=1), fee=0.0, slippage=0.0)

    assert result.num_trades == 1
    assert result.closed_trades[0].exit_price == pytest.approx(140.0)


def test_check_descriptions_are_readable():
    assert StopCheck(every=1).describe() == "매 봉"
    assert StopCheck(every=24).describe() == "24봉마다"
    assert "09시" in StopCheck(at_hour=9).describe()


def test_rejects_too_short_execution_data():
    tiny = hourly_frame([(1, 1, 1, 1)])
    with pytest.raises(ValueError, match="최소 2개"):
        run_multiframe_backtest(tiny, tiny, FixedPositions([1]))
