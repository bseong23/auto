"""봉 마감 시각 — 여기가 틀리면 백테스트와 다른 전략이 된다."""
from datetime import datetime, timedelta

import pytest

from upbit.schedule import KST, describe_next, next_close, seconds_until_next_close


def kst(*args):
    return datetime(*args, tzinfo=KST)


# ---------- 일봉: 업비트는 09:00 KST 마감 ----------

def test_daily_close_is_nine_am_kst():
    assert next_close("day", kst(2026, 8, 31, 15, 42)) == kst(2026, 9, 1, 9, 0)


def test_before_nine_am_the_close_is_today():
    assert next_close("day", kst(2026, 8, 31, 8, 42)) == kst(2026, 8, 31, 9, 0)


def test_exactly_at_close_moves_to_the_next_day():
    """마감 시각 정각이면 그 봉은 이미 닫혔다 — 다음 봉을 기다린다."""
    assert next_close("day", kst(2026, 8, 31, 9, 0)) == kst(2026, 9, 1, 9, 0)


# ---------- 분봉 ----------

@pytest.mark.parametrize(
    "interval,now,expected",
    [
        ("minute5", kst(2026, 8, 31, 10, 3), kst(2026, 8, 31, 10, 5)),
        ("minute5", kst(2026, 8, 31, 10, 5), kst(2026, 8, 31, 10, 10)),
        ("minute15", kst(2026, 8, 31, 10, 1), kst(2026, 8, 31, 10, 15)),
        ("minute60", kst(2026, 8, 31, 10, 30), kst(2026, 8, 31, 11, 0)),
        ("minute240", kst(2026, 8, 31, 10, 30), kst(2026, 8, 31, 12, 0)),
    ],
)
def test_minute_candles_close_on_the_boundary(interval, now, expected):
    assert next_close(interval, now) == expected


def test_minute_close_rolls_over_midnight():
    assert next_close("minute60", kst(2026, 8, 31, 23, 30)) == kst(2026, 9, 1, 0, 0)


# ---------- 주봉 / 월봉 ----------

def test_weekly_close_is_monday_nine_am():
    result = next_close("week", kst(2026, 9, 2, 10, 0))   # 수요일
    assert result.weekday() == 0 and result.hour == 9


def test_monthly_close_is_first_day_nine_am():
    assert next_close("month", kst(2026, 8, 31, 10, 0)) == kst(2026, 9, 1, 9, 0)


def test_monthly_close_rolls_over_the_year():
    assert next_close("month", kst(2026, 12, 15, 10, 0)) == kst(2027, 1, 1, 9, 0)


# ---------- 대기 시간 ----------

def test_wait_includes_settle_delay():
    """마감 직후엔 API에 확정봉이 아직 없을 수 있어 조금 기다린다."""
    from upbit.schedule import SETTLE_DELAY_SEC

    now = kst(2026, 8, 31, 8, 59, 0)
    assert seconds_until_next_close("day", now) == pytest.approx(60 + SETTLE_DELAY_SEC)


def test_wait_is_never_zero_or_negative():
    """0을 반환하면 바쁜 대기(busy loop)에 빠진다."""
    for minute in range(0, 60, 7):
        assert seconds_until_next_close("minute1", kst(2026, 8, 31, 10, minute)) >= 1.0


def test_unknown_interval_is_rejected():
    with pytest.raises(ValueError, match="모르는 interval"):
        next_close("minute7")


def test_describe_is_human_readable():
    text = describe_next("day", kst(2026, 8, 31, 8, 0))
    assert "2026-08-31 09:00 KST" in text and "후)" in text


def test_result_is_always_in_kst_regardless_of_local_timezone():
    """서버가 UTC에 있어도 업비트 기준으로 계산돼야 한다."""
    from datetime import timezone

    utc_now = datetime(2026, 8, 31, 0, 30, tzinfo=timezone.utc)  # KST 09:30
    assert next_close("day", utc_now) == kst(2026, 9, 1, 9, 0)
