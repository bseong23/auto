"""launchd 에이전트 일정 생성 검증.

일정이 틀리면 봉 마감과 무관한 시각에 판단한다 = 백테스트와 다른 전략이 된다.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "make_agent", Path(__file__).resolve().parent.parent / "scripts" / "make_agent.py"
)
make_agent = importlib.util.module_from_spec(spec)
sys.modules["make_agent"] = make_agent
spec.loader.exec_module(make_agent)


def test_daily_runs_just_after_the_upbit_close():
    """업비트 일봉은 09:00 KST 마감. 확정봉이 올라올 시간을 조금 준다."""
    assert make_agent.schedule_for("day") == [{"Hour": 9, "Minute": 1}]


def test_weekly_runs_on_monday():
    assert make_agent.schedule_for("week") == [{"Weekday": 1, "Hour": 9, "Minute": 1}]


def test_monthly_runs_on_the_first():
    assert make_agent.schedule_for("month") == [{"Day": 1, "Hour": 9, "Minute": 1}]


@pytest.mark.parametrize(
    "interval,expected_count",
    [("minute5", 12), ("minute15", 4), ("minute30", 2), ("minute60", 24), ("minute240", 6)],
)
def test_minute_intervals_cover_the_whole_day(interval, expected_count):
    assert len(make_agent.schedule_for(interval)) == expected_count


def test_sub_hour_schedules_repeat_every_hour():
    """Hour 를 안 주면 launchd 는 매시간 그 분에 실행한다."""
    slots = make_agent.schedule_for("minute15")
    assert all("Hour" not in slot for slot in slots)
    assert [s["Minute"] for s in slots] == [1, 16, 31, 46]


def test_multi_hour_schedules_follow_the_upbit_grid():
    """업비트 240분봉 마감은 01·05·09·13·17·21시 — 자정 기준이 아니다."""
    hours = [s["Hour"] for s in make_agent.schedule_for("minute240")]
    assert hours == [1, 5, 9, 13, 17, 21]


def test_unknown_interval_is_rejected():
    with pytest.raises(ValueError, match="모르는 interval"):
        make_agent.schedule_for("minute7")


def test_agent_never_runs_at_load():
    """켜자마자 돌면 봉 마감과 무관한 시각에 판단하게 된다."""
    args = type("A", (), {
        "ticker": "KRW-BTC", "interval": "day", "strategy": "ma",
        "order_krw": 5500, "stop_atr": None, "trailing": False, "live": False,
    })()
    assert make_agent.build(args)["RunAtLoad"] is False


def test_live_flag_is_carried_into_the_command():
    args = type("A", (), {
        "ticker": "KRW-BTC", "interval": "day", "strategy": "ma",
        "order_krw": 5500, "stop_atr": 2.0, "trailing": True, "live": True,
    })()
    command = make_agent.build(args)["ProgramArguments"]
    assert "--live" in command and "--trailing" in command
    assert command[command.index("--stop-atr") + 1] == "2.0"
