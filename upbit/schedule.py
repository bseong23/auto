"""봉 마감 시각 계산 — 언제 판단할지.

`--every 3600` 처럼 상대 시간으로 자면, 봇을 켠 시각 기준으로 돈다.
일봉 전략을 15시에 켜면 매일 15시에 판단하는데, 그때 최신 확정봉은 6시간 전 것이다.
신호가 하루씩 밀린다. 백테스트는 "봉 마감 직후 판단"을 가정했으므로
**사실상 다른 전략이 된다.**

업비트 기준 — 모든 봉의 격자가 **09:00 KST(= 00:00 UTC)** 에 앵커돼 있다:
- 일봉/주봉/월봉은 09:00 KST 에 마감된다
- 분봉은 09:00 부터 봉 길이 간격으로 마감된다. 60분 이하 봉은 자정 기준과 같은 격자지만
  **240분봉은 01·05·09·13·17·21시**다 (자정 기준 00·04·08… 이 아니다)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

#: 업비트 기준 시간대. 로컬 시간대에 의존하면 서버 위치에 따라 결과가 달라진다.
KST = timezone(timedelta(hours=9))

#: 마감 직후엔 API에 아직 확정봉이 안 올라와 있을 수 있어 조금 기다린다
SETTLE_DELAY_SEC = 30

MINUTE_INTERVALS = {
    "minute1": 1, "minute3": 3, "minute5": 5, "minute10": 10,
    "minute15": 15, "minute30": 30, "minute60": 60, "minute240": 240,
}


def next_close(interval: str, now: datetime | None = None) -> datetime:
    """`now` 이후 가장 가까운 봉 마감 시각(KST)."""
    current = (now or datetime.now(KST)).astimezone(KST)

    if interval in MINUTE_INTERVALS:
        step = MINUTE_INTERVALS[interval]
        # 업비트 분봉 격자는 자정이 아니라 **09:00 KST(=00:00 UTC)** 에 앵커돼 있다.
        # 60분을 나누는 봉(1~60분)은 어느 쪽에 앵커해도 같지만, 240분봉은 다르다:
        # 자정 기준이면 00·04·08·12·16·20시, 실제는 01·05·09·13·17·21시다.
        # 이걸 틀리면 봉 마감 1시간 전에 깨어나 3시간 묵은 봉으로 판단하게 된다.
        anchor = current.replace(hour=9, minute=0, second=0, microsecond=0)
        if current < anchor:
            anchor -= timedelta(days=1)
        elapsed_min = int((current - anchor).total_seconds() // 60)
        next_slot = (elapsed_min // step + 1) * step
        return anchor + timedelta(minutes=next_slot)

    if interval == "day":
        today_close = current.replace(hour=9, minute=0, second=0, microsecond=0)
        return today_close if current < today_close else today_close + timedelta(days=1)

    if interval == "week":
        candidate = current.replace(hour=9, minute=0, second=0, microsecond=0)
        # 업비트 주봉은 월요일 09:00 에 새로 시작한다
        days_ahead = (0 - candidate.weekday()) % 7
        candidate += timedelta(days=days_ahead)
        return candidate if candidate > current else candidate + timedelta(days=7)

    if interval == "month":
        candidate = current.replace(day=1, hour=9, minute=0, second=0, microsecond=0)
        if candidate > current:
            return candidate
        year, month = divmod(candidate.month, 12)
        return candidate.replace(year=candidate.year + year, month=month + 1)

    raise ValueError(f"모르는 interval: {interval}")


def seconds_until_next_close(interval: str, now: datetime | None = None) -> float:
    """다음 봉 마감 + 정착 대기까지 남은 초. 봉 마감 직후에 판단하기 위한 것."""
    current = (now or datetime.now(KST)).astimezone(KST)
    target = next_close(interval, current) + timedelta(seconds=SETTLE_DELAY_SEC)
    return max((target - current).total_seconds(), 1.0)


def describe_next(interval: str, now: datetime | None = None) -> str:
    target = next_close(interval, now)
    remaining = seconds_until_next_close(interval, now)
    hours, rest = divmod(int(remaining), 3600)
    minutes, seconds = divmod(rest, 60)
    parts = (f"{hours}시간 " if hours else "") + (f"{minutes}분 " if minutes or hours else "")
    return f"다음 봉 마감 {target:%Y-%m-%d %H:%M} KST ({parts}{seconds}초 후)"
