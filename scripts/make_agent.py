#!/usr/bin/env python
"""봇을 봉 마감마다 자동 실행하도록 launchd 에이전트를 만든다 (macOS).

조사 결과 실행이 하루 늦으면 수익률이 12%p 떨어졌다(docs/조사-실행타이밍.md).
cron 의 가치는 '빠른 것'이 아니라 **'빠뜨리지 않는 것'** 이다.

이 스크립트는 plist 를 **만들기만** 한다. 설치는 사람이 직접 한다 —
남의 컴퓨터에 백그라운드 작업을 몰래 심으면 안 되니까.

    python scripts/make_agent.py                     # 미리보기
    python scripts/make_agent.py --write             # 파일 생성
    python scripts/make_agent.py --write --live      # 실주문 (⚠️)

macOS 는 cron 대신 launchd 를 쓴다. cron 과 달리
**컴퓨터가 자던 시간의 실행을 깨어난 뒤 따라잡아 준다**(일봉 봇에 중요하다).
"""
import argparse
import os
import plistlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from upbit.data import VALID_INTERVALS
from upbit.schedule import MINUTE_INTERVALS

ROOT = Path(__file__).resolve().parent.parent
LABEL = "ai.upbit.bot"
AGENT_DIR = Path.home() / "Library" / "LaunchAgents"


def schedule_for(interval: str) -> list[dict]:
    """launchd 의 StartCalendarInterval 로 봉 마감 직후를 지정한다.

    업비트 일/주/월봉은 09:00 KST 마감. 마감 직후엔 API 에 확정봉이 아직
    안 올라와 있을 수 있어 1분 뒤로 잡는다.
    """
    if interval == "day":
        return [{"Hour": 9, "Minute": 1}]
    if interval == "week":
        return [{"Weekday": 1, "Hour": 9, "Minute": 1}]  # 월요일
    if interval == "month":
        return [{"Day": 1, "Hour": 9, "Minute": 1}]
    if interval in MINUTE_INTERVALS:
        step = MINUTE_INTERVALS[interval]
        if step < 60:
            return [{"Minute": m + 1} for m in range(0, 60, step)]
        hours = step // 60
        # 업비트 격자는 09:00 KST 앵커 — 240분봉은 01·05·09·13·17·21시
        return [{"Hour": (9 + h) % 24, "Minute": 1} for h in sorted(range(0, 24, hours),
                key=lambda h: (9 + h) % 24)]
    raise ValueError(f"모르는 interval: {interval}")


def build(args) -> dict:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"가상환경이 없다: {python}\n먼저 `make setup` 을 실행할 것.")

    command = [str(python), str(ROOT / "scripts" / "05_live.py"),
               "--ticker", args.ticker, "--interval", args.interval,
               "--strategy", args.strategy, "--order-krw", str(args.order_krw)]
    if args.stop_atr:
        command += ["--stop-atr", str(args.stop_atr)]
    if args.trailing:
        command.append("--trailing")
    if args.live:
        command.append("--live")

    return {
        "Label": LABEL,
        "ProgramArguments": command,
        "WorkingDirectory": str(ROOT),
        "StartCalendarInterval": schedule_for(args.interval),
        # 켤 때 바로 돌리지 않는다 — 봉 마감과 무관한 시각에 판단하면 안 된다
        "RunAtLoad": False,
        "StandardOutPath": str(ROOT / "reports" / "agent.log"),
        "StandardErrorPath": str(ROOT / "reports" / "agent.err.log"),
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        "ProcessType": "Background",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="launchd 에이전트 생성 (macOS)")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--strategy", default="ma")
    ap.add_argument("--order-krw", type=float, default=5500)
    ap.add_argument("--stop-atr", type=float, default=None)
    ap.add_argument("--trailing", action="store_true")
    ap.add_argument("--live", action="store_true", help="⚠️ 실주문")
    ap.add_argument("--write", action="store_true", help="실제로 파일 생성")
    args = ap.parse_args()

    config = build(args)
    target = AGENT_DIR / f"{LABEL}.plist"

    print(f"  라벨     : {LABEL}")
    print(f"  실행     : {' '.join(config['ProgramArguments'][1:])}")
    print(f"  모드     : {'🔴 실주문' if args.live else '🟢 모의'}")
    print(f"  일정     : {config['StartCalendarInterval']}")
    print(f"  로그     : reports/agent.log")
    print(f"  설치 위치: {target}")

    if args.live:
        print("\n  ⚠️  실주문 에이전트다. .env 의 UPBIT_ALLOW_LIVE=true 도 있어야 실제로 나간다.")
        print("      그리고 --live 는 확인 입력을 요구하는데, launchd 에는 입력이 없어")
        print("      **자동으로 취소된다**. 자동 실주문을 정말 원하면 그 확인 절차를")
        print("      의도적으로 손봐야 한다 — 지금은 안전하게 막혀 있다.")

    if not args.write:
        print("\n  (미리보기다. 실제로 만들려면 --write)")
        return

    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        plistlib.dump(config, f)
    print(f"\n  ✅ 생성됨: {target}")
    print("\n  등록:")
    print(f"    launchctl bootstrap gui/$(id -u) {target}")
    print("  해제:")
    print(f"    launchctl bootout gui/$(id -u)/{LABEL}")
    print("  상태 확인:")
    print(f"    launchctl print gui/$(id -u)/{LABEL} | head -20")
    print("  즉시 한 번 실행(테스트):")
    print(f"    launchctl kickstart -p gui/$(id -u)/{LABEL}")


if __name__ == "__main__":
    main()
