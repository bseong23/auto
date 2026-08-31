"""중복 실행 방지 — 봇이 두 개 돌면 주문이 두 배 나간다.

`UPBIT_MAX_ORDER_KRW` 상한은 **1회 주문** 기준이라 이걸 못 막는다.
터미널 두 개에서 실행하거나, cron 주기가 실행 시간보다 짧아서 앞 실행이
안 끝났는데 다음이 시작되면 조용히 두 배가 들어간다.

PID 파일로 막는다. 프로세스가 죽어서 남은 낡은 락은 자동으로 인계받는다
(죽은 프로세스의 락 때문에 봇을 못 켜면 그것도 문제이므로).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "data" / "bot.lock"


class AlreadyRunning(RuntimeError):
    """다른 봇이 이미 돌고 있다."""


def _is_alive(pid: int) -> bool:
    """그 PID의 프로세스가 살아 있나. 시그널 0은 실제로 보내지 않고 존재만 확인한다."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 살아있는데 내 권한으로 못 건드리는 것
    return True


class ProcessLock:
    """`with ProcessLock():` 로 감싸면 그 블록 안에서만 봇이 하나임이 보장된다."""

    def __init__(self, path: Path | None = None, label: str = ""):
        self.path = path or LOCK_PATH
        self.label = label
        self._held = False

    def acquire(self) -> "ProcessLock":
        existing = self._read()
        if existing is not None:
            pid = existing.get("pid", -1)
            if _is_alive(pid):
                raise AlreadyRunning(
                    f"이미 봇이 돌고 있다 (PID {pid}, 시작 {existing.get('started', '?')}"
                    f"{', ' + existing['label'] if existing.get('label') else ''}).\n"
                    f"정말 아니라면 락 파일을 지울 것: {self.path}"
                )
            # 죽은 프로세스가 남긴 낡은 락 — 인계받는다
            self.path.unlink(missing_ok=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "label": self.label,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._held = True
        return self

    def release(self) -> None:
        """내가 잡은 락만 푼다. 남의 락은 건드리지 않는다."""
        if not self._held:
            return
        existing = self._read()
        if existing is not None and existing.get("pid") == os.getpid():
            self.path.unlink(missing_ok=True)
        self._held = False

    def _read(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # 깨진 락 파일은 없는 것으로 친다

    def __enter__(self) -> "ProcessLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
