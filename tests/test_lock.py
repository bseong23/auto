"""중복 실행 방지 — 뚫리면 주문이 두 배 나간다."""
import json
import os

import pytest

from upbit.lock import AlreadyRunning, ProcessLock


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "bot.lock"


def test_lock_is_created_and_removed(lock_path):
    with ProcessLock(lock_path):
        assert lock_path.exists()
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    assert not lock_path.exists()


def test_second_lock_is_refused_while_first_is_held(lock_path):
    with ProcessLock(lock_path):
        with pytest.raises(AlreadyRunning, match="이미 봇이 돌고 있다"):
            ProcessLock(lock_path).acquire()


def test_lock_can_be_taken_after_release(lock_path):
    ProcessLock(lock_path).acquire().release()
    with ProcessLock(lock_path):
        assert lock_path.exists()


def test_stale_lock_from_a_dead_process_is_taken_over(lock_path):
    """죽은 프로세스의 락 때문에 봇을 영영 못 켜면 그것도 문제다."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": 999_999, "started": "2024-01-01T00:00:00"}))

    with ProcessLock(lock_path):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()


def test_corrupted_lock_file_is_ignored(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("이건 JSON이 아님")
    with ProcessLock(lock_path):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()


def test_release_does_not_remove_someone_elses_lock(lock_path):
    """내가 안 잡은 락을 지우면 남의 봇이 무방비가 된다."""
    mine = ProcessLock(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": os.getpid() + 1, "started": "x"}))

    mine.release()
    assert lock_path.exists()


def test_lock_is_released_even_if_the_body_raises(lock_path):
    with pytest.raises(ValueError):
        with ProcessLock(lock_path):
            raise ValueError("봇이 터졌다")
    assert not lock_path.exists()


def test_label_is_shown_in_the_error(lock_path):
    with ProcessLock(lock_path, label="KRW-BTC/day"):
        with pytest.raises(AlreadyRunning, match="KRW-BTC/day"):
            ProcessLock(lock_path).acquire()
