"""테스트 공통 가드.

pytest 가 돌 때마다 테스트의 모의 체결이 **실제 체결 장부(reports/fills.csv)** 에 기록되던
사고가 있었다. 실측 슬리피지 통계가 테스트 데이터로 오염된다.

여기서 운영 파일 경로를 전부 임시 폴더로 돌린다. 개별 테스트가 경로를 넘기는 걸 깜빡해도
운영 파일은 안전하다.
"""
import pytest

import upbit.journal
import upbit.live
import upbit.lock


@pytest.fixture(autouse=True)
def _isolate_runtime_files(tmp_path, monkeypatch):
    monkeypatch.setattr(upbit.journal, "FILLS_PATH", tmp_path / "fills.csv")
    monkeypatch.setattr(upbit.live, "STATE_PATH", tmp_path / "bot_state.json")
    monkeypatch.setattr(upbit.live, "LOG_PATH", tmp_path / "trades.log")
    monkeypatch.setattr(upbit.lock, "LOCK_PATH", tmp_path / "bot.lock")
