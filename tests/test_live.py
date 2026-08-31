"""실전 모듈의 안전장치 검증 — 여기가 뚫리면 진짜 돈이 나간다.

네트워크도 API 키도 쓰지 않는다. 전부 로컬에서 검증 가능한 것만.
"""
import json

import pytest

from upbit.live import MIN_ORDER_KRW, Config, SafetyError, Trader, load_state, save_state
from upbit.strategies import MACrossStrategy


@pytest.fixture
def strategy():
    return MACrossStrategy(5, 20)


@pytest.fixture
def locked_config():
    """키도 없고 실주문도 잠긴 기본 상태."""
    return Config(access_key="", secret_key="", max_order_krw=10_000, allow_live=False)


@pytest.fixture
def unlocked_config():
    return Config(
        access_key="fake-access", secret_key="fake-secret",
        max_order_krw=10_000, allow_live=True,
    )


# ---------- 실주문 잠금 ----------

def test_live_blocked_when_env_flag_is_off(strategy, locked_config):
    with pytest.raises(SafetyError, match="UPBIT_ALLOW_LIVE"):
        Trader(strategy, order_krw=6_000, live=True, config=locked_config)


def test_live_blocked_when_keys_are_missing(strategy):
    config = Config(access_key="", secret_key="", max_order_krw=10_000, allow_live=True)
    with pytest.raises(SafetyError, match="API 키"):
        Trader(strategy, order_krw=6_000, live=True, config=config)


def test_dry_run_needs_neither_flag_nor_keys(strategy, locked_config):
    trader = Trader(strategy, order_krw=6_000, live=False, config=locked_config)
    assert trader.live is False
    assert trader._upbit is None


def test_default_is_dry_run(strategy, locked_config):
    assert Trader(strategy, order_krw=6_000, config=locked_config).live is False


# ---------- 주문 금액 상한 ----------

def test_rejects_order_below_upbit_minimum(strategy, locked_config):
    with pytest.raises(SafetyError, match="최소 주문금액"):
        Trader(strategy, order_krw=MIN_ORDER_KRW - 1, config=locked_config)


def test_rejects_order_above_configured_cap(strategy, locked_config):
    with pytest.raises(SafetyError, match="상한"):
        Trader(strategy, order_krw=10_001, config=locked_config)


def test_accepts_order_exactly_at_cap(strategy, locked_config):
    assert Trader(strategy, order_krw=10_000, config=locked_config).order_krw == 10_000


def test_cap_applies_to_live_mode_too(strategy, unlocked_config):
    """실주문 모드에서도 상한이 먼저 걸려야 한다 (0 하나 더 붙은 오타 방어)."""
    with pytest.raises(SafetyError, match="상한"):
        Trader(strategy, order_krw=100_000, live=True, config=unlocked_config)


# ---------- 모의 주문은 절대 실주문으로 새지 않는다 ----------

def test_dry_run_buy_and_sell_never_touch_the_exchange(strategy, locked_config):
    trader = Trader(strategy, order_krw=6_000, config=locked_config)
    assert trader.buy(100.0)["simulated"] is True
    assert trader.sell(100.0)["simulated"] is True
    assert trader._upbit is None  # 거래소 클라이언트 자체가 없다


# ---------- 설정 로딩 ----------

def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("UPBIT_ACCESS_KEY", "A")
    monkeypatch.setenv("UPBIT_SECRET_KEY", "B")
    monkeypatch.setenv("UPBIT_MAX_ORDER_KRW", "7500")
    monkeypatch.setenv("UPBIT_ALLOW_LIVE", "TRUE")
    config = Config.from_env()
    assert config.has_keys and config.allow_live
    assert config.max_order_krw == 7500


@pytest.mark.parametrize("value", ["false", "False", "0", "yes", "", "  "])
def test_allow_live_only_opens_on_literal_true(monkeypatch, value):
    monkeypatch.setenv("UPBIT_ALLOW_LIVE", value)
    assert Config.from_env().allow_live is False


def test_config_without_keys_is_not_usable_for_live(monkeypatch):
    monkeypatch.setenv("UPBIT_ACCESS_KEY", "")
    monkeypatch.setenv("UPBIT_SECRET_KEY", "")
    assert Config.from_env().has_keys is False


# ---------- 상태 저장 ----------

def test_state_roundtrip(tmp_path, monkeypatch):
    import upbit.live as live_mod
    monkeypatch.setattr(live_mod, "STATE_PATH", tmp_path / "state.json")

    assert load_state()["position"] == 0          # 파일이 없으면 현금 상태
    save_state({"position": 1, "last_signal_time": "2024-01-01T00:00:00", "history": []})
    assert load_state()["position"] == 1

    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["position"] == 1
