"""알림 — 실패해도 봇이 멈추면 안 된다."""
from upbit.exchange import OrderResult
from upbit.notify import Notifier


def test_disabled_without_a_webhook_url():
    assert Notifier().enabled is False
    assert Notifier().send("무시됨") is False


def test_enabled_with_a_url():
    assert Notifier("https://example.com/hook").enabled is True


def test_whitespace_url_counts_as_missing():
    assert Notifier("   ").enabled is False


def test_send_failure_never_raises():
    """알림이 안 갔다고 포지션이 방치되면 본말전도다."""
    notifier = Notifier("http://127.0.0.1:1/dead")
    assert notifier.send("hello") is False


def test_malformed_url_never_raises():
    assert Notifier("not-a-url").send("hello") is False


def test_env_reads_webhook(monkeypatch):
    monkeypatch.setenv("UPBIT_WEBHOOK_URL", "https://example.com/x")
    assert Notifier.from_env().enabled is True


def test_message_helpers_do_not_raise_when_disabled():
    notifier = Notifier()
    order = OrderResult(uuid="u", side="bid", state="done",
                        executed_volume=0.001, executed_krw=100_000, paid_fee=50)
    assert notifier.order_filled("KRW-BTC", "buy", "신호", order, True) is False
    assert notifier.stop_hit("KRW-BTC", 1.0, 2.0) is False
    assert notifier.bot_stopped("테스트") is False
    assert notifier.heartbeat("KRW-BTC", 1.0, 1, 100.0) is False
