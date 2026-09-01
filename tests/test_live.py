"""실전 트레이더 검증.

가짜 거래소 덕분에 실제 돈 없이 주문 실패·부분체결·타임아웃 대응을 전부 검증한다.
여기가 뚫리면 진짜 돈이 샌다.
"""
import pandas as pd
import pytest

from upbit.exchange import (
    MIN_ORDER_KRW,
    AuthError,
    OrderRejected,
    RateLimited,
    UnknownOutcome,
)
from upbit.fake_exchange import FakeExchange
from upbit.live import Config, SafetyError, Trader, build_exchange, load_state, save_state
from upbit.risk import RiskRules
from upbit.strategies.base import Strategy

PRICE = 100_000_000.0


class FixedSignal(Strategy):
    """테스트용 — 항상 정해진 목표 포지션을 낸다."""

    name = "고정신호"

    def __init__(self, target: int):
        self.target = target

    def generate_positions(self, df):
        return pd.Series(self.target, index=df.index, dtype=int)


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "state.json"


@pytest.fixture
def config():
    return Config(access_key="", secret_key="", max_order_krw=10_000, allow_live=False)


def make_trader(exchange, target=1, state_path=None, config=None, risk=None, **kw):
    """신호를 고정하고 캔들 조회를 가짜로 대체한 트레이더."""
    trader = Trader(
        strategy=FixedSignal(target),
        exchange=exchange,
        ticker="KRW-BTC",
        order_krw=kw.pop("order_krw", 6_000),
        risk=risk,
        config=config or Config(max_order_krw=10_000, allow_live=False),
        state_path=state_path,
        sleep=lambda _: None,   # 테스트에서 실제로 자지 않는다
        **kw,
    )
    trader.current_signal = lambda count=200: (target, pd.Timestamp("2024-01-01").to_pydatetime())
    trader._atr_value = lambda: PRICE * 0.02
    return trader


# ---------- 안전장치 ----------

def test_live_blocked_when_env_flag_is_off(config):
    with pytest.raises(SafetyError, match="UPBIT_ALLOW_LIVE"):
        build_exchange(config, live=True, ticker="KRW-BTC", state={})


def test_live_blocked_when_keys_are_missing():
    config = Config(access_key="", secret_key="", allow_live=True)
    with pytest.raises(SafetyError, match="API 키"):
        build_exchange(config, live=True, ticker="KRW-BTC", state={})


def test_dry_run_uses_a_fake_exchange_and_needs_no_keys(config):
    exchange = build_exchange(config, live=False, ticker="KRW-BTC", state={})
    assert isinstance(exchange, FakeExchange)


def test_dry_run_carries_paper_balance_between_runs(config):
    state = {"paper": {"krw": 12_345.0, "coin": 0.5}}
    exchange = build_exchange(config, live=False, ticker="KRW-BTC", state=state)
    assert exchange.krw == 12_345.0 and exchange.coin == 0.5


def test_rejects_order_below_upbit_minimum():
    with pytest.raises(SafetyError, match="최소 주문금액"):
        make_trader(FakeExchange(), order_krw=MIN_ORDER_KRW - 1)


def test_rejects_order_above_configured_cap():
    with pytest.raises(SafetyError, match="상한"):
        make_trader(FakeExchange(), order_krw=10_001)


# ---------- 잔고가 진실이다 ----------

def test_position_comes_from_the_balance_not_the_state_file(state_path):
    """상태파일이 '보유 중'이라고 우겨도 잔고가 비었으면 현금이다."""
    save_state({"entry_price": PRICE, "stop_price": 1.0}, state_path)
    trader = make_trader(FakeExchange(krw=100_000, coin=0.0, price=PRICE), state_path=state_path)
    assert trader.current_position(PRICE) == 0


def test_dust_below_minimum_counts_as_no_position():
    """5,000원 미만은 팔 수도 없으니 보유로 치면 안 된다."""
    dust = FakeExchange(coin=0.00001, price=PRICE)   # 1,000원어치
    assert make_trader(dust).current_position(PRICE) == 0


def test_reconcile_adopts_balance_when_state_has_no_entry(state_path):
    """수동 매수했거나 상태파일이 지워진 경우 — 잔고를 진실로 받아들인다."""
    exchange = FakeExchange(krw=0, coin=0.001, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path)
    trader.step()
    assert load_state(state_path)["entry_price"] == pytest.approx(PRICE)


def test_reconcile_clears_entry_when_balance_is_empty(state_path):
    """사람이 앱에서 직접 팔았거나 주문이 실제로는 실패한 경우."""
    save_state({"entry_price": PRICE, "stop_price": PRICE * 0.9}, state_path)
    trader = make_trader(FakeExchange(krw=100_000, coin=0.0, price=PRICE),
                         target=0, state_path=state_path)
    trader.step()
    assert load_state(state_path)["entry_price"] is None


# ---------- 주문 실패는 상태를 바꾸면 안 된다 ----------

def test_rejected_buy_does_not_record_a_position(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    exchange.fail_next(OrderRejected("잔고 부족"))
    trader = make_trader(exchange, target=1, state_path=state_path)

    result = trader.step()

    assert result["action"] == "hold"
    assert result["order"] is None
    assert load_state(state_path)["entry_price"] is None
    assert exchange.coin == 0


def test_unknown_outcome_does_not_record_but_balance_reveals_the_truth(state_path):
    """타임아웃으로 결과를 몰라도, 다음 사이클에 잔고를 읽어 스스로 맞춘다."""
    exchange = FakeExchange(krw=100_000, price=PRICE)
    exchange.timeout_next_order()
    trader = make_trader(exchange, target=1, state_path=state_path)

    first = trader.step()
    assert first["action"] == "hold"                 # 성공으로 기록하지 않았고
    assert exchange.coin > 0                          # 실제로는 주문이 들어갔다

    trader.step()                                     # 다음 사이클에서 대조
    assert load_state(state_path)["entry_price"] is not None


def test_unknown_outcome_never_double_buys(state_path):
    """제일 위험한 시나리오 — 실패로 단정하고 재주문하면 두 번 산다."""
    exchange = FakeExchange(krw=100_000, price=PRICE)
    exchange.timeout_next_order()
    trader = make_trader(exchange, target=1, state_path=state_path)

    trader.step()
    trader.step()
    trader.step()

    buys = [c for c in exchange.calls if c[0] == "buy_market"]
    assert len(buys) == 1, "이미 들고 있는데 또 샀다"


def test_auth_error_propagates_and_stops_the_bot(state_path):
    """인증 오류는 삼키면 안 된다 — 계속 시도하면 계정이 차단될 수 있다."""
    exchange = FakeExchange(krw=100_000, price=PRICE)
    exchange.fail_next(AuthError("expired_access_key"))
    trader = make_trader(exchange, target=1, state_path=state_path)

    with pytest.raises(AuthError):
        trader.step()


def test_rate_limit_is_swallowed_for_this_cycle(state_path):
    """레이트리밋은 이번 주기만 건너뛰고 다음에 다시 시도한다."""
    exchange = FakeExchange(krw=100_000, price=PRICE)
    exchange.fail_next(RateLimited("too_many_requests"))
    trader = make_trader(exchange, target=1, state_path=state_path)

    assert trader.step()["action"] == "hold"
    assert trader.step()["action"] == "buy"


# ---------- 체결 확인 ----------

def test_pending_order_is_polled_until_filled(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE, leave_pending=True)
    trader = make_trader(exchange, target=1, state_path=state_path)
    trader.FILL_TIMEOUT = 3

    # 첫 폴링 이후 체결되도록 만든다
    original = exchange.get_order
    def settle(order_uuid):
        exchange.fill_pending(order_uuid)
        return original(order_uuid)
    exchange.get_order = settle

    assert trader.step()["action"] == "buy"


def test_never_filled_order_is_not_recorded_as_a_position(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE, leave_pending=True)
    trader = make_trader(exchange, target=1, state_path=state_path)
    trader.FILL_TIMEOUT = 0

    result = trader.step()
    assert result["action"] == "hold"
    assert load_state(state_path)["entry_price"] is None


def test_partial_fill_records_what_actually_filled(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE, partial_fill_ratio=0.5, fee=0.0)
    trader = make_trader(exchange, target=1, state_path=state_path)

    result = trader.step()
    assert result["action"] == "buy"
    assert result["order"].executed_krw == pytest.approx(3_000)  # 6,000의 절반


# ---------- 손절이 실전에 연결됐나 ----------

def test_stop_price_is_set_on_entry(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path,
                         risk=RiskRules(stop_loss_pct=0.05))
    trader.step()
    assert load_state(state_path)["stop_price"] == pytest.approx(PRICE * 0.95, rel=1e-3)


def test_stop_loss_sells_even_when_the_strategy_says_hold(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path,
                         risk=RiskRules(stop_loss_pct=0.05))
    trader.step()
    assert exchange.coin > 0

    exchange.move_price(0.90)                # 10% 하락 → 손절선 이탈
    result = trader.step()

    assert result["action"] == "sell" and result["reason"] == "손절"
    assert exchange.coin == pytest.approx(0)


def test_no_reentry_until_the_strategy_goes_flat(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path,
                         risk=RiskRules(stop_loss_pct=0.05))
    trader.step()
    exchange.move_price(0.90)
    trader.step()                             # 손절

    trader.step()                             # 전략은 여전히 매수 신호
    assert exchange.coin == pytest.approx(0), "손절 직후 재진입했다"
    assert load_state(state_path)["blocked"] is True


def test_trailing_stop_rises_with_price(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path,
                         risk=RiskRules(stop_loss_pct=0.05, trailing=True))
    trader.step()
    first_stop = load_state(state_path)["stop_price"]

    exchange.move_price(1.20)
    trader.step()

    assert load_state(state_path)["stop_price"] > first_stop


def test_without_risk_rules_no_stop_is_set(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path)
    trader.step()
    assert load_state(state_path)["stop_price"] is None


# ---------- 매도 ----------

def test_sell_on_signal(state_path):
    exchange = FakeExchange(krw=0, coin=0.001, price=PRICE)
    trader = make_trader(exchange, target=0, state_path=state_path)

    result = trader.step()
    assert result["action"] == "sell"
    assert exchange.coin == pytest.approx(0)


def test_dust_is_left_alone_instead_of_failing_forever(state_path):
    """평가액이 최소 주문금액 미만이면 팔 수 없다 — 계속 실패하지 말고 남긴다."""
    exchange = FakeExchange(coin=0.00001, price=PRICE)   # 1,000원어치
    trader = make_trader(exchange, target=0, state_path=state_path)

    assert trader.step()["action"] == "hold"
    assert not [c for c in exchange.calls if c[0] == "sell_market"]


# ---------- 기록 ----------

def test_history_records_actual_fill_not_the_requested_amount(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE, fee=0.0005)
    trader = make_trader(exchange, target=1, state_path=state_path)
    trader.step()

    entry = load_state(state_path)["history"][-1]
    assert entry["action"] == "buy"
    assert entry["fee"] > 0
    assert entry["avg_price"] == pytest.approx(PRICE)


def test_paper_balance_persists_across_runs(state_path):
    exchange = FakeExchange(krw=100_000, price=PRICE)
    make_trader(exchange, target=1, state_path=state_path).step()

    paper = load_state(state_path)["paper"]
    assert paper["krw"] < 100_000 and paper["coin"] > 0


# ---------- 설정 ----------

def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("UPBIT_ACCESS_KEY", "A")
    monkeypatch.setenv("UPBIT_SECRET_KEY", "B")
    monkeypatch.setenv("UPBIT_MAX_ORDER_KRW", "7500")
    monkeypatch.setenv("UPBIT_ALLOW_LIVE", "TRUE")
    config = Config.from_env()
    assert config.has_keys and config.allow_live and config.max_order_krw == 7500


@pytest.mark.parametrize("value", ["false", "False", "0", "yes", "", "  "])
def test_allow_live_only_opens_on_literal_true(monkeypatch, value):
    monkeypatch.setenv("UPBIT_ALLOW_LIVE", value)
    assert Config.from_env().allow_live is False


# ---------- 먼지 방지 ----------

def test_minimum_order_floor_is_above_the_exchange_minimum():
    """업비트 최소(5,000원)로 사면 수수료 때문에 즉시 못 파는 먼지가 된다."""
    from upbit.live import MIN_SAFE_ORDER_KRW
    assert MIN_SAFE_ORDER_KRW > MIN_ORDER_KRW

    with pytest.raises(SafetyError, match="먼지"):
        make_trader(FakeExchange(), order_krw=MIN_ORDER_KRW)


def test_buying_exactly_the_exchange_minimum_creates_unsellable_dust():
    """왜 하한을 올렸는지 증명 — 안전장치를 우회해 직접 확인한다."""
    exchange = FakeExchange(krw=100_000, price=PRICE, fee=0.0005)
    order = exchange.buy_market("KRW-BTC", MIN_ORDER_KRW)

    value = exchange.coin * PRICE
    assert value < MIN_ORDER_KRW, "수수료를 떼면 최소 주문금액에 못 미친다"
    with pytest.raises(OrderRejected, match="최소 주문금액"):
        exchange.sell_market("KRW-BTC", exchange.coin)


def test_safe_minimum_produces_a_sellable_position():
    from upbit.live import MIN_SAFE_ORDER_KRW

    exchange = FakeExchange(krw=100_000, price=PRICE, fee=0.0005)
    exchange.buy_market("KRW-BTC", MIN_SAFE_ORDER_KRW)
    assert exchange.sell_market("KRW-BTC", exchange.coin).is_filled


def test_dust_is_reported_and_counted_as_cash():
    exchange = FakeExchange(coin=0.00001, price=PRICE)   # 1,000원어치
    trader = make_trader(exchange)
    assert trader.dust_krw(PRICE) == pytest.approx(1_000)
    assert trader.current_position(PRICE) == 0


# ---------- 긴급 정지 ----------

def test_panic_sell_clears_position_and_persists(state_path):
    exchange = FakeExchange(krw=0, coin=0.001, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path)

    order = trader.panic_sell()

    assert order is not None and exchange.coin == pytest.approx(0)
    state = load_state(state_path)
    assert state["entry_price"] is None
    assert state["paper"]["coin"] == pytest.approx(0), "가상 잔고가 저장되지 않았다"


def test_panic_sell_blocks_reentry(state_path):
    """급한 상황에 껐다 켰다고 바로 다시 사면 곤란하다."""
    exchange = FakeExchange(krw=0, coin=0.001, price=PRICE)
    trader = make_trader(exchange, target=1, state_path=state_path)
    trader.panic_sell()

    assert load_state(state_path)["blocked"] is True
    trader.step()
    assert exchange.coin == pytest.approx(0), "차단 중인데 다시 샀다"


def test_panic_sell_with_nothing_to_sell_is_harmless(state_path):
    exchange = FakeExchange(krw=100_000, coin=0.0, price=PRICE)
    trader = make_trader(exchange, target=0, state_path=state_path)
    assert trader.panic_sell() is None
    assert load_state(state_path)["entry_price"] is None


# ---------- API 키 만료 ----------

def test_warns_when_the_api_key_expires_soon():
    """키가 만료되면 봇이 조용히 멈추고 포지션이 방치된다."""
    from datetime import datetime, timedelta, timezone

    soon = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    exchange = FakeExchange(price=PRICE, key_info=[{"expire_at": soon}])
    assert "3일 뒤 만료" in make_trader(exchange).check_api_key()


def test_no_warning_when_the_key_has_plenty_of_time():
    from datetime import datetime, timedelta, timezone

    later = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
    exchange = FakeExchange(price=PRICE, key_info=[{"expire_at": later}])
    assert make_trader(exchange).check_api_key() is None


def test_reports_an_already_expired_key():
    from datetime import datetime, timedelta, timezone

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    exchange = FakeExchange(price=PRICE, key_info=[{"expire_at": past}])
    assert "이미 만료" in make_trader(exchange).check_api_key()


def test_key_check_is_silent_when_unsupported_or_empty():
    assert make_trader(FakeExchange(price=PRICE)).check_api_key() is None


def test_malformed_expiry_does_not_crash():
    exchange = FakeExchange(price=PRICE, key_info=[{"expire_at": "말도안되는값"}])
    assert make_trader(exchange).check_api_key() is None



# ---------- 운영 파일 격리 ----------

def test_fills_from_tests_never_touch_the_real_journal(state_path):
    """pytest 가 실제 reports/fills.csv 를 오염시키던 버그의 회귀 테스트."""
    from pathlib import Path
    import upbit.journal

    real_journal = Path(__file__).resolve().parent.parent / "reports" / "fills.csv"
    before = real_journal.read_bytes() if real_journal.exists() else None

    exchange = FakeExchange(krw=100_000, price=PRICE)
    make_trader(exchange, target=1, state_path=state_path).step()

    after = real_journal.read_bytes() if real_journal.exists() else None
    assert before == after, "테스트 체결이 실제 장부에 기록됐다"
    assert upbit.journal.FILLS_PATH.exists(), "격리된 장부에는 기록돼야 한다"


def test_journal_path_can_be_injected(tmp_path, state_path):
    journal = tmp_path / "custom.csv"
    exchange = FakeExchange(krw=100_000, price=PRICE)
    trader = Trader(
        strategy=FixedSignal(1), exchange=exchange, order_krw=6_000,
        config=Config(max_order_krw=10_000), state_path=state_path,
        journal_path=journal, sleep=lambda _: None,
    )
    trader.current_signal = lambda count=200: (1, pd.Timestamp("2024-01-01").to_pydatetime())
    trader.step()
    assert journal.exists() and "fake-buy-1" in journal.read_text(encoding="utf-8")
