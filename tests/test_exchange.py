"""거래소 어댑터 검증 — 오류 분류와 가짜 거래소가 실제처럼 동작하는지."""
import pytest

from upbit.exchange import (
    MIN_ORDER_KRW,
    AuthError,
    Exchange,
    ExchangeError,
    OrderRejected,
    RateLimited,
    TransientError,
    UnknownOutcome,
    classify,
    coin_of,
    parse_order,
)
from upbit.fake_exchange import FakeExchange


# ---------- 오류 분류 ----------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("expired_access_key", AuthError),
        ("no_authorization_i_p", AuthError),
        ("out_of_scope", AuthError),
        ("insufficient_funds_bid", OrderRejected),
        ("under_min_total_ask", OrderRejected),
        ("too_many_requests", RateLimited),
        ("nonce_used", TransientError),
        ("처음보는오류", ExchangeError),
    ],
)
def test_error_names_map_to_our_categories(name, expected):
    assert isinstance(classify(name), expected)


def test_auth_errors_are_not_retryable_category():
    """인증 오류를 재시도 가능한 것으로 잘못 분류하면 계정이 차단될 수 있다."""
    error = classify("expired_access_key")
    assert isinstance(error, AuthError)
    assert not isinstance(error, (RateLimited, TransientError))


def test_unknown_outcome_is_not_a_rejection():
    """'알 수 없음'을 '거부'로 취급하면 이미 들어간 주문을 또 낸다."""
    assert not issubclass(UnknownOutcome, OrderRejected)
    assert issubclass(UnknownOutcome, ExchangeError)


def test_coin_of():
    assert coin_of("KRW-BTC") == "BTC"
    assert coin_of("KRW-DOGE") == "DOGE"


# ---------- 주문 응답 파싱 ----------

def test_parse_filled_order():
    order = parse_order({
        "uuid": "abc", "side": "bid", "state": "done",
        "executed_volume": "0.001", "executed_funds": "100000", "paid_fee": "50",
    })
    assert order.is_filled and not order.is_pending and not order.is_empty
    assert order.avg_price == pytest.approx(100_000_000)


def test_parse_order_falls_back_to_trades_when_funds_missing():
    """시장가 매수 응답에 executed_funds 가 없을 수 있다 — trades 로 합산."""
    order = parse_order({
        "uuid": "abc", "side": "bid", "state": "done", "executed_volume": "0.002",
        "trades": [{"funds": "60000"}, {"funds": "40000"}],
    })
    assert order.executed_krw == pytest.approx(100_000)


def test_parse_pending_order_is_not_filled():
    order = parse_order({"uuid": "x", "side": "bid", "state": "wait"})
    assert order.is_pending and order.is_empty and not order.is_filled
    assert order.avg_price is None


def test_partially_filled_then_cancelled_counts_as_filled():
    """부분체결 후 취소 = '더 이상 안 채워짐'. 체결된 만큼은 실제로 샀다."""
    order = parse_order({
        "uuid": "x", "side": "bid", "state": "cancel",
        "executed_volume": "0.0005", "executed_funds": "50000",
    })
    assert order.is_filled and not order.is_empty


# ---------- 가짜 거래소 ----------

def test_fake_satisfies_the_exchange_protocol():
    assert isinstance(FakeExchange(), Exchange)


def test_buy_moves_balances_and_charges_fee():
    ex = FakeExchange(krw=100_000, price=100_000_000, fee=0.0005)
    order = ex.buy_market("KRW-BTC", 50_000)

    assert ex.krw == pytest.approx(50_000)
    assert order.paid_fee == pytest.approx(25)
    assert ex.coin == pytest.approx((50_000 - 25) / 100_000_000)


def test_sell_returns_krw_minus_fee():
    ex = FakeExchange(krw=0, coin=0.001, price=100_000_000, fee=0.0005)
    ex.sell_market("KRW-BTC", 0.001)

    gross = 0.001 * 100_000_000
    assert ex.coin == pytest.approx(0)
    assert ex.krw == pytest.approx(gross * (1 - 0.0005))


def test_rejects_below_minimum_order():
    ex = FakeExchange(krw=100_000)
    with pytest.raises(OrderRejected, match="최소 주문금액"):
        ex.buy_market("KRW-BTC", MIN_ORDER_KRW - 1)


def test_rejects_when_krw_is_insufficient():
    ex = FakeExchange(krw=10_000)
    with pytest.raises(OrderRejected, match="잔고 부족"):
        ex.buy_market("KRW-BTC", 50_000)


def test_rejects_selling_more_than_held():
    ex = FakeExchange(coin=0.001, price=100_000_000)
    with pytest.raises(OrderRejected, match="보유 수량 부족"):
        ex.sell_market("KRW-BTC", 0.002)


def test_rejects_selling_dust_below_minimum():
    """평가액이 5,000원 미만이면 팔 수 없다 — 먼지가 남는 원인."""
    ex = FakeExchange(coin=0.00001, price=100_000_000)  # 1,000원어치
    with pytest.raises(OrderRejected, match="최소 주문금액"):
        ex.sell_market("KRW-BTC", 0.00001)


# ---------- 실패 주입 ----------

def test_injected_error_fires_once_then_recovers():
    ex = FakeExchange(krw=100_000)
    ex.fail_next(RateLimited("잠깐 멈춰"))

    with pytest.raises(RateLimited):
        ex.buy_market("KRW-BTC", 10_000)

    assert ex.buy_market("KRW-BTC", 10_000).is_filled  # 다음 번엔 정상


def test_injected_errors_are_consumed_in_order():
    ex = FakeExchange(krw=100_000)
    ex.fail_next(RateLimited("1"))
    ex.fail_next(AuthError("2"))

    with pytest.raises(RateLimited):
        ex.buy_market("KRW-BTC", 10_000)
    with pytest.raises(AuthError):
        ex.buy_market("KRW-BTC", 10_000)


def test_timeout_raises_but_the_order_actually_went_through():
    """제일 고약한 경우 — 실패로 단정하고 재주문하면 두 번 산다."""
    ex = FakeExchange(krw=100_000, price=100_000_000)
    ex.timeout_next_order()

    with pytest.raises(UnknownOutcome):
        ex.buy_market("KRW-BTC", 50_000)

    assert ex.coin > 0, "응답만 못 받았을 뿐 주문은 체결됐다"
    assert ex.krw == pytest.approx(50_000)


def test_rejection_really_does_not_change_balances():
    """거부는 확실히 안 들어간 것 — 잔고가 그대로여야 한다."""
    ex = FakeExchange(krw=100_000, price=100_000_000)
    ex.fail_next(OrderRejected("거부"))

    with pytest.raises(OrderRejected):
        ex.buy_market("KRW-BTC", 50_000)

    assert ex.krw == pytest.approx(100_000) and ex.coin == 0


# ---------- 부분체결 / 미체결 ----------

def test_partial_fill_buys_less_than_requested():
    ex = FakeExchange(krw=100_000, price=100_000_000, partial_fill_ratio=0.5)
    order = ex.buy_market("KRW-BTC", 50_000)

    assert order.executed_krw == pytest.approx(25_000)
    assert ex.krw == pytest.approx(75_000)
    assert order.is_filled and order.state == "cancel"


def test_pending_order_is_not_filled_until_settled():
    ex = FakeExchange(krw=100_000, leave_pending=True)
    order = ex.buy_market("KRW-BTC", 50_000)

    assert order.is_pending and not order.is_filled
    assert ex.get_order(order.uuid).is_pending
    ex.fill_pending(order.uuid)
    assert not ex.get_order(order.uuid).is_pending


def test_slippage_makes_fills_worse_than_quoted_price():
    ex = FakeExchange(krw=100_000, price=100_000_000, fee=0.0, slippage=0.01)
    order = ex.buy_market("KRW-BTC", 50_000)
    assert order.avg_price == pytest.approx(101_000_000)  # 1% 불리하게


def test_call_log_records_what_the_bot_did():
    ex = FakeExchange(krw=100_000)
    ex.get_krw_balance()
    ex.buy_market("KRW-BTC", 10_000)
    assert [name for name, _ in ex.calls] == ["get_krw_balance", "buy_market"]


def test_unknown_order_lookup_raises():
    with pytest.raises(ExchangeError, match="없는 주문"):
        FakeExchange().get_order("존재하지-않음")
