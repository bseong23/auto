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
    ex = FakeExchange(krw=100_000, price=100_000_000, partial_fill_ratio=0.5, fee=0.0)
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



# ---------- 호가 단위 ----------

@pytest.mark.parametrize("price,tick", [
    (108_000_000, 1_000), (1_500_000, 500), (600_000, 100), (150_000, 50),
    (50_000, 10), (3_000, 1), (115, 0.1), (5.5, 0.001), (0.5, 0.0001),
])
def test_krw_tick_table(price, tick):
    from upbit.exchange import krw_tick_size
    assert krw_tick_size(price) == tick


def test_tick_ratio_explains_doge_spread():
    """DOGE 115원의 호가단위 0.1원 → 0.087%. 실측 스프레드 0.87%(≈10틱)의 구조적 하한."""
    from upbit.exchange import tick_ratio
    assert tick_ratio(115) == pytest.approx(0.1 / 115)
    assert tick_ratio(108_000_000) < 0.0001


# ---------- 틱 정렬 / 수량 정밀도 ----------

def test_align_to_tick_rounds_toward_the_maker_side():
    from upbit.exchange import align_to_tick
    assert align_to_tick(108_676_543, "bid") == 108_676_000   # 매수는 내림
    assert align_to_tick(108_676_543, "ask") == 108_677_000   # 매도는 올림
    assert align_to_tick(115.17, "bid") == pytest.approx(115.1)
    assert align_to_tick(115.17, "ask") == pytest.approx(115.2)


def test_align_to_tick_is_idempotent_on_grid_prices():
    from upbit.exchange import align_to_tick
    assert align_to_tick(108_676_000, "bid") == 108_676_000
    assert align_to_tick(108_676_000, "ask") == 108_676_000


def test_floor_volume_never_rounds_up():
    from upbit.exchange import floor_volume
    assert floor_volume(0.000050589999) == 0.00005058
    assert floor_volume(1.0) == 1.0


# ---------- 가짜 거래소: 호가 / 지정가 / 취소 ----------

def test_best_quotes_straddle_the_price():
    ex = FakeExchange(price=100_000_000, spread=0.0004)
    bid, ask = ex.get_best_quotes("KRW-BTC")
    assert bid < 100_000_000 < ask
    assert (ask - bid) / 100_000_000 == pytest.approx(0.0004)


def test_limit_buy_fills_at_the_limit_price_without_slippage():
    ex = FakeExchange(krw=100_000, price=100_000_000, fee=0.0005, slippage=0.01)
    order = ex.buy_limit("KRW-BTC", 99_000_000, 0.0005)
    assert order.is_filled
    assert order.avg_price == pytest.approx(99_000_000)      # 슬리피지 1%가 적용되지 않았다
    assert ex.coin == pytest.approx(0.0005)
    assert ex.krw == pytest.approx(100_000 - 49_500 - 49_500 * 0.0005)


def test_limit_sell_fills_at_the_limit_price():
    ex = FakeExchange(krw=0, coin=0.001, price=100_000_000, fee=0.0005)
    order = ex.sell_limit("KRW-BTC", 101_000_000, 0.001)
    assert order.is_filled and order.avg_price == pytest.approx(101_000_000)
    assert ex.krw == pytest.approx(101_000 * (1 - 0.0005))


def test_unfilled_limit_order_stays_pending_and_reserves_funds():
    ex = FakeExchange(krw=100_000, price=100_000_000, limit_fills=False)
    order = ex.buy_limit("KRW-BTC", 99_000_000, 0.0005)
    assert order.is_pending
    assert ex.krw == pytest.approx(100_000 - 49_500)   # 주문금액이 묶인다
    assert ex.coin == 0


def test_cancel_releases_reserved_funds():
    ex = FakeExchange(krw=100_000, price=100_000_000, limit_fills=False)
    order = ex.buy_limit("KRW-BTC", 99_000_000, 0.0005)
    cancelled = ex.cancel_order(order.uuid)
    assert cancelled.state == "cancel" and cancelled.is_empty
    assert ex.krw == pytest.approx(100_000)


def test_cancel_of_a_filled_order_is_rejected_like_upbit():
    """취소 요청과 체결이 경합하는 경우 — 호출자는 재조회해서 실제 체결량을 믿어야 한다."""
    ex = FakeExchange(krw=100_000, price=100_000_000)
    order = ex.buy_limit("KRW-BTC", 99_000_000, 0.0005)
    assert order.is_filled
    with pytest.raises(OrderRejected, match="이미 체결"):
        ex.cancel_order(order.uuid)


def test_partial_limit_fill_returns_the_rest_on_cancel():
    ex = FakeExchange(krw=100_000, price=100_000_000, partial_fill_ratio=0.4, fee=0.0)
    order = ex.buy_limit("KRW-BTC", 100_000_000, 0.0005)
    assert order.executed_volume == pytest.approx(0.0002)
    assert ex.coin == pytest.approx(0.0002)
    assert ex.krw == pytest.approx(100_000 - 20_000)      # 미체결 60%는 돌려받았다


def test_limit_order_rejects_below_minimum_and_over_balance():
    ex = FakeExchange(krw=10_000, price=100_000_000)
    with pytest.raises(OrderRejected, match="최소 주문금액"):
        ex.buy_limit("KRW-BTC", 100_000_000, 0.00001)
    with pytest.raises(OrderRejected, match="잔고 부족"):
        ex.buy_limit("KRW-BTC", 100_000_000, 0.001)


def test_limit_timeout_injection_still_places_the_order():
    ex = FakeExchange(krw=100_000, price=100_000_000)
    ex.timeout_next_order()
    with pytest.raises(UnknownOutcome):
        ex.buy_limit("KRW-BTC", 99_000_000, 0.0005)
    assert ex.coin > 0
