"""가짜 거래소 — 실제 돈 없이 봇의 실패 대응을 검증한다.

실제 돈으로 처음 돌려보는 게 첫 테스트가 되면 안 된다.
여기서 주문 실패·부분체결·레이트리밋·타임아웃을 마음대로 일으켜서
봇이 올바르게 반응하는지 먼저 확인한다.

    ex = FakeExchange(krw=100_000, price=100_000_000)
    ex.fail_next(OrderRejected("잔고 부족"))     # 다음 주문만 실패
    ex.partial_fill_ratio = 0.5                  # 절반만 체결
    ex.timeout_next_order()                      # 주문 결과를 알 수 없게

수수료·슬리피지도 실제처럼 적용해서, 봇이 "주문한 금액"과
"실제로 산 수량"이 다르다는 걸 제대로 다루는지 본다.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .exchange import (
    MIN_ORDER_KRW,
    ExchangeError,
    OrderRejected,
    OrderResult,
    UnknownOutcome,
    coin_of,
)

#: 업비트 원화마켓 수수료 (편도)
FAKE_FEE = 0.0005


@dataclass
class FakeExchange:
    """설정 가능한 가짜 거래소.

    기본값은 '모든 게 정상'. 실패를 보고 싶으면 명시적으로 켜야 한다.
    """

    krw: float = 1_000_000.0
    coin: float = 0.0
    price: float = 100_000_000.0
    fee: float = FAKE_FEE

    #: 체결 비율 — 1.0이면 전량, 0.5면 절반만 체결(부분체결 시뮬레이션)
    partial_fill_ratio: float = 1.0
    #: 체결가가 주문 시점 가격에서 미끄러지는 정도
    slippage: float = 0.0
    #: True면 주문이 즉시 체결되지 않고 'wait' 상태로 남는다
    leave_pending: bool = False

    #: 기록 — 테스트에서 "몇 번 주문했나"를 확인할 때 쓴다
    orders: dict[str, OrderResult] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    _queued_errors: list[Exception] = field(default_factory=list)
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    # ---------- 실패 주입 ----------

    def fail_next(self, error: Exception) -> None:
        """다음 주문 한 번만 이 오류로 실패시킨다. 여러 번 부르면 순서대로 소비된다."""
        self._queued_errors.append(error)

    def timeout_next_order(self) -> None:
        """다음 주문의 결과를 알 수 없게 만든다 (요청은 갔는데 응답을 못 받은 상황)."""
        self._queued_errors.append(UnknownOutcome("응답 없음 — 주문이 들어갔는지 알 수 없다"))

    def _take_error(self) -> Exception | None:
        return self._queued_errors.pop(0) if self._queued_errors else None

    # ---------- 조회 ----------

    def get_krw_balance(self) -> float:
        self.calls.append(("get_krw_balance", ()))
        return self.krw

    def get_coin_balance(self, ticker: str) -> float:
        self.calls.append(("get_coin_balance", (ticker,)))
        return self.coin

    def get_current_price(self, ticker: str) -> float:
        self.calls.append(("get_current_price", (ticker,)))
        return self.price

    def get_order(self, order_uuid: str) -> OrderResult:
        self.calls.append(("get_order", (order_uuid,)))
        if order_uuid not in self.orders:
            raise ExchangeError(f"없는 주문: {order_uuid}")
        return self.orders[order_uuid]

    # ---------- 주문 ----------

    def buy_market(self, ticker: str, krw: float) -> OrderResult:
        self.calls.append(("buy_market", (ticker, krw)))
        error = self._take_error()
        if error is not None:
            if isinstance(error, UnknownOutcome):
                # 응답은 못 받았지만 주문은 실제로 들어갔다 — 제일 고약한 경우
                self._settle_buy(ticker, krw)
            raise error

        if krw < MIN_ORDER_KRW:
            raise OrderRejected(f"최소 주문금액 미달: {krw:,.0f}원 < {MIN_ORDER_KRW:,}원")
        if krw > self.krw:
            raise OrderRejected(f"잔고 부족: {self.krw:,.0f}원 < {krw:,.0f}원")

        return self._settle_buy(ticker, krw)

    def _settle_buy(self, ticker: str, krw: float) -> OrderResult:
        order_uuid = f"fake-buy-{next(self._counter)}"
        if self.leave_pending:
            result = OrderResult(uuid=order_uuid, side="bid", state="wait")
            self.orders[order_uuid] = result
            self.krw -= krw  # 미체결이어도 주문금액은 묶인다
            return result

        spend = krw * self.partial_fill_ratio
        fill_price = self.price * (1 + self.slippage)
        fee = spend * self.fee
        volume = (spend - fee) / fill_price

        self.krw -= spend
        self.coin += volume

        result = OrderResult(
            uuid=order_uuid, side="bid",
            state="done" if self.partial_fill_ratio >= 1.0 else "cancel",
            executed_volume=volume, executed_krw=spend, paid_fee=fee,
        )
        self.orders[order_uuid] = result
        return result

    def sell_market(self, ticker: str, volume: float) -> OrderResult:
        self.calls.append(("sell_market", (ticker, volume)))
        error = self._take_error()
        if error is not None:
            if isinstance(error, UnknownOutcome):
                self._settle_sell(ticker, volume)
            raise error

        if volume <= 0:
            raise OrderRejected(f"매도 수량이 0 이하다: {volume}")
        if volume > self.coin + 1e-12:
            raise OrderRejected(f"보유 수량 부족: {self.coin:.8f} < {volume:.8f}")
        if volume * self.price < MIN_ORDER_KRW:
            raise OrderRejected(
                f"최소 주문금액 미달: {volume * self.price:,.0f}원 < {MIN_ORDER_KRW:,}원"
            )

        return self._settle_sell(ticker, volume)

    def _settle_sell(self, ticker: str, volume: float) -> OrderResult:
        order_uuid = f"fake-sell-{next(self._counter)}"
        if self.leave_pending:
            result = OrderResult(uuid=order_uuid, side="ask", state="wait")
            self.orders[order_uuid] = result
            self.coin -= volume
            return result

        sold = min(volume * self.partial_fill_ratio, self.coin)
        fill_price = self.price * (1 - self.slippage)
        gross = sold * fill_price
        fee = gross * self.fee

        self.coin -= sold
        self.krw += gross - fee

        result = OrderResult(
            uuid=order_uuid, side="ask",
            state="done" if self.partial_fill_ratio >= 1.0 else "cancel",
            executed_volume=sold, executed_krw=gross, paid_fee=fee,
        )
        self.orders[order_uuid] = result
        return result

    # ---------- 테스트 편의 ----------

    def move_price(self, ratio: float) -> float:
        """가격을 비율만큼 움직인다. 0.95 → 5% 하락."""
        self.price *= ratio
        return self.price

    def fill_pending(self, order_uuid: str) -> OrderResult:
        """미체결로 남겨둔 주문을 체결시킨다."""
        order = self.orders[order_uuid]
        if not order.is_pending:
            return order
        # 간단히: 묶여있던 금액/수량이 그대로 체결된 것으로 처리
        order.state = "done"
        return order

    @property
    def equity(self) -> float:
        return self.krw + self.coin * self.price
