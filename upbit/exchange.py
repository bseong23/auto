"""거래소 어댑터 — 봇과 업비트 API 사이의 유일한 경계.

## 왜 pyupbit 을 직접 안 쓰나

pyupbit 의 주문 메서드는 **실패를 조용히 삼킨다**:

    def buy_market_order(self, ticker, price, ...):
        try:
            ...
        except Exception as x:
            print(x.__class__.__name__)
            return None          # ← 오류 종류가 통째로 사라진다

pyupbit 내부의 `error_handler` 는 `InsufficientFundsBid`, `TooManyRequests`,
`ExpiredAccessKey` 같은 **타입별 예외를 제대로 던지는데**, 주문 메서드가 그걸
전부 잡아서 `None` 으로 뭉개버린다. 그러면 호출하는 쪽에서는
"잔고가 부족했나 / 레이트리밋인가 / 네트워크가 끊겼나"를 구분할 수 없다.

구분이 안 되면 대응도 못 한다:
- 레이트리밋인데 바로 재시도하면 더 오래 막힌다
- 인증 오류인데 계속 시도하면 계정이 차단될 수 있다
- 네트워크 오류면 재시도해야 한다

그래서 pyupbit 의 **내부 요청 함수를 직접 호출**한다. 그쪽은 예외가 살아있고,
`requests` 로 kwargs 가 그대로 전달돼서 **타임아웃도 걸 수 있다**
(pyupbit 공개 메서드에는 타임아웃이 아예 없다 — 연결이 멈추면 봇이 영원히 대기한다).

## 제일 중요한 개념: UnknownOutcome

주문 요청이 타임아웃되면 **주문이 들어갔는지 안 들어갔는지 알 수 없다.**
요청은 도착했는데 응답만 못 받았을 수도 있다.

이걸 "실패"로 처리하면 이미 체결된 주문을 못 본 채로 재주문해서 **두 번 산다**.
그래서 별도의 `UnknownOutcome` 으로 구분하고, 호출하는 쪽이 반드시
**실제 잔고·주문내역을 조회해 맞춰보게(reconcile)** 한다.
"""
from __future__ import annotations

import math
import uuid as uuid_module
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: 업비트 최소 주문 금액 (원화마켓)
MIN_ORDER_KRW = 5_000

#: 네트워크 타임아웃 (연결, 읽기) 초
DEFAULT_TIMEOUT = (5, 15)

API_BASE = "https://api.upbit.com/v1"


# ---------------------------------------------------------------- 오류 분류

class ExchangeError(RuntimeError):
    """거래소 관련 오류의 부모."""


class AuthError(ExchangeError):
    """인증 실패 — 키 만료, IP 미등록, 권한 부족.

    **재시도하면 안 된다.** 계속 시도하면 계정이 차단될 수 있다. 즉시 멈추고 사람을 부른다.
    """


class OrderRejected(ExchangeError):
    """거래소가 주문을 거부했다 — 잔고 부족, 최소금액 미달 등.

    주문은 **확실히 안 들어갔다.** 재시도해도 조건이 그대로면 또 거부된다.
    """


class RateLimited(ExchangeError):
    """요청이 너무 잦다. 기다렸다 재시도해야 한다."""


class TransientError(ExchangeError):
    """일시적 오류 — 네트워크, 서버 5xx. 재시도하면 될 수 있다."""


class UnknownOutcome(ExchangeError):
    """주문이 들어갔는지 **알 수 없다**.

    타임아웃처럼 요청은 갔는데 응답을 못 받은 경우. 실패로 단정하고 재주문하면
    두 번 살 수 있다. 반드시 실제 잔고/주문내역을 조회해 확인해야 한다.
    """


#: 업비트 오류 이름 → 우리 분류
ERROR_MAP = {
    # 인증 — 재시도 금지
    "invalid_access_key": AuthError,
    "expired_access_key": AuthError,
    "jwt_verification": AuthError,
    "no_authorization_i_p": AuthError,
    "out_of_scope": AuthError,
    "invalid_query_payload": AuthError,
    # 주문 거부 — 조건이 그대로면 재시도 무의미
    "insufficient_funds_bid": OrderRejected,
    "insufficient_funds_ask": OrderRejected,
    "under_min_total_bid": OrderRejected,
    "under_min_total_ask": OrderRejected,
    "create_bid_error": OrderRejected,
    "create_ask_error": OrderRejected,
    "validation_error": OrderRejected,
    # 재시도 가능
    "nonce_used": TransientError,
    "too_many_requests": RateLimited,
}


def classify(name: str, message: str = "") -> ExchangeError:
    """업비트 오류 이름을 우리 예외로 바꾼다."""
    kind = ERROR_MAP.get(name, ExchangeError)
    return kind(f"{name}: {message}" if message else name)


# ---------------------------------------------------------------- 주문 결과

@dataclass
class OrderResult:
    """주문 하나의 상태. 거래소가 알려준 사실만 담는다."""

    uuid: str
    side: str  # "bid"(매수) / "ask"(매도)
    state: str  # "wait"(미체결) / "done"(체결완료) / "cancel"(취소)
    executed_volume: float = 0.0  # 체결된 코인 수량
    executed_krw: float = 0.0  # 체결된 원화 금액 (수수료 제외 전)
    paid_fee: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_filled(self) -> bool:
        """체결이 끝났나. 부분체결 후 취소된 것도 '더 안 채워짐'이므로 포함."""
        return self.state in ("done", "cancel") and self.executed_volume > 0

    @property
    def is_pending(self) -> bool:
        return self.state == "wait"

    @property
    def is_empty(self) -> bool:
        """한 조각도 체결되지 않았다."""
        return self.executed_volume == 0

    @property
    def avg_price(self) -> float | None:
        if self.executed_volume == 0:
            return None
        return self.executed_krw / self.executed_volume

    def describe(self) -> str:
        if self.is_empty:
            return f"{self.side} 미체결({self.state})"
        price = self.avg_price
        return (
            f"{self.side} {self.state} — {self.executed_volume:.8f}개 "
            f"@ {price:,.0f}원 = {self.executed_krw:,.0f}원 (수수료 {self.paid_fee:,.0f}원)"
        )


def parse_order(payload: dict) -> OrderResult:
    """업비트 주문 응답(dict)을 OrderResult 로 변환."""
    executed_volume = float(payload.get("executed_volume") or 0)
    # 시장가 매수는 executed_funds 가 없을 수 있어 trades 로 합산한다
    executed_krw = payload.get("executed_funds")
    if executed_krw is None:
        trades = payload.get("trades") or []
        executed_krw = sum(float(t.get("funds", 0)) for t in trades)
    return OrderResult(
        uuid=payload.get("uuid", ""),
        side=payload.get("side", ""),
        state=payload.get("state", ""),
        executed_volume=executed_volume,
        executed_krw=float(executed_krw or 0),
        paid_fee=float(payload.get("paid_fee") or 0),
        raw=payload,
    )


# ---------------------------------------------------------------- 인터페이스

@runtime_checkable
class Exchange(Protocol):
    """봇이 거래소에 요구하는 최소 기능.

    이 인터페이스만 지키면 진짜 업비트든 가짜 거래소든 봇은 구분하지 못한다.
    덕분에 주문 실패·부분체결·레이트리밋을 실제 돈 없이 테스트할 수 있다.
    """

    def get_krw_balance(self) -> float: ...
    def get_coin_balance(self, ticker: str) -> float: ...
    def get_current_price(self, ticker: str) -> float: ...
    def get_best_quotes(self, ticker: str) -> tuple[float, float]: ...  # (최우선 매수호가, 최우선 매도호가)
    def buy_market(self, ticker: str, krw: float) -> OrderResult: ...
    def sell_market(self, ticker: str, volume: float) -> OrderResult: ...
    def buy_limit(self, ticker: str, price: float, volume: float) -> OrderResult: ...
    def sell_limit(self, ticker: str, price: float, volume: float) -> OrderResult: ...
    def cancel_order(self, order_uuid: str) -> OrderResult: ...
    def get_order(self, order_uuid: str) -> OrderResult: ...


#: 업비트 KRW 마켓 호가 단위 (가격 하한, 호가 단위). 2024-10 개편 기준.
KRW_TICK_TABLE = (
    (2_000_000, 1_000), (1_000_000, 500), (500_000, 100), (100_000, 50),
    (10_000, 10), (1_000, 1), (100, 0.1), (10, 0.01), (1, 0.001), (0, 0.0001),
)


def krw_tick_size(price: float) -> float:
    """가격대별 호가 단위. pyupbit.get_tick_size 는 이게 아니라 '호가에 맞춰 반올림한 가격'을 준다."""
    for floor, tick in KRW_TICK_TABLE:
        if price >= floor:
            return tick
    return 0.0001


def tick_ratio(price: float) -> float:
    """호가 단위 / 가격. 스프레드가 구조적으로 얼마나 넓을 수밖에 없는지의 하한.

    BTC(1억) 0.001%, DOGE(115원) 0.087% — 저가 코인은 스프레드만으로 잦은 매매가 불가능하다.
    """
    return krw_tick_size(price) / price


def align_to_tick(price: float, side: str) -> float:
    """지정가를 호가 단위에 맞춘다. 매수는 내림, 매도는 올림 — 둘 다 **메이커 쪽으로** 보수적."""
    tick = krw_tick_size(price)
    steps = price / tick
    aligned = (math.floor(steps) if side == "bid" else math.ceil(steps)) * tick
    return round(aligned, 4)


def floor_volume(volume: float, decimals: int = 8) -> float:
    """업비트 수량 정밀도(소수 8자리)에 맞춰 내림. 올림하면 잔고 초과로 거부된다."""
    factor = 10 ** decimals
    return math.floor(volume * factor) / factor


def coin_of(ticker: str) -> str:
    """'KRW-BTC' → 'BTC'"""
    return ticker.split("-")[-1]


# ---------------------------------------------------------------- 진짜 업비트

class UpbitExchange:
    """실제 업비트. pyupbit 의 내부 요청 함수를 써서 오류 타입과 타임아웃을 살린다.

    pyupbit 의 공개 주문 메서드(`buy_market_order` 등)는 쓰지 않는다 —
    모듈 최상단 설명 참고.
    """

    def __init__(self, access_key: str, secret_key: str, timeout=DEFAULT_TIMEOUT):
        import pyupbit
        from pyupbit import request_api

        self._pyupbit = pyupbit
        self._request_api = request_api
        self._timeout = timeout
        self._client = pyupbit.Upbit(access_key, secret_key)
        self._check_compatibility()

    def _check_compatibility(self) -> None:
        """pyupbit 내부 구조에 의존하므로, 바뀌었으면 조용히 틀리는 대신 시끄럽게 죽는다."""
        missing = [
            name
            for name, obj in (
                ("_call_post", getattr(self._request_api, "_call_post", None)),
                ("_call_get", getattr(self._request_api, "_call_get", None)),
                ("_call_delete", getattr(self._request_api, "_call_delete", None)),
                ("_request_headers", getattr(self._client, "_request_headers", None)),
            )
            if obj is None
        ]
        if missing:
            raise ExchangeError(
                f"pyupbit 내부 구조가 바뀌었다 (없는 것: {', '.join(missing)}). "
                f"upbit/exchange.py 의 UpbitExchange 를 고쳐야 한다. "
                f"설치된 버전: {getattr(self._pyupbit, '__version__', '알 수 없음')}"
            )

    # ---- 저수준 요청 ----

    def _request(self, method: str, path: str, params: dict | None = None) -> Any:
        """서명된 요청 하나. 오류를 우리 분류로 바꿔서 던진다."""
        import requests

        url = f"{API_BASE}{path}"
        headers = self._client._request_headers(params) if params else self._client._request_headers()
        caller = {"POST": self._request_api._call_post,
                  "GET": self._request_api._call_get,
                  "DELETE": self._request_api._call_delete}[method]

        kwargs: dict[str, Any] = {"headers": headers, "timeout": self._timeout}
        if params:
            if method == "POST":
                import json as json_module

                headers["Accept"] = "application/json"
                headers["Content-Type"] = "application/json"
                kwargs["data"] = json_module.dumps(params)
            else:
                kwargs["params"] = params

        try:
            response = caller(url, **kwargs)
        except requests.Timeout as exc:
            # 주문이 들어갔는지 알 수 없다 — 실패로 단정하면 두 번 살 수 있다
            raise UnknownOutcome(f"{method} {path} 타임아웃: {exc}") from exc
        except requests.ConnectionError as exc:
            if method in ("POST", "DELETE"):
                raise UnknownOutcome(f"{method} {path} 연결 실패: {exc}") from exc
            raise TransientError(f"{method} {path} 연결 실패: {exc}") from exc
        except Exception as exc:
            raise self._translate(exc) from exc

        return response.json()

    def _translate(self, exc: Exception) -> ExchangeError:
        """pyupbit 예외를 우리 분류로. 예외 클래스에 붙은 name 을 그대로 쓴다."""
        name = getattr(exc, "name", "") or type(exc).__name__
        message = getattr(exc, "msg", "") or str(exc)
        if name in ERROR_MAP:
            return classify(name, message)
        lowered = type(exc).__name__.lower()
        if "toomanyrequests" in lowered:
            return RateLimited(message or "요청 한도 초과")
        if any(k in lowered for k in ("accesskey", "jwt", "authorization", "scope")):
            return AuthError(message or type(exc).__name__)
        return TransientError(f"{type(exc).__name__}: {message}")

    # ---- 조회 ----

    def _balances(self) -> list[dict]:
        payload = self._request("GET", "/accounts")
        if not isinstance(payload, list):
            raise ExchangeError(f"잔고 응답 형식이 예상과 다르다: {payload!r}")
        return payload

    def get_krw_balance(self) -> float:
        for row in self._balances():
            if row.get("currency") == "KRW":
                return float(row.get("balance", 0))
        return 0.0

    def get_coin_balance(self, ticker: str) -> float:
        want = coin_of(ticker)
        for row in self._balances():
            if row.get("currency") == want:
                return float(row.get("balance", 0))
        return 0.0

    def get_current_price(self, ticker: str) -> float:
        price = self._pyupbit.get_current_price(ticker)
        if price is None:
            raise TransientError(f"{ticker} 현재가 조회 실패")
        return float(price)

    def get_order(self, order_uuid: str) -> OrderResult:
        return parse_order(self._request("GET", "/order", {"uuid": order_uuid}))

    def get_best_quotes(self, ticker: str) -> tuple[float, float]:
        book = self._pyupbit.get_orderbook(ticker)
        if not book or not book.get("orderbook_units"):
            raise TransientError(f"{ticker} 호가 조회 실패")
        top = book["orderbook_units"][0]
        return float(top["bid_price"]), float(top["ask_price"])

    # ---- 주문 ----

    def buy_limit(self, ticker: str, price: float, volume: float) -> OrderResult:
        """지정가 매수. 호가에 맞춰 내림 정렬한다."""
        price = align_to_tick(price, "bid")
        volume = floor_volume(volume)
        if price * volume < MIN_ORDER_KRW:
            raise OrderRejected(f"최소 주문금액 미달: {price * volume:,.0f}원 < {MIN_ORDER_KRW:,}원")
        return parse_order(self._request("POST", "/orders", {
            "market": ticker, "side": "bid", "ord_type": "limit",
            "price": f"{price:.4f}".rstrip("0").rstrip("."), "volume": f"{volume:.8f}",
        }))

    def sell_limit(self, ticker: str, price: float, volume: float) -> OrderResult:
        """지정가 매도. 호가에 맞춰 올림 정렬한다."""
        price = align_to_tick(price, "ask")
        volume = floor_volume(volume)
        if volume <= 0:
            raise OrderRejected(f"매도 수량이 0 이하다: {volume}")
        if price * volume < MIN_ORDER_KRW:
            raise OrderRejected(f"최소 주문금액 미달: {price * volume:,.0f}원 < {MIN_ORDER_KRW:,}원")
        return parse_order(self._request("POST", "/orders", {
            "market": ticker, "side": "ask", "ord_type": "limit",
            "price": f"{price:.4f}".rstrip("0").rstrip("."), "volume": f"{volume:.8f}",
        }))

    def cancel_order(self, order_uuid: str) -> OrderResult:
        """미체결 주문 취소. 이미 체결된 주문이면 거래소가 거부한다 → 호출자가 재조회해야 한다."""
        return parse_order(self._request("DELETE", "/order", {"uuid": order_uuid}))

    def buy_market(self, ticker: str, krw: float) -> OrderResult:
        """시장가 매수. krw 원어치를 산다."""
        if krw < MIN_ORDER_KRW:
            raise OrderRejected(f"최소 주문금액 미달: {krw:,.0f}원 < {MIN_ORDER_KRW:,}원")
        return parse_order(
            self._request(
                "POST", "/orders",
                {"market": ticker, "side": "bid", "price": str(krw), "ord_type": "price"},
            )
        )

    def sell_market(self, ticker: str, volume: float) -> OrderResult:
        """시장가 매도. volume 개를 판다."""
        if volume <= 0:
            raise OrderRejected(f"매도 수량이 0 이하다: {volume}")
        return parse_order(
            self._request(
                "POST", "/orders",
                {"market": ticker, "side": "ask", "volume": f"{volume:.8f}", "ord_type": "market"},
            )
        )

    def api_key_info(self) -> list[dict]:
        """API 키 목록과 만료일 — 시작할 때 확인해 경고하는 용도."""
        return self._request("GET", "/api_keys")
