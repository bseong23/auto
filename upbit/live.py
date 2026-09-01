"""5단계: 소액 실전 — 여기서만 API 키가 필요하다.

⚠️ 이 파일은 실제 돈을 움직인다.

## 설계 원칙

**1. 잔고가 진실이다.**
상태파일에 "보유 중"이라고 적혀 있어도, 실제로 뭘 들고 있는지는 거래소만 안다.
파일과 실제가 어긋나는 경로는 많다 — 주문 실패, 사람이 앱에서 직접 매도,
봇 중복 실행, 부분 체결, 상태파일 삭제. 그래서 포지션은 **매번 잔고에서 읽는다.**
상태파일에는 거래소가 모르는 것(진입가, 손절선)만 저장한다.

**2. 주문은 체결을 확인해야 끝난 것이다.**
주문 요청이 성공했다고 체결된 게 아니다. 미체결로 남거나 부분만 체결될 수 있다.
`get_order` 로 폴링해서 실제 체결을 확인한다.

**3. '알 수 없음'은 '실패'가 아니다.**
타임아웃이 나면 주문이 들어갔는지 알 수 없다. 실패로 단정하고 재주문하면
두 번 산다. 이 경우엔 잔고를 다시 읽어서 맞춘다.

**4. 모의 모드도 같은 코드로 돈다.**
모의(dry-run)는 주문만 흉내내는 게 아니라 가짜 거래소를 상대로 **실제와 똑같은
주문 경로**를 탄다. 그래야 모의에서 검증한 게 실전에서도 유효하다.

## 안전장치

- 기본이 모의 — 실주문은 `.env` 의 `UPBIT_ALLOW_LIVE=true` **그리고** `live=True` 둘 다여야 열린다
- 1회 주문 금액 상한 (`UPBIT_MAX_ORDER_KRW`)
- 키는 환경변수에서만 읽는다. 코드에 하드코딩·깃 커밋 금지

그리고 제일 중요한 건 코드가 아니라 원칙이다:
**잃어도 되는 소액만.** 적금·투자 자금은 절대 여기 넣지 말 것.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .data import get_ohlcv
from .exchange import (
    MIN_ORDER_KRW,
    AuthError,
    Exchange,
    ExchangeError,
    OrderRejected,
    OrderResult,
    RateLimited,
    TransientError,
    UnknownOutcome,
    align_to_tick,
    coin_of,
    floor_volume,
)
from .fake_exchange import FakeExchange, PaperExchange
from .journal import record_fill
from .notify import Notifier
from .risk import RiskRules
from .strategies.base import Strategy

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "bot_state.json"
LOG_PATH = ROOT / "reports" / "trades.log"

#: 최소 주문금액(5,000원)으로 사면 수수료를 떼고 4,997원어치가 남아
#: **그 즉시 매도 불가능한 먼지**가 된다. 매도 시점에 5,000원을 넘기려면
#: 수수료와 가격 변동만큼 여유가 있어야 한다. 10% 여유 = 약 5% 하락까지 버틴다.
MIN_SAFE_ORDER_KRW = int(MIN_ORDER_KRW * 1.1)

log = logging.getLogger("upbit.live")


class SafetyError(RuntimeError):
    """안전장치에 걸렸을 때. 이 예외가 뜨면 주문은 나가지 않았다."""


@dataclass
class Execution:
    """한 번의 매수/매도 의도가 여러 주문(지정가 + 시장가 마무리)으로 체결된 결과를 합친다.

    장부·알림·상태 기록은 이 합계를 본다. `legs` 에 개별 주문이 남아 있어
    "지정가로 얼마나 잡았고 시장가로 얼마나 마무리했나"를 나중에 잴 수 있다.
    """

    side: str
    legs: list  # [("limit" | "market", OrderResult), ...]

    @property
    def filled_legs(self) -> list:
        return [(kind, o) for kind, o in self.legs if not o.is_empty]

    @property
    def executed_volume(self) -> float:
        return sum(o.executed_volume for _, o in self.filled_legs)

    @property
    def executed_krw(self) -> float:
        return sum(o.executed_krw for _, o in self.filled_legs)

    @property
    def paid_fee(self) -> float:
        return sum(o.paid_fee for _, o in self.filled_legs)

    @property
    def avg_price(self) -> float | None:
        return self.executed_krw / self.executed_volume if self.executed_volume else None

    @property
    def is_empty(self) -> bool:
        return self.executed_volume == 0

    @property
    def uuid(self) -> str:
        ids = [o.uuid for _, o in self.filled_legs] or [o.uuid for _, o in self.legs]
        return "+".join(ids)

    @property
    def order_type(self) -> str:
        kinds = {kind for kind, _ in self.filled_legs}
        if kinds == {"limit"}:
            return "limit"
        if kinds == {"market"}:
            return "market"
        return "limit+market" if kinds else "none"

    @property
    def limit_fill_pct(self) -> float | None:
        """체결 수량 중 지정가(메이커)로 잡은 비율. 지정가를 안 썼으면 None."""
        if not any(kind == "limit" for kind, _ in self.legs):
            return None
        total = self.executed_volume
        limit_part = sum(o.executed_volume for kind, o in self.filled_legs if kind == "limit")
        return limit_part / total if total else 0.0

    def describe(self) -> str:
        if self.is_empty:
            return f"{self.side} 미체결"
        tag = {"limit": "지정가", "market": "시장가", "limit+market": "지정가+시장가"}[self.order_type]
        pct = f" (지정가 {self.limit_fill_pct:.0%})" if self.order_type == "limit+market" else ""
        return (f"{self.side} {tag}{pct} — {self.executed_volume:.8f}개 @ {self.avg_price:,.0f}원 "
                f"= {self.executed_krw:,.0f}원 (수수료 {self.paid_fee:,.0f}원)")


# ---------------------------------------------------------------- 설정

@dataclass
class Config:
    access_key: str = ""
    secret_key: str = ""
    max_order_krw: float = 10_000
    allow_live: bool = False
    #: 모의 모드에서 쓸 가상 원화 (실제 돈 아님)
    paper_krw: float = 1_000_000
    #: "limit"(지정가 → 미체결분 시장가 마무리) 또는 "market"
    order_type: str = "limit"
    #: 지정가 체결을 기다리는 최대 초. 지나면 취소하고 시장가로 마무리한다
    limit_wait_sec: int = 90

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(ROOT / ".env")
        return cls(
            access_key=os.getenv("UPBIT_ACCESS_KEY", "").strip(),
            secret_key=os.getenv("UPBIT_SECRET_KEY", "").strip(),
            max_order_krw=float(os.getenv("UPBIT_MAX_ORDER_KRW", "10000")),
            allow_live=os.getenv("UPBIT_ALLOW_LIVE", "false").strip().lower() == "true",
            paper_krw=float(os.getenv("UPBIT_PAPER_KRW", "1000000")),
            order_type=os.getenv("UPBIT_ORDER_TYPE", "limit").strip().lower() or "limit",
            limit_wait_sec=int(os.getenv("UPBIT_LIMIT_WAIT_SEC", "90")),
        )

    @property
    def has_keys(self) -> bool:
        return bool(self.access_key and self.secret_key)


def setup_logging(verbose: bool = True) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


# ---------------------------------------------------------------- 상태

def default_state() -> dict:
    """거래소가 모르는 것만 담는다. 포지션은 여기 없다 — 잔고에서 읽는다."""
    return {
        "entry_price": None,     # 손절선 계산 기준
        "stop_price": None,      # 현재 손절선 (추적이면 갱신됨)
        "high_water": None,      # 진입 후 최고가
        "blocked": False,        # 손절 직후 재진입 차단
        "paper": None,           # 모의 모드 가상 잔고
        "history": [],
    }


def load_state(path: Path | None = None) -> dict:
    target = path or STATE_PATH
    if not target.exists():
        return default_state()
    state = default_state()
    state.update(json.loads(target.read_text(encoding="utf-8")))
    return state


def save_state(state: dict, path: Path | None = None) -> None:
    target = path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 거래소 준비

def build_exchange(config: Config, live: bool, ticker: str, state: dict) -> Exchange:
    """실전이면 진짜 업비트, 모의면 가짜 거래소. 봇 코드는 둘을 구분하지 않는다."""
    if live:
        if not config.allow_live:
            raise SafetyError(
                "실주문이 잠겨 있다. .env에서 UPBIT_ALLOW_LIVE=true 로 바꿔야 열린다.\n"
                "(그 전에 백테스트를 충분히 돌렸는지, 잃어도 되는 돈인지 다시 생각할 것.)"
            )
        if not config.has_keys:
            raise SafetyError(
                "API 키가 없다. .env.example을 .env로 복사하고 "
                "UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY를 채울 것."
            )
        from .exchange import UpbitExchange

        return UpbitExchange(config.access_key, config.secret_key)

    # 모의 — 실제 시세로 돌되 주문은 가상 잔고에만 반영. 가상 잔고는 실행 간에 유지된다.
    paper = state.get("paper") or {"krw": config.paper_krw, "coin": 0.0}
    return PaperExchange(krw=float(paper["krw"]), coin=float(paper["coin"]))


# ---------------------------------------------------------------- 트레이더

class Trader:
    """전략 신호를 받아 주문까지 연결하는 실행기."""

    #: 체결을 기다리는 최대 시간(초)
    FILL_TIMEOUT = 30

    def __init__(
        self,
        strategy: Strategy,
        exchange: Exchange,
        ticker: str = "KRW-BTC",
        interval: str = "day",
        order_krw: float = MIN_ORDER_KRW,
        risk: RiskRules | None = None,
        live: bool = False,
        config: Config | None = None,
        state_path: Path | None = None,
        notifier: Notifier | None = None,
        journal_path: Path | None = None,
        order_type: str | None = None,
        limit_wait_sec: int | None = None,
        sleep=time.sleep,
    ):
        self.strategy = strategy
        self.exchange = exchange
        self.ticker = ticker
        self.interval = interval
        self.config = config or Config.from_env()
        self.risk = risk or RiskRules()
        self.live = live
        self.state_path = state_path
        self.notifier = notifier or Notifier()
        self.journal_path = journal_path  # None 이면 reports/fills.csv
        self.order_type = (order_type or self.config.order_type)
        if self.order_type not in ("limit", "market"):
            raise SafetyError(f"order_type 은 'limit' 또는 'market' 이어야 한다: {self.order_type!r}")
        self.limit_wait_sec = self.config.limit_wait_sec if limit_wait_sec is None else limit_wait_sec
        self._sleep = sleep
        self.order_krw = self._validate_order_size(order_krw)

    # ---------- 안전장치 ----------

    def _validate_order_size(self, amount: float) -> float:
        if amount < MIN_SAFE_ORDER_KRW:
            raise SafetyError(
                f"1회 주문금액이 너무 작다: {amount:,.0f}원 (최소 {MIN_SAFE_ORDER_KRW:,}원)\n"
                f"업비트 최소 주문금액은 {MIN_ORDER_KRW:,}원이지만, 딱 그만큼 사면 수수료를 떼고 "
                f"{MIN_ORDER_KRW * 0.9995:,.0f}원어치가 남아 **즉시 매도할 수 없는 먼지**가 된다. "
                f"매도 시점에도 {MIN_ORDER_KRW:,}원을 넘기려면 여유가 필요하다."
            )
        if amount > self.config.max_order_krw:
            raise SafetyError(
                f"1회 주문금액 {amount:,.0f}원이 상한({self.config.max_order_krw:,.0f}원)을 넘는다. "
                ".env의 UPBIT_MAX_ORDER_KRW를 확인할 것 — 오타로 0 하나 더 붙었을 수 있다."
            )
        return float(amount)

    # ---------- 조회 ----------

    def current_price(self) -> float:
        return self.exchange.get_current_price(self.ticker)

    def held_volume(self) -> float:
        return self.exchange.get_coin_balance(self.ticker)

    def current_position(self, price: float) -> int:
        """**잔고가 진실이다.** 상태파일이 아니라 실제 보유량으로 판단한다.

        평가액이 최소 주문금액 미만이면 팔 수도 없는 먼지이므로 '현금'으로 본다.
        (먼지에 더 사면 합쳐져서 다시 팔 수 있게 된다.)
        """
        return 1 if self.dust_krw(price) == 0 and self.held_volume() > 0 else 0

    def dust_krw(self, price: float) -> float:
        """팔 수 없는 잔량의 평가액. 0이면 먼지가 아니다."""
        value = self.held_volume() * price
        return value if 0 < value < MIN_ORDER_KRW else 0.0

    def current_signal(self, count: int = 200) -> tuple[int, datetime]:
        """최근 **확정** 캔들 기준 목표 포지션.

        마지막 캔들은 아직 진행 중이라 신호로 쓰지 않는다.
        진행 중인 봉으로 판단하면 봉이 끝날 때까지 신호가 계속 바뀐다.
        """
        df = get_ohlcv(self.ticker, self.interval, count, use_cache=False)
        closed = df.iloc[:-1]
        positions = self.strategy.generate_positions(closed)
        return int(positions.iloc[-1]), closed.index[-1].to_pydatetime()

    # ---------- 주문 ----------

    def _await_fill(self, order: OrderResult, timeout: float | None = None) -> OrderResult:
        """체결될 때까지 폴링한다. 주문 요청 성공 ≠ 체결."""
        if not order.is_pending:
            return order
        limit = self.FILL_TIMEOUT if timeout is None else timeout
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            self._sleep(1)
            try:
                order = self.exchange.get_order(order.uuid)
            except (TransientError, RateLimited) as exc:
                log.warning("체결 확인 실패, 재시도: %s", exc)
                continue
            if not order.is_pending:
                return order
        return order

    def _settle_unfilled(self, order: OrderResult) -> OrderResult:
        """대기 시간이 지난 지정가를 취소하고 **실제 체결량을 재조회**한다.

        취소 요청과 체결이 경합할 수 있다 — 취소하려는 순간 체결돼 버리면 거래소가 취소를
        거부한다. 그때 '취소됐으니 0체결'이라고 믿고 시장가를 또 내면 **두 번 산다.**
        그래서 취소 결과가 어떻든 반드시 재조회한 값을 진실로 쓴다.
        """
        if not order.is_pending:
            return order
        try:
            self.exchange.cancel_order(order.uuid)
        except OrderRejected:
            log.info("취소 시점에 이미 체결돼 있었다 (%s) — 재조회한다", order.uuid)
        except (TransientError, RateLimited, UnknownOutcome) as exc:
            log.warning("취소 요청 결과를 모른다: %s — 재조회한다", exc)
        for _ in range(3):
            try:
                return self.exchange.get_order(order.uuid)
            except (TransientError, RateLimited) as exc:
                log.warning("재조회 실패, 재시도: %s", exc)
                self._sleep(1)
        log.error("주문 %s 의 최종 상태를 확인하지 못했다 — 다음 사이클에 잔고로 맞춘다", order.uuid)
        return order

    def _handle_order_error(self, what: str, exc: Exception) -> None:
        if isinstance(exc, UnknownOutcome):
            log.error("%s 결과를 알 수 없다: %s — 잔고를 다시 읽어 맞춘다", what, exc)
        elif isinstance(exc, OrderRejected):
            log.error("%s 거부됨(주문 안 들어감): %s", what, exc)
        else:
            log.error("%s 실패: %s", what, exc)

    def buy(self, urgent: bool = False) -> Execution | None:
        """매수. 기본은 지정가(메이커) → 대기 → 미체결분 시장가. 실패하면 None — **상태를 바꾸면 안 된다.**

        지정가는 스프레드(왕복 ~0.04%)를 아끼지만 체결이 안 될 수 있다. 조사(08)에서 실행이
        하루 늦으면 -12%p 였다. 그래서 끝까지 기다리지 않고 시장가로 마무리한다.
        """
        mode = "실전" if self.live else "모의"
        legs: list = []
        budget = self.order_krw

        if self.order_type == "limit" and not urgent:
            try:
                bid, _ask = self.exchange.get_best_quotes(self.ticker)
                price = align_to_tick(bid, "bid")          # 최우선 매수호가에 줄을 선다 (메이커)
                volume = floor_volume(budget / price)
                log.warning("[%s] 지정가 매수 %s %.8f개 @ %s (대기 %ds)",
                            mode, self.ticker, volume, f"{price:,.0f}", self.limit_wait_sec)
                order = self.exchange.buy_limit(self.ticker, price, volume)
            except AuthError:
                raise
            except ExchangeError as exc:
                self._handle_order_error("지정가 매수", exc)
                return None
            order = self._await_fill(order, timeout=self.limit_wait_sec)
            order = self._settle_unfilled(order)
            legs.append(("limit", order))
            budget -= order.executed_krw + order.paid_fee
            if budget < MIN_ORDER_KRW:
                if budget > 0 and not order.is_empty:
                    log.info("지정가로 대부분 체결, 잔여 %s원은 최소 주문금액 미만이라 남긴다", f"{budget:,.0f}")
                return self._finish("bid", legs)

        try:
            log.warning("[%s] 시장가 매수 %s %s원%s", mode, self.ticker, f"{budget:,.0f}",
                        " (지정가 미체결분 마무리)" if legs else "")
            order = self.exchange.buy_market(self.ticker, budget)
        except AuthError:
            raise
        except ExchangeError as exc:
            self._handle_order_error("시장가 매수", exc)
            return self._finish("bid", legs) if legs else None
        legs.append(("market", self._await_fill(order)))
        return self._finish("bid", legs)

    def sell(self, urgent: bool = False) -> Execution | None:
        """보유 전량 매도. 손절·긴급(urgent)은 처음부터 시장가 — 떨어지는 장에서 매도 지정가는 안 잡힌다."""
        mode = "실전" if self.live else "모의"
        volume = self.held_volume()
        price = self.current_price()
        if volume * price < MIN_ORDER_KRW:
            log.warning("매도할 수량이 최소 주문금액 미만이다 (평가액 %s원) — 먼지로 남긴다",
                        f"{volume * price:,.0f}")
            return None
        legs: list = []

        if self.order_type == "limit" and not urgent:
            try:
                _bid, ask = self.exchange.get_best_quotes(self.ticker)
                limit_price = align_to_tick(ask, "ask")   # 최우선 매도호가에 줄을 선다 (메이커)
                log.warning("[%s] 지정가 매도 %s %.8f개 @ %s (대기 %ds)",
                            mode, self.ticker, volume, f"{limit_price:,.0f}", self.limit_wait_sec)
                order = self.exchange.sell_limit(self.ticker, limit_price, floor_volume(volume))
            except AuthError:
                raise
            except ExchangeError as exc:
                self._handle_order_error("지정가 매도", exc)
                return None
            order = self._await_fill(order, timeout=self.limit_wait_sec)
            order = self._settle_unfilled(order)
            legs.append(("limit", order))
            volume = self.held_volume()                     # 잔여는 잔고가 진실
            price = self.current_price()
            if volume * price < MIN_ORDER_KRW:
                return self._finish("ask", legs)

        try:
            log.warning("[%s] 시장가 매도 %s %.8f개%s", mode, self.ticker, volume,
                        " (지정가 미체결분 마무리)" if legs else "")
            order = self.exchange.sell_market(self.ticker, floor_volume(volume))
        except AuthError:
            raise
        except ExchangeError as exc:
            self._handle_order_error("시장가 매도", exc)
            return self._finish("ask", legs) if legs else None
        legs.append(("market", self._await_fill(order)))
        return self._finish("ask", legs)

    def _finish(self, side: str, legs: list) -> Execution | None:
        execution = Execution(side=side, legs=legs)
        if execution.is_empty:
            log.error("%s 주문이 한 조각도 체결되지 않았다", "매수" if side == "bid" else "매도")
            return None
        log.info("%s 체결 — %s", "매수" if side == "bid" else "매도", execution.describe())
        return execution

    # ---------- 1회 실행 ----------

    def step(self) -> dict:
        """신호를 확인하고 필요하면 주문한다. 1회분.

        순서: 잔고 확인 → 상태 대조 → 손절 확인 → 전략 신호 → 주문 → 상태 저장
        """
        state = load_state(self.state_path)
        price = self.current_price()
        held = self.current_position(price)
        dust = self.dust_krw(price)
        if dust:
            log.warning(
                "팔 수 없는 잔량이 있다: %s원 (최소 주문금액 %s원 미만). "
                "다음 매수 때 합쳐지면 다시 팔 수 있다.",
                f"{dust:,.0f}", f"{MIN_ORDER_KRW:,}",
            )
        self._reconcile(state, held, price)

        target, candle_time = self.current_signal()
        log.info(
            "[%s] %s 캔들 %s | 현재가 %s원 | 보유:%d → 목표:%d%s",
            "실전" if self.live else "모의", self.ticker,
            candle_time.strftime("%Y-%m-%d %H:%M"), f"{price:,.0f}", held, target,
            f" | 손절선 {state['stop_price']:,.0f}" if state.get("stop_price") else "",
        )

        action, order, reason = "hold", None, ""

        # 1) 손절이 전략 신호보다 우선한다
        if held == 1 and self._stop_hit(state, price):
            order = self.sell(urgent=True)   # 손절은 시장가 — 지정가 기다릴 때가 아니다
            if order is not None:
                action, reason = "sell", "손절"
                state["blocked"] = True

        # 2) 전략이 현금으로 돌아오면 재진입 차단 해제
        if target == 0:
            state["blocked"] = False

        # 3) 전략 신호
        if action == "hold":
            if target == 1 and held == 0 and not state.get("blocked"):
                order = self.buy()
                if order is not None:
                    action, reason = "buy", "신호"
            elif target == 0 and held == 1:
                order = self.sell()
                if order is not None:
                    action, reason = "sell", "신호"
            elif target == 1 and held == 0 and state.get("blocked"):
                log.info("손절 후 재진입 차단 중 — 전략이 한 번 현금으로 돌아와야 다시 산다")
            else:
                log.info("변화 없음 — 아무것도 안 한다")

        self._apply(state, action, order, price, reason)
        self._persist_paper(state)
        save_state(state, self.state_path)

        return {
            "action": action, "reason": reason, "position": self.current_position(price),
            "price": price, "candle_time": candle_time, "live": self.live,
            "order": order, "stop_price": state.get("stop_price"),
        }

    def check_api_key(self, warn_days: int = 7) -> str | None:
        """API 키 만료가 임박했는지 확인한다. 문제가 있으면 경고 문구를 돌려준다.

        키가 만료되면 봇이 조용히 멈춘다 — 그때 포지션을 들고 있으면 방치된다.
        확인 자체가 실패해도 봇을 막지는 않는다(권한이 없어 조회만 안 될 수도 있다).
        """
        getter = getattr(self.exchange, "api_key_info", None)
        if getter is None:
            return None
        try:
            keys = getter()
        except ExchangeError as exc:
            log.warning("API 키 정보를 확인하지 못했다: %s", exc)
            return None

        from datetime import timedelta

        soonest = None
        for entry in keys or []:
            raw = entry.get("expire_at")
            if not raw:
                continue
            try:
                expires = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            soonest = expires if soonest is None or expires < soonest else soonest

        if soonest is None:
            return None
        remaining = soonest - datetime.now(soonest.tzinfo)
        if remaining <= timedelta(0):
            return f"API 키가 이미 만료됐다 ({soonest:%Y-%m-%d})"
        if remaining <= timedelta(days=warn_days):
            # timedelta.days 는 내림이라 2.99일이 '2일'이 된다.
            # 만료 경고에서 하루를 줄여 말하면 위험하므로 올림으로 표시한다.
            import math

            days = math.ceil(remaining.total_seconds() / 86_400)
            return (f"API 키가 {days}일 뒤 만료된다 ({soonest:%Y-%m-%d}). "
                    "만료되면 봇이 멈추고 포지션이 방치된다 — 미리 갱신할 것.")
        return None

    def panic_sell(self) -> OrderResult | None:
        """긴급 전량 매도 + 상태 정리.

        재진입을 막아둔다 — 급한 상황에 껐다 켰다고 바로 다시 사면 곤란하다.
        전략이 한 번 현금 신호를 내야 차단이 풀린다.
        """
        state = load_state(self.state_path)
        order = self.sell(urgent=True)
        self._apply(state, "sell", order, self.current_price(), "긴급")
        state["blocked"] = True
        self._persist_paper(state)
        save_state(state, self.state_path)
        if order is None:
            log.warning("긴급 매도: 팔 것이 없거나 주문이 실패했다")
        return order

    # ---------- 내부 ----------

    def _reconcile(self, state: dict, held: int, price: float) -> None:
        """상태파일과 실제 잔고가 어긋났으면 **실제 잔고에 맞춘다.**"""
        recorded = state.get("entry_price") is not None
        if held == 1 and not recorded:
            log.warning(
                "잔고에 코인이 있는데 진입 기록이 없다 — 현재가(%s원)를 진입가로 간주한다. "
                "(수동 매수했거나 상태파일이 지워졌을 수 있다)", f"{price:,.0f}"
            )
            self._open(state, price)
        elif held == 0 and recorded:
            log.warning("진입 기록이 있는데 잔고가 비었다 — 기록을 지운다. "
                        "(수동 매도했거나 주문이 실제로는 실패했을 수 있다)")
            self._close(state)

    def _open(self, state: dict, price: float) -> None:
        state["entry_price"] = price
        state["high_water"] = price
        state["stop_price"] = self.risk.stop_price(price, self._atr_value())
        state["blocked"] = False

    def _close(self, state: dict) -> None:
        state["entry_price"] = None
        state["high_water"] = None
        state["stop_price"] = None

    def _atr_value(self) -> float | None:
        """ATR 손절용. 확정 캔들만 쓴다."""
        if not self.risk.needs_atr:
            return None
        from .indicators import atr

        df = get_ohlcv(self.ticker, self.interval, 200, use_cache=False).iloc[:-1]
        value = atr(df, self.risk.atr_window).iloc[-1]
        return None if value != value else float(value)  # NaN 체크

    def _stop_hit(self, state: dict, price: float) -> bool:
        """손절선을 넘었나. 넘기 전이면 추적 손절선을 올린다.

        백테스트는 봉의 저가로 판정하지만 실전은 확인 시점의 현재가로만 본다.
        **봉 중간의 급락은 놓칠 수 있다** — 실전 성적이 백테스트보다 나쁠 수 있는 이유.
        """
        stop = state.get("stop_price")
        if stop is None:
            return False
        if price <= stop:
            log.warning("손절선 이탈 — 현재가 %s ≤ 손절선 %s", f"{price:,.0f}", f"{stop:,.0f}")
            self.notifier.stop_hit(self.ticker, price, stop)
            return True

        if self.risk.trailing:
            high_water = max(state.get("high_water") or price, price)
            raised = self.risk.stop_price(high_water, self._atr_value())
            state["high_water"] = high_water
            if raised is not None and raised > stop:
                log.info("추적 손절선 상향: %s → %s", f"{stop:,.0f}", f"{raised:,.0f}")
                state["stop_price"] = raised
        return False

    def _apply(self, state: dict, action: str, order, price: float, reason: str = "") -> None:
        if action == "buy" and order is not None:
            self._open(state, order.avg_price or price)
        elif action == "sell":
            self._close(state)

        if action != "hold" and order is not None:
            # 실측 슬리피지를 남긴다 — 백테스트 가정(0.05%)이 맞는지 나중에 검증할 자료
            record_fill(self.ticker, action, reason or "신호", price, order, self.live,
                        path=self.journal_path,
                        order_type=getattr(order, "order_type", "market"),
                        limit_fill_pct=getattr(order, "limit_fill_pct", None))
            self.notifier.order_filled(self.ticker, action, reason or "신호", order, self.live)
            state.setdefault("history", []).append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "uuid": order.uuid,
                "volume": order.executed_volume,
                "krw": order.executed_krw,
                "fee": order.paid_fee,
                "avg_price": order.avg_price,
                "live": self.live,
            })

    def _persist_paper(self, state: dict) -> None:
        """모의 모드 가상 잔고를 저장해 실행 간에 이어지게 한다."""
        if isinstance(self.exchange, FakeExchange):
            state["paper"] = {"krw": self.exchange.krw, "coin": self.exchange.coin}
