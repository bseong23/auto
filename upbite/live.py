"""5단계: 소액 실전 — 여기서만 API 키가 필요하다.

⚠️ 이 파일은 실제 돈을 움직인다. 안전장치를 겹겹이 걸어놨다:

  1. **기본이 모의(dry-run)** — 주문을 흉내만 내고 실제로 안 낸다.
  2. 실주문은 `.env`의 `UPBIT_ALLOW_LIVE=true` **그리고** `live=True` 둘 다여야 열린다.
  3. 1회 주문 금액은 `UPBIT_MAX_ORDER_KRW` 로 상한을 건다.
  4. 키는 환경변수에서만 읽는다. 코드에 하드코딩·깃 커밋 금지.

그리고 제일 중요한 건 코드가 아니라 원칙이다:
**잃어도 되는 소액만.** 적금·투자 자금은 절대 여기 넣지 말 것.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyupbit
from dotenv import load_dotenv

from .data import get_ohlcv
from .strategies.base import Strategy

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "bot_state.json"
LOG_PATH = ROOT / "reports" / "trades.log"

#: 업비트 최소 주문 금액 (원화마켓)
MIN_ORDER_KRW = 5_000

log = logging.getLogger("upbite.live")


class SafetyError(RuntimeError):
    """안전장치에 걸렸을 때. 이 예외가 뜨면 주문은 나가지 않았다."""


@dataclass
class Config:
    access_key: str = ""
    secret_key: str = ""
    max_order_krw: float = 10_000
    allow_live: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(ROOT / ".env")
        return cls(
            access_key=os.getenv("UPBIT_ACCESS_KEY", "").strip(),
            secret_key=os.getenv("UPBIT_SECRET_KEY", "").strip(),
            max_order_krw=float(os.getenv("UPBIT_MAX_ORDER_KRW", "10000")),
            allow_live=os.getenv("UPBIT_ALLOW_LIVE", "false").strip().lower() == "true",
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


def load_state() -> dict:
    """봇이 지금 코인을 들고 있는지 기억한다 (재시작해도 유지)."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"position": 0, "last_signal_time": None, "history": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


class Trader:
    """전략 신호를 받아 주문까지 연결하는 실행기."""

    def __init__(
        self,
        strategy: Strategy,
        ticker: str = "KRW-BTC",
        interval: str = "day",
        order_krw: float = MIN_ORDER_KRW,
        live: bool = False,
        config: Config | None = None,
    ):
        self.strategy = strategy
        self.ticker = ticker
        self.interval = interval
        self.config = config or Config.from_env()
        self.order_krw = self._validate_order_size(order_krw)
        self.live = self._validate_live_mode(live)
        self._upbit = None

        if self.live:
            self._upbit = pyupbit.Upbit(self.config.access_key, self.config.secret_key)

    # ---------- 안전장치 ----------

    def _validate_order_size(self, amount: float) -> float:
        if amount < MIN_ORDER_KRW:
            raise SafetyError(
                f"1회 주문금액이 업비트 최소 주문금액({MIN_ORDER_KRW:,}원)보다 작다: {amount:,.0f}원"
            )
        if amount > self.config.max_order_krw:
            raise SafetyError(
                f"1회 주문금액 {amount:,.0f}원이 상한({self.config.max_order_krw:,.0f}원)을 넘는다. "
                ".env의 UPBIT_MAX_ORDER_KRW를 확인할 것 — 오타로 0 하나 더 붙었을 수 있다."
            )
        return float(amount)

    def _validate_live_mode(self, live: bool) -> bool:
        if not live:
            return False
        if not self.config.allow_live:
            raise SafetyError(
                "실주문이 잠겨 있다. .env에서 UPBIT_ALLOW_LIVE=true 로 바꿔야 열린다.\n"
                "(그 전에 백테스트를 충분히 돌렸는지, 잃어도 되는 돈인지 다시 생각할 것.)"
            )
        if not self.config.has_keys:
            raise SafetyError(
                "API 키가 없다. .env.example을 .env로 복사하고 "
                "UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY를 채울 것."
            )
        return True

    # ---------- 조회 ----------

    def current_signal(self, count: int = 200) -> tuple[int, float, datetime]:
        """최근 확정 캔들 기준 목표 포지션과 현재가를 반환.

        마지막 캔들은 아직 진행 중(미확정)이라 신호로 쓰지 않는다.
        진행 중인 봉으로 판단하면 봉이 끝날 때까지 신호가 계속 바뀐다.
        """
        df = get_ohlcv(self.ticker, self.interval, count, use_cache=False)
        closed = df.iloc[:-1]
        positions = self.strategy.generate_positions(closed)
        price = pyupbit.get_current_price(self.ticker)
        if price is None:
            raise RuntimeError(f"{self.ticker} 현재가 조회 실패")
        return int(positions.iloc[-1]), float(price), closed.index[-1].to_pydatetime()

    def balances(self) -> dict:
        if not self.live or self._upbit is None:
            return {"mode": "dry-run", "krw": None, "coin": None}
        coin = self.ticker.split("-")[1]
        return {
            "mode": "live",
            "krw": self._upbit.get_balance("KRW"),
            "coin": self._upbit.get_balance(coin),
        }

    # ---------- 주문 ----------

    def buy(self, price: float) -> dict:
        if not self.live:
            log.info(
                "[모의] 시장가 매수 %s %s원 (현재가 %s)",
                self.ticker, f"{self.order_krw:,.0f}", f"{price:,.0f}",
            )
            return {"simulated": True, "side": "buy", "krw": self.order_krw, "price": price}

        krw = self._upbit.get_balance("KRW")
        if krw < self.order_krw:
            raise SafetyError(f"원화 잔고 부족: {krw:,.0f}원 < 주문 {self.order_krw:,.0f}원")

        log.warning("[실주문] 시장가 매수 %s %s원", self.ticker, f"{self.order_krw:,.0f}")
        return self._upbit.buy_market_order(self.ticker, self.order_krw)

    def sell(self, price: float) -> dict:
        if not self.live:
            log.info("[모의] 시장가 매도 %s (현재가 %s)", self.ticker, f"{price:,.0f}")
            return {"simulated": True, "side": "sell", "price": price}

        coin = self.ticker.split("-")[1]
        volume = self._upbit.get_balance(coin)
        if not volume or volume * price < MIN_ORDER_KRW:
            raise SafetyError(
                f"매도할 수량이 없거나 최소 주문금액 미만이다 "
                f"(평가액 {(volume or 0) * price:,.0f}원)"
            )

        log.warning("[실주문] 시장가 매도 %s %.8f", self.ticker, volume)
        return self._upbit.sell_market_order(self.ticker, volume)

    # ---------- 1회 실행 ----------

    def step(self) -> dict:
        """신호를 확인하고 포지션이 바뀌어야 하면 주문한다. 1회분."""
        state = load_state()
        held = int(state.get("position", 0))
        target, price, candle_time = self.current_signal()

        mode = "실전" if self.live else "모의"
        log.info(
            "[%s] %s 캔들 %s | 현재가 %s원 | 보유:%d → 목표:%d",
            mode, self.ticker, candle_time.strftime("%Y-%m-%d %H:%M"),
            f"{price:,.0f}", held, target,
        )

        action, order = "hold", None
        if target == 1 and held == 0:
            action, order = "buy", self.buy(price)
        elif target == 0 and held == 1:
            action, order = "sell", self.sell(price)
        else:
            log.info("변화 없음 — 아무것도 안 한다.")

        if action != "hold":
            state["position"] = 1 if action == "buy" else 0
            state["last_signal_time"] = candle_time.isoformat()
            state.setdefault("history", []).append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "action": action,
                    "price": price,
                    "live": self.live,
                }
            )
            save_state(state)

        return {
            "action": action,
            "position": state["position"],
            "price": price,
            "candle_time": candle_time,
            "live": self.live,
            "order": order,
        }
