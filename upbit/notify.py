"""알림 — 봇이 뭘 했는지, 죽었는지 알아야 한다.

로그 파일만 남기면 사람이 안 본다. 주문이 나가거나 손절이 터지거나 봇이 멈추면
바로 알아야 한다.

웹훅 하나로 처리한다 (Slack / Discord 둘 다 `{"text": ...}` 또는
`{"content": ...}` 형태를 받는다). 설정이 없으면 조용히 아무것도 안 한다 —
알림 설정을 안 했다고 봇이 안 돌면 안 되니까.

**알림 실패는 절대 봇을 멈추지 않는다.** 알림은 부수적인 것이고,
그것 때문에 포지션이 방치되면 본말전도다.
"""
from __future__ import annotations

import json
import logging
import os
from urllib import error, request

log = logging.getLogger("upbit.notify")

TIMEOUT_SEC = 5


class Notifier:
    """웹훅 알림. URL이 없으면 로그만 남기고 넘어간다."""

    def __init__(self, webhook_url: str = "", enabled: bool = True):
        self.webhook_url = webhook_url.strip()
        self.enabled = enabled and bool(self.webhook_url)

    @classmethod
    def from_env(cls) -> "Notifier":
        return cls(webhook_url=os.getenv("UPBIT_WEBHOOK_URL", ""))

    def send(self, text: str) -> bool:
        """알림 발송. 성공하면 True. **실패해도 예외를 던지지 않는다.**"""
        if not self.enabled:
            return False

        try:
            # Request 생성도 try 안에 둔다 — 잘못된 URL은 여기서 ValueError 를 던진다.
            # .env 에 오타 하나 났다고 봇이 죽으면 안 된다.
            payload = json.dumps({"text": text, "content": text}).encode()
            req = request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=TIMEOUT_SEC) as response:
                return 200 <= response.status < 300
        except (error.URLError, error.HTTPError, OSError, ValueError) as exc:
            # 알림이 안 갔다고 봇이 멈추면 본말전도다
            log.warning("알림 발송 실패(무시하고 계속): %s", exc)
            return False

    # ---- 상황별 메시지 ----

    def order_filled(self, ticker: str, action: str, reason: str, order, live: bool) -> bool:
        mark = "🔴 실주문" if live else "🟢 모의"
        verb = "매수" if action == "buy" else "매도"
        return self.send(f"{mark} {ticker} {verb} ({reason})\n{order.describe()}")

    def stop_hit(self, ticker: str, price: float, stop: float) -> bool:
        return self.send(
            f"⚠️ {ticker} 손절 발동 — 현재가 {price:,.0f}원 ≤ 손절선 {stop:,.0f}원"
        )

    def bot_stopped(self, why: str) -> bool:
        return self.send(f"🛑 봇이 멈췄다: {why}")

    def heartbeat(self, ticker: str, price: float, position: int, equity: float | None) -> bool:
        state = "보유 중" if position else "현금"
        tail = f" | 평가액 {equity:,.0f}원" if equity is not None else ""
        return self.send(f"💓 {ticker} {price:,.0f}원 · {state}{tail}")
