"""체결 기록 — 백테스트 가정이 실제와 맞는지 검증하기 위한 자료.

백테스트는 슬리피지를 0.05%로 **가정**했다. 그게 맞는지는 실제로 주문해봐야 안다.
실측이 0.2%면 백테스트 수익률은 전부 다시 계산해야 한다.

그래서 주문마다 이걸 남긴다:
- 판단 시점의 가격 (신호를 낼 때 본 값)
- 실제 체결 평균가
- 둘의 차이 = **실측 슬리피지**

CSV로 남기므로 나중에 pandas로 바로 읽어서 분석할 수 있다.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILLS_PATH = ROOT / "reports" / "fills.csv"

FIELDS = [
    "time", "mode", "ticker", "action", "reason", "uuid",
    "decision_price", "avg_price", "slippage_pct",
    "volume", "executed_krw", "fee", "fee_pct",
]


def record_fill(
    ticker: str,
    action: str,
    reason: str,
    decision_price: float,
    order,
    live: bool,
    path: Path | None = None,
) -> dict:
    """체결 하나를 CSV에 덧붙이고, 기록한 내용을 돌려준다.

    slippage_pct: 판단 시점 가격 대비 실제 체결가가 얼마나 불리했나.
      매수는 비싸게 샀으면 +, 매도는 싸게 팔았으면 +. 둘 다 '손해 본 정도'다.
    """
    target = path or FILLS_PATH
    avg_price = order.avg_price

    slippage = None
    if avg_price and decision_price:
        raw = (avg_price - decision_price) / decision_price
        slippage = raw if action == "buy" else -raw

    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "live" if live else "paper",
        "ticker": ticker,
        "action": action,
        "reason": reason,
        "uuid": order.uuid,
        "decision_price": round(decision_price, 4) if decision_price else "",
        "avg_price": round(avg_price, 4) if avg_price else "",
        "slippage_pct": round(slippage * 100, 4) if slippage is not None else "",
        "volume": f"{order.executed_volume:.8f}",
        "executed_krw": round(order.executed_krw, 4),
        "fee": round(order.paid_fee, 4),
        "fee_pct": (
            round(order.paid_fee / order.executed_krw * 100, 4)
            if order.executed_krw else ""
        ),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    is_new = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def summarize(path: Path | None = None) -> dict:
    """지금까지 기록된 체결의 실측 슬리피지·수수료 요약.

    백테스트에 넣은 가정과 비교하라고 만든 것이다.
    """
    target = path or FILLS_PATH
    if not target.exists():
        return {"count": 0}

    with target.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"count": 0}

    def numbers(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key, "")
            if value not in ("", None):
                out.append(float(value))
        return out

    slippages = numbers("slippage_pct")
    fees = numbers("fee_pct")
    live_rows = [r for r in rows if r["mode"] == "live"]

    return {
        "count": len(rows),
        "live_count": len(live_rows),
        "slippage_mean_pct": sum(slippages) / len(slippages) if slippages else None,
        "slippage_max_pct": max(slippages) if slippages else None,
        "fee_mean_pct": sum(fees) / len(fees) if fees else None,
        "total_fee_krw": sum(numbers("fee")),
    }
