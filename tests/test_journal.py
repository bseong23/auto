"""체결 기록 — 백테스트 가정을 검증할 자료가 제대로 남는지."""
import csv

import pytest

from upbit.exchange import OrderResult
from upbit.journal import record_fill, summarize


def order(volume=0.001, krw=100_000, fee=50, uid="u1", side="bid"):
    return OrderResult(uuid=uid, side=side, state="done",
                       executed_volume=volume, executed_krw=krw, paid_fee=fee)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "fills.csv"


def test_writes_header_once(path):
    record_fill("KRW-BTC", "buy", "신호", 100_000_000, order(), False, path)
    record_fill("KRW-BTC", "sell", "신호", 100_000_000, order(uid="u2"), False, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("time,") and len(lines) == 3


def filled_at(price: float, volume: float = 0.001, side: str = "bid"):
    """평균 체결가가 price 가 되도록 만든 주문."""
    return order(volume=volume, krw=price * volume, side=side)


def test_buy_slippage_is_positive_when_filled_higher(path):
    """비싸게 샀으면 손해 — 양수."""
    row = record_fill("KRW-BTC", "buy", "신호", 100_000_000,
                      filled_at(100_100_000), False, path)
    assert row["slippage_pct"] == pytest.approx(0.1)


def test_sell_slippage_is_positive_when_filled_lower(path):
    """싸게 팔았으면 손해 — 역시 양수. 부호를 통일해야 평균이 의미 있다."""
    row = record_fill("KRW-BTC", "sell", "신호", 100_000_000,
                      filled_at(99_900_000, side="ask"), False, path)
    assert row["slippage_pct"] == pytest.approx(0.1)


def test_favourable_fill_is_negative(path):
    """싸게 샀으면 이득 — 음수."""
    row = record_fill("KRW-BTC", "buy", "신호", 100_000_000,
                      filled_at(99_000_000), False, path)
    assert row["slippage_pct"] < 0


def test_fee_percent_is_recorded(path):
    row = record_fill("KRW-BTC", "buy", "신호", 100_000_000,
                      order(krw=100_000, fee=50), False, path)
    assert row["fee_pct"] == pytest.approx(0.05)


def test_mode_distinguishes_paper_from_live(path):
    record_fill("KRW-BTC", "buy", "신호", 1e8, order(uid="a"), False, path)
    record_fill("KRW-BTC", "buy", "신호", 1e8, order(uid="b"), True, path)

    modes = [r["mode"] for r in csv.DictReader(path.open(encoding="utf-8"))]
    assert modes == ["paper", "live"]


def test_summary_of_empty_journal(tmp_path):
    assert summarize(tmp_path / "없음.csv") == {"count": 0}


def test_summary_aggregates(path):
    record_fill("KRW-BTC", "buy", "신호", 100_000_000,
                order(volume=0.001, krw=100_050, fee=50, uid="a"), False, path)
    record_fill("KRW-BTC", "buy", "신호", 100_000_000,
                order(volume=0.001, krw=100_100, fee=60, uid="b"), True, path)

    stats = summarize(path)
    assert stats["count"] == 2 and stats["live_count"] == 1
    assert stats["total_fee_krw"] == pytest.approx(110)
    assert stats["slippage_max_pct"] > stats["slippage_mean_pct"] or stats["count"] == 1


def test_unfilled_order_records_blank_slippage(path):
    empty = OrderResult(uuid="x", side="bid", state="cancel")
    row = record_fill("KRW-BTC", "buy", "신호", 100_000_000, empty, False, path)
    assert row["slippage_pct"] == "" and row["avg_price"] == ""



def test_old_journal_header_is_migrated_in_place(path):
    """예전 열 구성 파일에 새 열(order_type, limit_fill_pct)이 덧붙는다. 기존 행은 빈 값."""
    path.write_text("time,mode,ticker,action,reason,uuid,decision_price,avg_price,slippage_pct,"
                    "volume,executed_krw,fee,fee_pct\n"
                    "2026-08-31T18:05:46,paper,KRW-BTC,buy,신호,fake-buy-1,1e8,1e8,0.0,0.00005,5497,2.75,0.05\n",
                    encoding="utf-8")
    record_fill("KRW-BTC", "buy", "신호", 1e8, order(uid="u2"), False, path,
                order_type="limit", limit_fill_pct=1.0)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["order_type"] == "" and rows[1]["order_type"] == "limit"
    assert rows[1]["limit_fill_pct"] == "100.0"


def test_summary_reports_limit_fill_rate(path):
    record_fill("KRW-BTC", "buy", "신호", 1e8, order(uid="a"), False, path, order_type="limit", limit_fill_pct=1.0)
    record_fill("KRW-BTC", "buy", "신호", 1e8, order(uid="b"), False, path, order_type="limit+market", limit_fill_pct=0.4)
    record_fill("KRW-BTC", "sell", "손절", 1e8, order(uid="c", side="ask"), False, path, order_type="market")
    stats = summarize(path)
    assert stats["limit_attempts"] == 2
    assert stats["limit_fill_mean_pct"] == pytest.approx(70.0)
