#!/usr/bin/env python
"""5단계: 소액 실전 봇 — 기본은 모의(페이퍼 트레이딩).

모의 모드는 주문을 "흉내"만 내는 게 아니다. **실제 시세**로 움직이는 가상 계좌를
상대로 **실전과 완전히 같은 주문 경로**를 탄다. 그래서 모의에서 검증한 게
실전에서도 유효하다.

    python scripts/05_live.py                          # 모의 1회
    python scripts/05_live.py --loop --every 3600       # 모의 반복
    python scripts/05_live.py --stop-atr 2.0 --trailing # 손절 켜고 모의
    python scripts/05_live.py --status                  # 현재 상태만 보기
    python scripts/05_live.py --reset-paper             # 가상 잔고 초기화

    python scripts/05_live.py --live --order-krw 6000   # 실주문

⚠️ 실주문 전 체크리스트
   □ docs/실전-준비-점검.md 의 🔴치명 항목이 전부 해결됐다
   □ 03_backtest.py / 04_optimize.py 의 **검증 구간** 성적을 봤다
   □ API 키에 출금 권한이 없고 IP 화이트리스트가 걸려 있다
   □ 넣는 돈은 전부 잃어도 생활에 지장 없다
   □ 적금·투자 자금과 완전히 분리된 돈이다
"""
import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from upbit.data import VALID_INTERVALS
from upbit.exchange import AuthError, ExchangeError, RateLimited
from upbit.fake_exchange import FakeExchange
from upbit.live import (
    MIN_SAFE_ORDER_KRW,
    Config,
    SafetyError,
    Trader,
    build_exchange,
    default_state,
    load_state,
    save_state,
    setup_logging,
)
from upbit.risk import RiskRules
from upbit.strategies import BollingerStrategy, MACrossStrategy, RSIStrategy

STRATEGIES = {
    "ma": lambda a: MACrossStrategy(a.fast, a.slow),
    "ema": lambda a: MACrossStrategy(a.fast, a.slow, use_ema=True),
    "rsi": lambda a: RSIStrategy(),
    "bb": lambda a: BollingerStrategy(),
}

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n중단 요청 — 이번 주기 끝나고 종료한다.")


def _confirm() -> bool:
    """실주문 최종 확인. 대화형 입력이 불가능하면(파이프/크론) 무조건 취소한다."""
    print("\n⚠️  실제 돈이 나갑니다. 잃어도 되는 소액인지 확인하세요.")
    try:
        return input("계속하려면 '실행' 을 입력: ").strip() == "실행"
    except (EOFError, KeyboardInterrupt):
        print("\n확인 입력을 받을 수 없다 — 안전하게 취소한다.")
        return False


def build_args():
    ap = argparse.ArgumentParser(description="업비트 자동매매 봇 (기본: 모의)")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--strategy", default="ma", choices=list(STRATEGIES))
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=20)
    ap.add_argument("--order-krw", type=float, default=MIN_SAFE_ORDER_KRW)
    ap.add_argument("--stop-atr", type=float, default=None, help="ATR 배수 손절")
    ap.add_argument("--stop-pct", type=float, default=None, help="고정 %% 손절")
    ap.add_argument("--take-profit", type=float, default=None, help="고정 %% 익절")
    ap.add_argument("--trailing", action="store_true", help="추적 손절")
    ap.add_argument("--live", action="store_true", help="실제 주문 (안전장치 통과해야 동작)")
    ap.add_argument("--loop", action="store_true", help="반복 실행")
    ap.add_argument("--every", type=int, default=3600, help="반복 주기(초)")
    ap.add_argument("--status", action="store_true", help="현재 상태만 출력하고 종료")
    ap.add_argument("--reset-paper", action="store_true", help="모의 가상 잔고 초기화")
    return ap.parse_args()


def main() -> None:
    args = build_args()
    setup_logging()
    config = Config.from_env()

    if args.reset_paper:
        state = load_state()
        state["paper"] = None
        save_state(state)
        print(f"모의 가상 잔고를 초기화했다 ({config.paper_krw:,.0f}원)")
        return

    state = load_state()
    strategy = STRATEGIES[args.strategy](args)
    risk = RiskRules(
        stop_loss_pct=args.stop_pct, atr_multiple=args.stop_atr,
        take_profit_pct=args.take_profit, trailing=args.trailing,
    )

    try:
        exchange = build_exchange(config, args.live, args.ticker, state)
    except SafetyError as e:
        print(f"\n❌ 안전장치에 걸렸다 (주문은 나가지 않았다):\n   {e}")
        sys.exit(1)

    try:
        trader = Trader(
            strategy=strategy, exchange=exchange, ticker=args.ticker,
            interval=args.interval, order_krw=args.order_krw, risk=risk,
            live=args.live, config=config,
        )
    except SafetyError as e:
        print(f"\n❌ 안전장치에 걸렸다 (주문은 나가지 않았다):\n   {e}")
        sys.exit(1)

    price = trader.current_price()
    held = trader.current_position(price)
    print("=" * 62)
    print(f"  모드     : {'🔴 실주문 (진짜 돈)' if args.live else '🟢 모의 (가상 잔고 · 실제 시세)'}")
    print(f"  종목/봉  : {args.ticker} / {args.interval}")
    print(f"  전략     : {strategy.name}")
    print(f"  손절     : {risk.describe()}")
    print(f"  주문금액 : {args.order_krw:,.0f}원 (상한 {config.max_order_krw:,.0f}원)")
    print(f"  현재가   : {price:,.0f}원")
    print(f"  보유상태 : {'보유 중' if held else '현금'}", end="")
    if isinstance(exchange, FakeExchange):
        print(f"  |  가상잔고 {exchange.krw:,.0f}원 + {exchange.coin:.8f}개"
              f" = {exchange.equity:,.0f}원")
    else:
        print()
    if state.get("stop_price"):
        print(f"  손절선   : {state['stop_price']:,.0f}원")
    print("=" * 62)

    if args.status:
        return

    if args.live and not _confirm():
        print("취소했다. 잘한 선택일 수도 있다.")
        return

    signal.signal(signal.SIGINT, _handle_sigint)

    while True:
        try:
            result = trader.step()
            if result["action"] != "hold":
                tag = "실주문" if result["live"] else "모의"
                print(f"  → {result['action'].upper()} ({result['reason']}, {tag})"
                      f"  {result['order'].describe()}")
        except AuthError as e:
            # 인증 오류는 계속 시도하면 계정이 차단될 수 있다 — 즉시 멈춘다
            print(f"\n❌ 인증 오류로 중단한다: {e}")
            print("   API 키 만료 / IP 미등록 / 권한 부족을 확인할 것.")
            sys.exit(1)
        except RateLimited as e:
            print(f"  ⏳ 요청 한도 초과 — 이번 주기는 건너뛴다: {e}")
        except SafetyError as e:
            print(f"  ⚠️  안전장치: {e}")
        except ExchangeError as e:
            print(f"  ⚠️  거래소 오류(계속 진행): {e}")
        except Exception as e:  # 봇이 죽으면 포지션이 방치된다
            print(f"  ⚠️  예상 못한 오류(계속 진행): {type(e).__name__}: {e}")

        if not args.loop or _stop:
            break
        for _ in range(args.every):
            if _stop:
                break
            time.sleep(1)
        if _stop:
            break

    print("종료.")


if __name__ == "__main__":
    main()
