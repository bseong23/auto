#!/usr/bin/env python
"""5단계: 소액 실전 봇 — 기본은 모의(dry-run).

    # 모의: 신호만 보고 주문은 흉내만 (API 키 없어도 됨)
    python scripts/05_live.py

    # 모의 루프: 1시간마다 확인
    python scripts/05_live.py --loop --every 3600

    # 실주문: .env에 키 + UPBIT_ALLOW_LIVE=true 여야만 열린다
    python scripts/05_live.py --live --order-krw 6000

⚠️ 실주문 전 체크리스트
   □ 03_backtest.py 로 이 전략을 검증했다
   □ 04_optimize.py 의 '검증 구간' 성적을 봤다 (훈련 성적 말고)
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
from upbit.live import MIN_ORDER_KRW, Config, SafetyError, Trader, load_state, setup_logging
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


def main() -> None:
    ap = argparse.ArgumentParser(description="업비트 자동매매 봇 (기본: 모의)")
    ap.add_argument("--ticker", default="KRW-BTC")
    ap.add_argument("--interval", default="day", choices=VALID_INTERVALS)
    ap.add_argument("--strategy", default="ma", choices=list(STRATEGIES))
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=20)
    ap.add_argument("--order-krw", type=float, default=MIN_ORDER_KRW)
    ap.add_argument("--live", action="store_true", help="실제 주문 (안전장치 통과해야 동작)")
    ap.add_argument("--loop", action="store_true", help="반복 실행")
    ap.add_argument("--every", type=int, default=3600, help="반복 주기(초)")
    args = ap.parse_args()

    setup_logging()
    config = Config.from_env()
    strategy = STRATEGIES[args.strategy](args)

    print("=" * 60)
    print(f"  모드     : {'🔴 실주문 (진짜 돈)' if args.live else '🟢 모의 (돈 안 나감)'}")
    print(f"  종목/봉  : {args.ticker} / {args.interval}")
    print(f"  전략     : {strategy.name}")
    print(f"  주문금액 : {args.order_krw:,.0f}원 (상한 {config.max_order_krw:,.0f}원)")
    print(f"  현재상태 : {'보유 중' if load_state().get('position') else '현금'}")
    print("=" * 60)

    # 안전장치를 먼저 통과시킨다 — 막힐 거면 확인 입력을 받기 전에 막는다.
    try:
        trader = Trader(
            strategy=strategy,
            ticker=args.ticker,
            interval=args.interval,
            order_krw=args.order_krw,
            live=args.live,
            config=config,
        )
    except SafetyError as e:
        print(f"\n❌ 안전장치에 걸렸다 (주문은 나가지 않았다):\n   {e}")
        sys.exit(1)

    if args.live and not _confirm():
        print("취소했다. 잘한 선택일 수도 있다.")
        return

    signal.signal(signal.SIGINT, _handle_sigint)

    while True:
        try:
            result = trader.step()
            if result["action"] != "hold":
                print(f"  → {result['action'].upper()} 실행 "
                      f"({'실주문' if result['live'] else '모의'})")
        except SafetyError as e:
            print(f"  ⚠️  안전장치: {e}")
        except Exception as e:  # 네트워크 끊김 등으로 봇이 죽으면 안 된다
            print(f"  ⚠️  오류(계속 진행): {type(e).__name__}: {e}")

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
