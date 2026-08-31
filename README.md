# upbite — 업비트 자동매매 학습 프로젝트

> **목적: 돈 버는 봇이 아니라 공부/포트폴리오용.**
> 실제 돈은 "잃어도 되는 소액"만. 적금·투자 자금은 절대 건드리지 않는다.

핸드오프 문서([업비트-자동매매-핸드오프.md](업비트-자동매매-핸드오프.md))의 5단계를
실제로 돌아가는 코드로 구현한 것.

---

## 빠른 시작

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/01_fetch.py          # 데이터 받아보기 (API 키 X)
.venv/bin/python scripts/02_indicators.py     # 지표 계산해서 눈으로 보기
.venv/bin/python scripts/03_backtest.py       # 전략 백테스팅 ★
.venv/bin/python scripts/04_optimize.py       # 과최적화 검증 ★★
.venv/bin/python scripts/05_live.py           # 봇 (기본은 모의, 돈 안 나감)
```

테스트: `.venv/bin/python -m pytest -q`

---

## 단계별 매핑

| 단계 | 하는 일 | 파일 | API 키 |
|---|---|---|---|
| 1 | 데이터 수집 | `upbite/data.py`, `scripts/01_fetch.py` | 불필요 |
| 2 | 지표 계산 | `upbite/indicators.py`, `scripts/02_indicators.py` | 불필요 |
| 3 | 매수/매도 규칙 | `upbite/strategies/` | 불필요 |
| 4 | **백테스팅** | `upbite/backtest.py`, `scripts/03_backtest.py` | 불필요 |
| 4.5 | **과최적화 검증** | `upbite/optimize.py`, `scripts/04_optimize.py` | 불필요 |
| 5 | 소액 실전 | `upbite/live.py`, `scripts/05_live.py` | **필요** |

---

## 구조

```
upbite/
├── data.py          # OHLCV 수집 + CSV 캐시
├── indicators.py    # SMA/EMA/RSI/볼린저/MACD/ATR
├── strategies/      # 전략 (교체 가능 — Strategy 패턴)
│   ├── base.py          Strategy 인터페이스
│   ├── ma_cross.py      이동평균 교차 (추세추종)
│   ├── rsi.py           RSI 역추세
│   ├── bollinger.py     볼린저밴드 역추세
│   └── buy_and_hold.py  존버 (벤치마크)
├── backtest.py      # 백테스팅 엔진 + 성과지표
├── optimize.py      # 그리드서치 / 홀드아웃 / 워크포워드
└── live.py          # 실전 트레이더 (안전장치 4겹)
```

### 새 전략 추가하기

`Strategy`를 상속해서 `generate_positions`만 구현하면 백테스터·봇에 그대로 꽂힌다.

```python
from upbite.strategies.base import Strategy, _hold_between

class MyStrategy(Strategy):
    name = "내전략"

    def generate_positions(self, df):
        # 1 = 보유, 0 = 현금
        return _hold_between(entries=..., exits=...)
```

`tests/test_strategies.py`의 `STRATEGIES` 리스트에 넣으면
"미래참조 없음 / 포지션 이진값 / 워밍업 구간 현금" 공통 규약을 자동으로 검사한다.

---

## 이 프로젝트가 지키는 원칙

### 1. 미래를 보지 않는다 (lookahead 방지)

신호는 **t봉 종가**로 판단하고, 체결은 **t+1봉 시가**로 한다.
같은 봉 종가에 사고파는 코드는 실제로 불가능한 거래라 수익률이 부풀려진다.

`tests/test_backtest.py`가 이걸 직접 검증한다 — 뒤쪽 가격을 5배로 조작해도
앞쪽 자산곡선이 한 톨도 안 변해야 통과.

### 2. 수수료·슬리피지를 뺀다

업비트 원화마켓 편도 0.05% + 슬리피지 0.05%.
가격이 **전혀 안 움직여도** 매매를 반복하면 잃는다는 걸 테스트로 박아뒀다.

### 3. 훈련 성적을 믿지 않는다

파라미터를 잘게 쪼개 탐색할수록 훈련 성적은 **무조건** 좋아진다.
좋아지는 건 전략이 아니라 '과거를 외운 정도'다.

`scripts/04_optimize.py`는 훈련/검증을 분리하고 워크포워드로 여러 번 재검증한다.
실제로 돌려보면 이런 게 나온다:

```
훈련 구간 수익률 : +178.71%
검증 구간 수익률 :   +3.95%   ← 실제로 믿을 수 있는 숫자
→ 검증 구간 절반 이상에서 잃었다. 실전에 넣을 전략이 아니다.
```

이게 정상이다. 이 숫자를 보라고 만든 도구다.

---

## 5단계 실전 전 — 안전장치

`live.py`는 안전장치를 4겹으로 걸어놨다.

1. **기본이 모의(dry-run)** — 주문을 흉내만 내고 실제로 안 낸다
2. 실주문은 `.env`의 `UPBIT_ALLOW_LIVE=true` **그리고** `--live` 플래그 **둘 다** 필요
3. 1회 주문 금액 상한 (`UPBIT_MAX_ORDER_KRW`) — 0 하나 더 붙은 오타 방어
4. 키는 환경변수에서만 읽음 (`.env`는 `.gitignore`에 등록)

```bash
cp .env.example .env      # 그리고 키 채우기 — 절대 커밋 금지
```

### 실주문 전 체크리스트

- [ ] `03_backtest.py`로 이 전략을 검증했다
- [ ] `04_optimize.py`의 **검증 구간** 성적을 봤다 (훈련 성적 말고)
- [ ] 넣는 돈은 전부 잃어도 생활에 지장 없다
- [ ] 적금·투자 자금과 완전히 분리된 돈이다

---

## 알아둘 것

- **"알아서 돈 버는 봇"은 없다.** 진짜 있으면 만든 사람이 조용히 부자 됐지 안 판다.
- 어떤 전략도 항상 이기지 않는다. 봇은 **자동으로 잃기도** 한다.
- 백테스트 결과는 '과거 그 구간에서 그랬다'는 뜻일 뿐, 미래 보장이 아니다.
- 짧은 봉(1·5분) = 노이즈 많고 수수료 자주 나감. 초보는 `day` / `minute60`.

잃으면 수업료, 잘되면 보너스. **실력·포폴은 무조건 남는다.**
