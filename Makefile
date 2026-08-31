# 자주 쓰는 명령 모음. `make` 만 치면 목록이 나온다.
PY := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help
.PHONY: help setup test fetch indicators backtest optimize chart bot docs clean

help:  ## 이 목록 보기
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## 가상환경 만들고 의존성 설치
	python3 -m venv .venv
	$(PIP) install -q -r requirements.txt
	@echo "완료. 다음: make test"

test:  ## 테스트 전체 실행
	$(PY) -m pytest -q

fetch:  ## 1단계 — 데이터 가져와보기
	$(PY) scripts/01_fetch.py

indicators:  ## 2단계 — 지표 계산해서 보기
	$(PY) scripts/02_indicators.py

backtest:  ## 4단계 — 전략 백테스팅
	$(PY) scripts/03_backtest.py

optimize:  ## 과최적화 검증 (훈련/검증 분리 + 워크포워드)
	$(PY) scripts/04_optimize.py

chart:  ## 결과 그래프 생성 (reports/)
	$(PY) scripts/06_chart.py

docs-images:  ## README용 이미지 갱신 (docs/images/)
	$(PY) scripts/06_chart.py --count 800 --out docs/images
	$(PY) scripts/06_chart.py --count 800 --stop-atr 2.0 --trailing --out docs/images

docs:  ## 실험 기록 재생성 (docs/실험기록.md + 그래프)
	$(PY) scripts/07_experiments.py

bot:  ## 5단계 — 봇 실행 (모의, 돈 안 나감)
	$(PY) scripts/05_live.py

clean:  ## 캐시/결과물 삭제
	rm -rf reports data/cache .pytest_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
