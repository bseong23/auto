"""업비트 자동매매 학습 프로젝트.

목적은 돈이 아니라 배우고 만들어보는 것. 실전은 잃어도 되는 소액만.

빠른 사용:

    from upbit import get_ohlcv, run_backtest, MACrossStrategy

    df = get_ohlcv("KRW-BTC", "day", 800)
    result = run_backtest(df, MACrossStrategy(5, 20))
    print(result.summary())
"""
from __future__ import annotations

__version__ = "0.1.0"

from .backtest import UPBIT_FEE, BacktestResult, Trade, compare, run_backtest
from .data import describe, get_ohlcv, get_tickers
from .optimize import grid_search, holdout_test, split, walk_forward
from .risk import RiskRules
from .strategies import (
    BollingerStrategy,
    BuyAndHoldStrategy,
    MACrossStrategy,
    RSIStrategy,
    Strategy,
)

__all__ = [
    "__version__",
    # 데이터
    "get_ohlcv",
    "get_tickers",
    "describe",
    # 전략
    "Strategy",
    "MACrossStrategy",
    "RSIStrategy",
    "BollingerStrategy",
    "BuyAndHoldStrategy",
    # 백테스팅
    "run_backtest",
    "compare",
    "BacktestResult",
    "Trade",
    "UPBIT_FEE",
    # 손절
    "RiskRules",
    # 최적화·검증
    "grid_search",
    "holdout_test",
    "walk_forward",
    "split",
]
