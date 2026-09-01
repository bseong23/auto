"""파라미터 탐색과 과최적화(overfitting) 검증.

백테스트에서 제일 흔한 자기기만:
    "5/20보다 7/23이 낫네? 그럼 7/23으로 하자"
→ 과거 데이터에 맞춰 숫자를 고른 것뿐이고, 미래엔 그냥 안 먹힌다.

그래서 여기선 두 가지를 강제한다:
1. **훈련/검증 분리** — 앞 구간에서 고른 파라미터를 뒷 구간에서 검증.
2. **워크포워드** — 구간을 밀어가며 여러 번 반복. 한 번의 운을 걸러낸다.

검증 구간 성적이 훈련 구간보다 훨씬 나쁘면 그건 과최적화다.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterable

import pandas as pd

from .backtest import UPBIT_FEE, run_backtest
from .strategies.base import Strategy

StrategyFactory = Callable[..., Strategy]


def grid_search(
    df: pd.DataFrame,
    factory: StrategyFactory,
    param_grid: dict[str, Iterable],
    metric: str = "total_return",
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
    min_trades: int = 3,
) -> pd.DataFrame:
    """파라미터 조합을 전부 돌려보고 성적표를 반환한다 (metric 내림차순).

    min_trades: 거래 수가 이보다 적으면 통계적으로 무의미하므로 걸러낸다.
                (2번 거래해서 둘 다 이겼다고 '승률 100% 전략'이 아니다.)
    """
    keys = list(param_grid)
    rows = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        try:
            strategy = factory(**params)
        except ValueError:
            continue  # fast >= slow 같은 무의미한 조합
        res = run_backtest(df, strategy, fee=fee, slippage=slippage)
        rows.append(
            {
                **params,
                "총수익률": res.total_return,
                "CAGR": res.cagr,
                "MDD": res.mdd,
                "샤프": res.sharpe,
                "승률": res.win_rate,
                "거래수": res.num_trades,
                "_metric": getattr(res, metric),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("유효한 파라미터 조합이 하나도 없다.")

    enough = table[table["거래수"] >= min_trades]
    table = enough if not enough.empty else table
    return table.sort_values("_metric", ascending=False).reset_index(drop=True)


def split(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """앞쪽 train_ratio는 훈련용, 나머지는 검증용. 시계열이라 섞으면 안 된다."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio는 0과 1 사이여야 한다.")
    cut = int(len(df) * train_ratio)
    if cut < 2 or len(df) - cut < 2:
        raise ValueError("데이터가 너무 짧아서 훈련/검증으로 나눌 수 없다.")
    return df.iloc[:cut], df.iloc[cut:]


def holdout_test(
    df: pd.DataFrame,
    factory: StrategyFactory,
    param_grid: dict[str, Iterable],
    train_ratio: float = 0.7,
    metric: str = "total_return",
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
) -> dict:
    """훈련 구간에서 최적 파라미터를 고르고, 검증 구간에서 정직하게 채점한다."""
    train, test = split(df, train_ratio)

    ranked = grid_search(train, factory, param_grid, metric, fee, slippage)
    best_params = {k: ranked.iloc[0][k] for k in param_grid}
    best_params = {
        k: int(v) if isinstance(v, float) and float(v).is_integer() else v
        for k, v in best_params.items()
    }

    strategy = factory(**best_params)
    train_res = run_backtest(train, strategy, fee=fee, slippage=slippage)
    test_res = run_backtest(test, strategy, fee=fee, slippage=slippage)

    return {
        "best_params": best_params,
        "strategy": strategy,
        "train": train_res,
        "test": test_res,
        "ranking": ranked,
        "degradation": train_res.total_return - test_res.total_return,
    }


def walk_forward(
    df: pd.DataFrame,
    factory: StrategyFactory,
    param_grid: dict[str, Iterable],
    n_splits: int = 4,
    train_ratio: float = 0.6,
    metric: str = "total_return",
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """구간을 밀어가며 '훈련→검증'을 반복한다.

    각 구간마다 파라미터를 새로 고르므로, 실전에서 주기적으로 재최적화하는
    상황을 흉내낸다. 검증 성적이 구간마다 들쭉날쭉하면 그 전략은 못 믿는다.
    """
    if n_splits < 1:
        raise ValueError("n_splits는 1 이상이어야 한다.")

    window = len(df) // n_splits
    if window < 40:
        raise ValueError(
            f"구간당 캔들이 {window}개뿐이라 의미가 없다. count를 늘리거나 n_splits를 줄일 것."
        )

    rows = []
    for i in range(n_splits):
        chunk = df.iloc[i * window : (i + 1) * window]
        try:
            outcome = holdout_test(chunk, factory, param_grid, train_ratio, metric, fee, slippage)
        except ValueError:
            continue
        test_res = outcome["test"]
        rows.append(
            {
                "구간": f"#{i + 1}",
                "훈련기간": f"{chunk.index[0]:%y-%m-%d}~",
                "검증기간": f"{test_res.equity.index[0]:%y-%m-%d}~{test_res.equity.index[-1]:%y-%m-%d}",
                "선택된 파라미터": outcome["best_params"],
                "훈련수익률": outcome["train"].total_return,
                "검증수익률": test_res.total_return,
                "검증MDD": test_res.mdd,
                "검증거래수": test_res.num_trades,
            }
        )
    return pd.DataFrame(rows)


def windowed_returns(
    df: pd.DataFrame,
    strategy: Strategy,
    n_splits: int = 6,
    warmup: int = 250,
    fee: float = UPBIT_FEE,
    slippage: float = 0.0005,
    initial_capital: float = 1_000_000,
) -> list[float]:
    """연속된 검증창 n_splits 개에서 각 창 동안 번 수익률만 모은다.

    파라미터를 재최적화하지 않는다 — '필터를 붙였나', '봉을 바꿨나' 같은 **구조적 선택**을
    여러 시장 국면에서 비교하는 용도다. 한 창에서만 좋은 건 운이다.

    **워밍업이 핵심이다.** 200일선을 쓰는 전략을 64일짜리 창에 넣으면 지표가 전부 NaN 이라
    아무것도 안 산다. 그건 전략이 나쁜 게 아니라 평가가 틀린 것이다. 각 창 앞에 `warmup`
    개의 봉을 붙여 지표를 데운 뒤, 창 안에서 늘어난 자산만 센다. 워밍업 중 잡은 포지션을
    들고 창에 진입할 수 있는데 그건 실전과 같으므로 그대로 둔다.
    """
    usable = len(df) - warmup
    if usable < n_splits * 30:
        return []
    window = usable // n_splits
    out: list[float] = []
    for i in range(n_splits):
        start = warmup + i * window
        frame = df.iloc[start - warmup : start + window]
        if len(frame) < warmup + 20:
            continue
        equity = run_backtest(frame, strategy, initial_capital, fee, slippage).equity
        base = equity.iloc[warmup]
        if base > 0:
            out.append(float(equity.iloc[-1] / base - 1))
    return out


def summarize_windows(returns: list[float]) -> dict:
    """평균만 보면 한 번의 대박에 속는다. 흑자 비율과 최악의 창을 같이 본다."""
    if not returns:
        return {"평균": float("nan"), "흑자": "0/0", "최악": float("nan"), "n": 0}
    wins = sum(1 for r in returns if r > 0)
    return {"평균": sum(returns) / len(returns), "흑자": f"{wins}/{len(returns)}",
            "최악": min(returns), "n": len(returns)}
