"""백테스트 결과 시각화.

숫자표만 보면 "총수익률 +71%"가 좋아 보인다. 하지만 그 안에 -30% 구간이
반년쯤 있었다면 실제로는 못 버티고 중간에 던졌을 것이다.
자산곡선과 낙폭 그래프는 그 '견딜 수 있는가'를 눈으로 보여준다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 저장되도록
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult
from .indicators import sma

#: 전략별 고정 색 — 여러 그래프에서 같은 전략이 같은 색으로 보이게
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
BENCHMARK_COLOR = "#94a3b8"


def use_korean_font() -> str | None:
    """한글이 네모(□)로 깨지지 않게 폰트를 잡는다. 없으면 조용히 넘어간다."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False  # 폰트에 −(U+2212)가 없어서 깨짐
            return candidate
    return None


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def _format_dates(ax, index: pd.DatetimeIndex) -> None:
    span_days = (index[-1] - index[0]).days
    fmt = "%y-%m" if span_days > 120 else "%m-%d"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def plot_result(
    result: BacktestResult,
    df: pd.DataFrame,
    path: str | Path,
    fast: int = 5,
    slow: int = 20,
    title: str | None = None,
) -> Path:
    """단일 전략 결과를 3단 그래프로 저장한다.

    위: 가격 + 이동평균 + 매수/매도 시점
    중: 자산곡선 (전략 vs 사서 존버)
    아래: 낙폭(고점 대비 하락률)
    """
    use_korean_font()
    fig, (ax_price, ax_equity, ax_dd) = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.4], "hspace": 0.12},
    )

    index = df.index
    close = df["close"]

    # ── 위: 가격 + 이동평균 + 매매 시점 ──
    ax_price.plot(index, close, color="#0f172a", linewidth=1.1, label="종가")
    ax_price.plot(index, sma(close, fast), color="#2563eb", linewidth=0.9, label=f"MA{fast}")
    ax_price.plot(index, sma(close, slow), color="#f59e0b", linewidth=0.9, label=f"MA{slow}")

    for trade in result.trades:
        ax_price.scatter(trade.entry_time, trade.entry_price, marker="^",
                         s=52, color="#059669", zorder=5, edgecolors="white", linewidths=0.6)
        if not trade.is_open:
            stopped = trade.exit_reason.startswith("손절")
            ax_price.scatter(trade.exit_time, trade.exit_price, marker="v", s=52,
                             color="#dc2626" if stopped else "#64748b",
                             zorder=5, edgecolors="white", linewidths=0.6)

    # 보유 구간 음영 — 언제 시장에 들어가 있었는지
    ax_price.fill_between(index, *ax_price.get_ylim(), where=result.positions.astype(bool),
                          color="#2563eb", alpha=0.06, step="post")

    ax_price.set_ylabel("가격 (원)")
    ax_price.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_price.set_title(title or f"{result.strategy_name}  —  {result.risk.describe()}",
                       fontsize=13, fontweight="bold", pad=12)
    ax_price.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:,.0f}M" if v >= 1e6 else f"{v:,.0f}")
    _style(ax_price)

    # ── 중: 자산곡선 ──
    ax_equity.plot(index, result.equity, color="#2563eb", linewidth=1.5,
                   label=f"{result.strategy_name}  {result.total_return:+.1%}")
    if result.benchmark is not None:
        ax_equity.plot(index, result.benchmark, color=BENCHMARK_COLOR, linewidth=1.2,
                       linestyle="--", label=f"사서 존버  {result.benchmark_return:+.1%}")
    ax_equity.axhline(result.initial_capital, color="#cbd5e1", linewidth=0.8, zorder=0)
    ax_equity.set_ylabel("자산 (원)")
    ax_equity.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_equity.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:,.1f}M" if v >= 1e6 else f"{v:,.0f}")
    _style(ax_equity)

    # ── 아래: 낙폭 ──
    drawdown = result.equity / result.equity.cummax() - 1
    ax_dd.fill_between(index, drawdown, 0, color="#dc2626", alpha=0.28)
    ax_dd.plot(index, drawdown, color="#dc2626", linewidth=0.8)
    ax_dd.set_ylabel("낙폭")
    ax_dd.set_xlabel("")
    ax_dd.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_dd.annotate(f"최대낙폭 {result.mdd:.1%}", xy=(0.01, 0.08),
                   xycoords="axes fraction", fontsize=9, color="#dc2626", fontweight="bold")
    _style(ax_dd)
    _format_dates(ax_dd, index)

    return _save(fig, path)


def plot_comparison(
    results: list[BacktestResult], path: str | Path, title: str = "전략 비교"
) -> Path:
    """여러 전략의 자산곡선을 한 장에 겹쳐 그린다."""
    use_korean_font()
    fig, (ax_equity, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12},
    )

    ranked = sorted(results, key=lambda r: r.total_return, reverse=True)
    for i, result in enumerate(ranked):
        is_benchmark = result.strategy_name == "사서 존버"
        ax_equity.plot(
            result.equity.index, result.equity,
            color=BENCHMARK_COLOR if is_benchmark else PALETTE[i % len(PALETTE)],
            linewidth=1.1 if is_benchmark else 1.5,
            linestyle="--" if is_benchmark else "-",
            label=f"{result.strategy_name}  {result.total_return:+.1%}  (MDD {result.mdd:.1%})",
        )
        drawdown = result.equity / result.equity.cummax() - 1
        ax_dd.plot(result.equity.index, drawdown,
                   color=BENCHMARK_COLOR if is_benchmark else PALETTE[i % len(PALETTE)],
                   linewidth=1.0, linestyle="--" if is_benchmark else "-")

    ax_equity.axhline(ranked[0].initial_capital, color="#cbd5e1", linewidth=0.8, zorder=0)
    ax_equity.set_ylabel("자산 (원)")
    ax_equity.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax_equity.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_equity.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:,.1f}M" if v >= 1e6 else f"{v:,.0f}")
    _style(ax_equity)

    ax_dd.set_ylabel("낙폭")
    ax_dd.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax_dd)
    _format_dates(ax_dd, ranked[0].equity.index)

    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
