"""
metrics.py
----------
Standard risk-adjusted performance metrics, computed the same way for every
strategy/benchmark so comparisons are apples-to-apples.

All ratios are annualized assuming 252 trading days/year (standard convention
carried over from the Week 2/6 material). `rf` (risk-free rate) is annualized
and defaults to 0 - Sharpe/Sortino are computed on raw daily returns, which is
the more common convention for short-horizon backtests where the risk-free
drag is small relative to strategy noise; pass a nonzero `rf` to use excess
returns instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(daily_returns: pd.Series) -> float:
    total_growth = (1 + daily_returns).prod()
    n = len(daily_returns)
    if n == 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (TRADING_DAYS / n) - 1


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf / TRADING_DAYS
    denom = excess.std()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (excess.mean() / denom) * np.sqrt(TRADING_DAYS)


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    denom = downside.std()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return (excess.mean() / denom) * np.sqrt(TRADING_DAYS)


def equity_curve(daily_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1 + daily_returns.fillna(0)).cumprod()


def max_drawdown(daily_returns: pd.Series) -> float:
    curve = equity_curve(daily_returns)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return drawdown.min()


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    curve = equity_curve(daily_returns)
    running_max = curve.cummax()
    return curve / running_max - 1


def calmar_ratio(daily_returns: pd.Series) -> float:
    mdd = max_drawdown(daily_returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(daily_returns) / abs(mdd)


def turnover(weights_history: pd.DataFrame) -> float:
    """
    Average one-way turnover per rebalance: mean of sum(|w_t - w_{t-1}|)
    across all rebalance events. A turnover of 0.40 means, on average, 40%
    of the portfolio's notional was traded at each rebalance.
    """
    diffs = weights_history.diff().abs().sum(axis=1).dropna()
    if len(diffs) == 0:
        return 0.0
    return diffs.mean()


def summarize(daily_returns: pd.Series, weights_history: pd.DataFrame | None = None,
              label: str = "") -> dict:
    """One-row performance summary dict, ready to be assembled into a
    comparison table across strategies."""
    out = {
        "Strategy": label,
        "Ann. Return": annualized_return(daily_returns),
        "Ann. Vol": annualized_vol(daily_returns),
        "Sharpe": sharpe_ratio(daily_returns),
        "Sortino": sortino_ratio(daily_returns),
        "Max Drawdown": max_drawdown(daily_returns),
        "Calmar": calmar_ratio(daily_returns),
    }
    if weights_history is not None:
        out["Avg Turnover/Rebalance"] = turnover(weights_history)
    return out
