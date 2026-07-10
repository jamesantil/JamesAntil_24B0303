"""
optimizer.py
------------
Convex portfolio optimizers, solved with `cvxpy`, one objective per detected
regime. Everything here is long-only and fully invested (weights sum to 1,
each weight in [0, upper_bound]) — no leverage, no shorting, matching a
retail/real-fund-style constraint set.

Regime -> objective mapping (the "why" is in the README's "Key Decisions"
section; short version here):

    Bull    -> Maximize Sharpe ratio (tangency portfolio).
               In a calmly rising market you want full participation,
               risk-adjusted.
    Bear    -> Minimize variance, with a capped equity weight.
               Capital preservation matters more than upside; a hard cap
               stops the optimizer from "min-variance-ing" its way into
               100% equity just because a Bear regime's ESTIMATED equity
               vol happens to look temporarily low on a short trailing
               window.
    Crisis  -> Minimize variance, with a tighter equity cap and a gold floor.
               Flight-to-quality: force meaningful ballast into the
               historically negatively-correlated safe-haven asset rather
               than trusting a covariance estimate computed on a handful of
               chaotic days.

Estimation window
------------------
`mu` and `Sigma` passed into every function here MUST already be computed
from a strictly historical (causal) return window by the caller
(backtest.py) — this module does no lookback itself, it only solves the
optimization given whatever moments it's handed.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

ASSET_ORDER = ["EQUITY", "BONDS", "GOLD"]


def _fallback_equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def max_sharpe_weights(mu: np.ndarray, Sigma: np.ndarray,
                        upper_bound: float = 1.0) -> np.ndarray:
    """
    Tangency (max-Sharpe) portfolio via the standard convex reformulation.

    Maximizing mu'w / sqrt(w'Sigma w) subject to sum(w)=1, w>=0 is NOT
    directly convex, but for long-only, no-risk-free-asset portfolios the
    following substitution is: let y = w / k for some k > 0 chosen so that
    mu'y = 1. Then minimizing y'Sigma y subject to mu'y = 1, y >= 0 and
    finally normalizing w = y / sum(y) recovers the max-Sharpe weights
    exactly (this works whenever at least one asset has positive expected
    excess return, which we enforce by falling back to min-variance if not).
    """
    n = len(mu)
    if np.all(mu <= 0):
        # No asset has positive expected return this window - max-Sharpe is
        # degenerate (nothing to "buy" for return). Fall back to min-variance.
        return min_variance_weights(mu, Sigma, upper_bound=upper_bound)

    y = cp.Variable(n)
    Sigma_psd = _nearest_psd(Sigma)
    objective = cp.Minimize(cp.quad_form(y, Sigma_psd))
    constraints = [mu @ y == 1, y >= 0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if y.value is None or not np.isfinite(y.value).all() or y.value.sum() <= 1e-9:
            raise ValueError("degenerate solve")
        w = np.asarray(y.value).flatten()
        w = np.clip(w, 0, None)
        w = w / w.sum()
    except Exception:
        return min_variance_weights(mu, Sigma, upper_bound=upper_bound)

    if upper_bound < 1.0 and (w > upper_bound + 1e-6).any():
        w = _project_with_cap(w, upper_bound)
    return w


def min_variance_weights(mu: np.ndarray, Sigma: np.ndarray,
                          upper_bound: float = 1.0,
                          asset_caps: dict[str, float] | None = None,
                          asset_floors: dict[str, float] | None = None,
                          asset_names: list[str] | None = None) -> np.ndarray:
    """
    Minimum-variance long-only portfolio, with optional PER-ASSET cap/floor
    constraints (used to encode "defensive tilt" in Bear/Crisis regimes,
    e.g. cap EQUITY at 15%, floor GOLD at 30% in Crisis).
    """
    n = len(mu)
    asset_names = asset_names or ASSET_ORDER[:n]
    Sigma_psd = _nearest_psd(Sigma)

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0, w <= upper_bound]

    if asset_caps:
        for name, cap in asset_caps.items():
            if name in asset_names:
                idx = asset_names.index(name)
                constraints.append(w[idx] <= cap)
    if asset_floors:
        for name, floor in asset_floors.items():
            if name in asset_names:
                idx = asset_names.index(name)
                constraints.append(w[idx] >= floor)

    objective = cp.Minimize(cp.quad_form(w, Sigma_psd))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if w.value is None or not np.isfinite(w.value).all():
            raise ValueError("degenerate solve")
        wv = np.clip(np.asarray(w.value).flatten(), 0, None)
        if wv.sum() <= 1e-9:
            raise ValueError("all-zero solve")
        wv = wv / wv.sum()
    except Exception:
        wv = _fallback_equal_weight(n)
    return wv


def solve_regime_weights(regime: str, mu: np.ndarray, Sigma: np.ndarray,
                          asset_names: list[str] | None = None) -> np.ndarray:
    """Dispatches to the regime-appropriate optimizer. This is the single
    function backtest.py calls at every rebalance."""
    asset_names = asset_names or ASSET_ORDER[: len(mu)]

    if regime == "Bull":
        return max_sharpe_weights(mu, Sigma, upper_bound=1.0)

    elif regime == "Bear":
        return min_variance_weights(
            mu, Sigma, upper_bound=1.0,
            asset_caps={"EQUITY": 0.45},
            asset_names=asset_names,
        )

    elif regime == "Crisis":
        return min_variance_weights(
            mu, Sigma, upper_bound=1.0,
            asset_caps={"EQUITY": 0.15},
            asset_floors={"GOLD": 0.30},
            asset_names=asset_names,
        )
    else:
        raise ValueError(f"Unknown regime label: {regime!r}")


def static_weights(kind: str, asset_names: list[str] | None = None) -> np.ndarray:
    """Fixed-weight benchmark portfolios (never re-optimized)."""
    asset_names = asset_names or ASSET_ORDER
    n = len(asset_names)
    if kind == "60_40":
        w = np.zeros(n)
        if "EQUITY" in asset_names:
            w[asset_names.index("EQUITY")] = 0.60
        if "BONDS" in asset_names:
            w[asset_names.index("BONDS")] = 0.40
        if w.sum() == 0:
            w = _fallback_equal_weight(n)
        else:
            w = w / w.sum()
        return w
    elif kind == "equal_weight":
        return _fallback_equal_weight(n)
    else:
        raise ValueError(f"Unknown static portfolio kind: {kind!r}")


def _nearest_psd(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetrizes and lightly ridges a covariance estimate so cvxpy's
    quad_form never chokes on a matrix that's technically not quite PSD due
    to floating point noise on small estimation windows."""
    S = (Sigma + Sigma.T) / 2
    S = S + eps * np.eye(S.shape[0])
    return S


def _project_with_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Simple water-filling projection: clip anything above `cap`, redistribute
    the excess proportionally to uncapped assets, iterate to convergence."""
    w = w.copy()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.sum() == 0:
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()
