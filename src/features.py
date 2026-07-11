"""
features.py
-----------
Turns raw aligned prices (+ VIX) into the numerical feature matrix the HMM
actually sees. Three feature families, chosen deliberately:

    momentum    -> direction: is the market trending, and over what horizon?
    volatility  -> uncertainty: crisis regimes are defined far more reliably
                   by vol spikes than by direction alone.
    VIX level   -> an independent, market-implied (forward-looking, not just
                   realized) fear gauge that often leads realized vol.

All rolling/expanding windows here are strictly backward-looking by
construction (pandas `.rolling(N)` / `.expanding()` never reach into the
future), which is the first line of defense against lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = (5, 21, 63, 126)     # ~1wk, 1mo, 1qtr, 6mo (trading days)
VOL_WINDOWS = (5, 21, 63)               # ~1wk, 1mo, 1qtr

# The subset of engineered columns actually fed into the HMM.
HMM_FEATURE_COLS = [
    "eq_log_ret",
    "mom_21d",
    "mom_63d",
    "vol_21d",
    "vix_level",
    "vix_chg_5d",
]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns for every asset column. Log returns are used throughout
    because they're additive across time (Week-2/Week-1 material) — a
    convenient property for building multi-horizon features cleanly."""
    log_ret = np.log(prices).diff()
    return log_ret


def build_feature_frame(prices: pd.DataFrame, vix: pd.Series,
                         equity_col: str = "EQUITY") -> pd.DataFrame:
    """
    Builds the full engineered feature DataFrame from raw EQUITY/BONDS/GOLD
    prices and the VIX level. Every column here is causal: value at row `t`
    is a function of data with index <= t only.

    Returns a DataFrame indexed identically to `prices`, containing:
      - eq_log_ret                  daily log return of the equity leg
      - mom_{5,21,63,126}d          equity price momentum at each horizon
      - vol_{5,21,63}d              annualized realized vol of equity log-returns
      - vix_level                   raw India VIX level
      - vix_chg_5d                  5-day change in VIX (captures vol-of-vol/fear spikes)
      - bond_eq_corr_63d            rolling 63d correlation of bond vs equity returns
                                     (this flips sign in genuine flight-to-quality events,
                                     a classically useful crisis-regime signal)
    """
    log_ret = compute_log_returns(prices)

    feat = pd.DataFrame(index=prices.index)
    feat["eq_log_ret"] = log_ret[equity_col]

    for w in MOMENTUM_WINDOWS:
        feat[f"mom_{w}d"] = prices[equity_col].pct_change(w)

    for w in VOL_WINDOWS:
        feat[f"vol_{w}d"] = log_ret[equity_col].rolling(w).std() * np.sqrt(252)

    feat["vix_level"] = vix
    feat["vix_chg_5d"] = vix.diff(5)

    other_asset = [c for c in prices.columns if c not in (equity_col,)][0]
    feat["bond_eq_corr_63d"] = (
        log_ret[equity_col].rolling(63).corr(log_ret["BONDS"])
        if "BONDS" in log_ret.columns else np.nan
    )

    feat = feat.dropna()
    return feat


def expanding_zscore(series: pd.Series, min_periods: int = 63) -> pd.Series:
    """
    Safe (leakage-free) z-score: at every time t, mu/sigma are computed using
    only observations up to and including t (an expanding window), never the
    full-sample mean/std.

    `min_periods` guards against absurd z-scores from a 2-3 observation
    "expanding" window early in the series (a std computed from almost no
    data is nearly meaningless and can blow up the z-score).

    NOTE: this function is provided for exploratory/demo use (e.g. the "look
    at the whole history" plots in the notebook). Inside the actual
    walk-forward backtest (backtest.py), scaling is instead fit strictly on
    each fold's TRAINING slice and applied to that fold's test slice — see
    `src/walkforward.py` and `fit_scaler` / `apply_scaler` below.
    """
    mu = series.expanding(min_periods=min_periods).mean()
    sigma = series.expanding(min_periods=min_periods).std()
    return (series - mu) / sigma


def fit_scaler(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit a z-score scaler (mean, std) using ONLY the training slice.
    This is the walk-forward-safe analogue of sklearn's `scaler.fit(X_train)`."""
    mu = train_df.mean()
    sigma = train_df.std().replace(0, 1e-8)
    return mu, sigma


def apply_scaler(df: pd.DataFrame, mu: pd.Series, sigma: pd.Series) -> pd.DataFrame:
    """Apply a previously-fit (train-only) scaler to any slice (train or test)."""
    return (df - mu) / sigma
