"""
backtest.py
-----------
The engine that ties everything together: for every walk-forward fold, it

  1. fits the HMM on the TRAINING slice only (features scaled with
     train-only mean/std),
  2. labels the fitted states Bull/Bear/Crisis from train-only statistics,
  3. walks forward through the TEST slice day by day; on each scheduled
     rebalance date it CAUSALLY decodes the current regime (using only data
     up to and including that date - see `regime_hmm.causal_predict_regimes`),
     estimates mu/Sigma from a trailing window of strictly-historical asset
     returns, and solves the regime-appropriate cvxpy optimization,
  4. drifts portfolio weights with daily asset returns between rebalances,
     and records per-day gross returns plus per-rebalance turnover so
     transaction costs can be applied AFTER the fact at any bps level
     without re-running the (expensive) HMM fits.

All folds are walked in chronological order and their test-period results
are concatenated - the reported backtest covers ONLY the out-of-sample test
periods (the very first `min_train_size` observations, which the HMM needs
just to be fit for the first time, are never scored - scoring them would
require having already trained on data that didn't exist yet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as feat_mod
from . import regime_hmm
from . import optimizer as opt_mod

DEFAULT_REBALANCE_FREQ = 5          # trading days between rebalances (~weekly)
DEFAULT_MU_SIGMA_LOOKBACK = 126     # trailing window (trading days, ~6mo) for mu/Sigma estimation
DEFAULT_TC_BPS = 7.5                # transaction cost, basis points of turnover per rebalance


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    feat_raw: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    asset_names: list[str] | None = None,
    rebalance_freq: int = DEFAULT_REBALANCE_FREQ,
    mu_sigma_lookback: int = DEFAULT_MU_SIGMA_LOOKBACK,
    n_states: int = regime_hmm.N_STATES,
    verbose: bool = True,
) -> dict:
    """
    Runs the full leakage-safe walk-forward backtest.

    Returns a dict with:
        gross_returns       : pd.Series, daily portfolio returns before costs
        turnover            : pd.Series, per-day turnover (0 except on rebalance days)
        weights             : pd.DataFrame, daily post-drift weights per asset
        regime_at_rebalance : pd.Series, regime label at each rebalance date
        regime_daily        : pd.Series, regime label held on every day (ffilled)
        fold_models         : list of (label_map, transition_matrix_df) per fold
        test_index          : DatetimeIndex covered by the backtest (union of all test folds)
    """
    asset_names = asset_names or opt_mod.ASSET_ORDER
    asset_returns = prices[asset_names].pct_change()

    all_gross_returns = []
    all_turnover = []
    all_weights = []
    all_regime_at_rebalance = []
    fold_models = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_feat_raw = feat_raw.iloc[train_idx][feat_mod.HMM_FEATURE_COLS]
        test_feat_raw = feat_raw.iloc[test_idx][feat_mod.HMM_FEATURE_COLS]

        mu_scale, sigma_scale = feat_mod.fit_scaler(train_feat_raw)
        train_scaled = feat_mod.apply_scaler(train_feat_raw, mu_scale, sigma_scale).values
        test_scaled = feat_mod.apply_scaler(test_feat_raw, mu_scale, sigma_scale).values

        model = regime_hmm.fit_hmm(train_scaled, n_states=n_states)
        train_states = model.predict(train_scaled)
        label_map = regime_hmm.label_states(model, feat_raw.iloc[train_idx], train_states)
        tm_df = regime_hmm.transition_matrix_df(model, label_map)
        fold_models.append({"fold": fold_i, "label_map": label_map, "transition_matrix": tm_df})

        test_dates = feat_raw.index[test_idx]
        rebalance_positions = np.arange(0, len(test_idx), rebalance_freq)

        regime_states_at_rebal = regime_hmm.causal_predict_regimes(
            model, train_scaled, test_scaled, rebalance_positions
        )
        regime_labels_at_rebal = [label_map[s] for s in regime_states_at_rebal]
        rebalance_date_set = {test_dates[p]: lbl for p, lbl in zip(rebalance_positions, regime_labels_at_rebal)}

        if verbose:
            counts = pd.Series(regime_labels_at_rebal).value_counts().to_dict()
            print(f"[backtest] Fold {fold_i+1}/{len(splits)}  "
                  f"train={feat_raw.index[train_idx][0].date()}..{feat_raw.index[train_idx][-1].date()}  "
                  f"test={test_dates[0].date()}..{test_dates[-1].date()}  "
                  f"regimes-at-rebalance={counts}")

        current_weights = None
        for d in test_dates:
            day_turnover = 0.0
            if d in rebalance_date_set:
                regime = rebalance_date_set[d]
                mu, Sigma = _estimate_moments(asset_returns, d, mu_sigma_lookback)
                new_weights = opt_mod.solve_regime_weights(regime, mu, Sigma, asset_names)
                if current_weights is not None:
                    day_turnover = float(np.abs(new_weights - current_weights).sum())
                else:
                    day_turnover = float(np.abs(new_weights).sum())
                current_weights = new_weights
                all_regime_at_rebalance.append((d, regime))

            r_today = asset_returns.loc[d].values
            if np.isnan(r_today).any():
                # first day of the series has no prior-day return; skip contribution
                r_today = np.nan_to_num(r_today)

            gross_return_today = float(current_weights @ r_today)

            all_gross_returns.append((d, gross_return_today))
            all_turnover.append((d, day_turnover))
            all_weights.append((d, current_weights.copy()))

            grown = current_weights * (1 + r_today)
            denom = grown.sum()
            current_weights = grown / denom if denom > 0 else current_weights

    gross_returns = pd.Series(dict(all_gross_returns)).sort_index()
    gross_returns.name = "gross_return"
    turnover_s = pd.Series(dict(all_turnover)).sort_index()
    turnover_s.name = "turnover"
    weights_df = pd.DataFrame(
        {d: w for d, w in all_weights}, index=asset_names
    ).T.sort_index()

    regime_at_rebalance = pd.Series(
        {d: r for d, r in all_regime_at_rebalance}
    ).sort_index()
    regime_at_rebalance.name = "regime"
    regime_daily = regime_at_rebalance.reindex(gross_returns.index).ffill()

    return {
        "gross_returns": gross_returns,
        "turnover": turnover_s,
        "weights": weights_df,
        "regime_at_rebalance": regime_at_rebalance,
        "regime_daily": regime_daily,
        "fold_models": fold_models,
        "test_index": gross_returns.index,
    }


def apply_transaction_costs(gross_returns: pd.Series, turnover: pd.Series,
                             tc_bps: float = DEFAULT_TC_BPS) -> pd.Series:
    """Net returns after charging `tc_bps` basis points on each unit of
    portfolio turnover incurred that day (0 on non-rebalance days)."""
    tc_rate = tc_bps / 10_000.0
    net = gross_returns - tc_rate * turnover
    net.name = "net_return"
    return net


def static_benchmark_returns(prices: pd.DataFrame, test_index: pd.DatetimeIndex,
                              kind: str, asset_names: list[str] | None = None) -> pd.Series:
    """Buy-and-hold-with-periodic-rebalance-to-target benchmark (rebalanced
    monthly back to fixed target weights, so it isn't just a pure drift
    portfolio - this is the standard definition of a '60/40' benchmark)."""
    asset_names = asset_names or opt_mod.ASSET_ORDER
    target_w = opt_mod.static_weights(kind, asset_names)
    asset_returns = prices[asset_names].pct_change().reindex(test_index)

    current_weights = target_w.copy()
    rets = []
    for i, d in enumerate(test_index):
        if i % 21 == 0:  # monthly rebalance back to target
            current_weights = target_w.copy()
        r_today = np.nan_to_num(asset_returns.loc[d].values)
        rets.append((d, float(current_weights @ r_today)))
        grown = current_weights * (1 + r_today)
        denom = grown.sum()
        current_weights = grown / denom if denom > 0 else current_weights

    s = pd.Series(dict(rets)).sort_index()
    s.name = f"{kind}_return"
    return s


def _estimate_moments(asset_returns: pd.DataFrame, as_of_date: pd.Timestamp,
                       lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window (annualized) mean and covariance of asset returns,
    using ONLY data strictly before `as_of_date` - the causality boundary
    for every optimization decision made in the backtest."""
    hist = asset_returns.loc[:as_of_date].iloc[:-1]  # exclude as_of_date itself
    hist = hist.dropna().tail(lookback)
    if len(hist) < 20:
        # Not enough history yet - shouldn't happen once past the initial
        # training window, but guard defensively.
        mu = np.zeros(asset_returns.shape[1])
        Sigma = np.eye(asset_returns.shape[1]) * 1e-4
        return mu, Sigma
    mu = hist.mean().values * 252
    Sigma = hist.cov().values * 252
    return mu, Sigma
