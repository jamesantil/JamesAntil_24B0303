"""
regime_hmm.py
-------------
Fits a Gaussian HMM (via `hmmlearn`) to the engineered feature matrix and
turns its arbitrary integer state labels (0, 1, 2) into human-readable
regime names (Bull / Bear / Crisis).

Why n_components = 3
---------------------
This is a modeling CHOICE, not something the HMM discovers on its own, and
it's the direct, literal translation of the project brief: "detects what
mood the market is in (bull/bear/crisis)". We tested 2 and 4-state variants
during development (see README "Key Decisions"): 2 states collapses Bear and
Crisis into one indistinguishable "down" state (loses exactly the
distinction the project asks us to make); 4+ states starts splitting Bull
into "Bull" / "Bull-strong" sub-states that don't correspond to anything
economically meaningful and are much harder to interpret and act on with a
3-way regime-conditional optimizer. 3 is the smallest model that actually
answers the question being asked.

Why covariance_type = "diag"
------------------------------
With only ~6 features and a training window that (in the early walk-forward
folds) can be as small as ~500-750 observations, a full covariance matrix
(21 free parameters per state for 6 features) is a lot to estimate reliably.
"diag" (6 params per state) assumes features are conditionally uncorrelated
within a regime — a simplification, but a much lower-variance one, and it
keeps Baum-Welch numerically stable on the smaller early training folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm

N_STATES = 3
COVARIANCE_TYPE = "diag"
N_ITER = 200
RANDOM_STATE = 42

REGIME_NAMES = ("Bull", "Bear", "Crisis")
REGIME_COLORS = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}


def fit_hmm(X_train: np.ndarray, n_states: int = N_STATES,
            covariance_type: str = COVARIANCE_TYPE,
            random_state: int = RANDOM_STATE) -> hmm.GaussianHMM:
    """Fits a GaussianHMM on a TRAINING slice only (X_train must already be
    z-scored using train-only statistics — see features.fit_scaler)."""
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=N_ITER,
        random_state=random_state,
    )
    model.fit(X_train)
    return model


def label_states(model: hmm.GaussianHMM, feat_raw_train: pd.DataFrame,
                  state_sequence_train: np.ndarray,
                  vol_col: str = "vol_21d", mom_col: str = "mom_21d") -> dict[int, str]:
    """
    Maps arbitrary HMM state indices {0, ..., n_states-1} to {"Bull","Bear","Crisis"}.

    Heuristic (applied on RAW, un-scaled training features, using the state
    labels the model itself assigned to the training data — no manual
    day-by-day labelling by us, satisfying the "without manually labelling
    any days" requirement):

        1. The state with the highest mean realized volatility -> "Crisis".
           (Crisis periods are far more reliably identified by vol spikes
           than by direction — this is the single most robust cut.)
        2. Of the two remaining states, the one with the higher mean
           momentum -> "Bull", the other -> "Bear".

    This mapping is refit inside every walk-forward fold (state indices are
    arbitrary per-fit, so a fixed global mapping would be wrong) — see
    `backtest.py`.
    """
    tmp = feat_raw_train.copy()
    tmp["state"] = state_sequence_train

    stats = tmp.groupby("state")[[vol_col, mom_col]].mean()

    crisis_state = stats[vol_col].idxmax()
    remaining = [s for s in stats.index if s != crisis_state]
    if len(remaining) == 2:
        bull_state = stats.loc[remaining, mom_col].idxmax()
        bear_state = [s for s in remaining if s != bull_state][0]
    else:
        # degenerate case (fewer than 3 distinct states actually occupied)
        bull_state = remaining[0] if remaining else crisis_state
        bear_state = remaining[0] if remaining else crisis_state

    mapping = {crisis_state: "Crisis", bull_state: "Bull", bear_state: "Bear"}
    # Ensure every possible state index has *some* label even if a state was
    # never realized in this particular training fold.
    for s in range(model.n_components):
        mapping.setdefault(s, "Bear")
    return mapping


def causal_predict_regimes(model: hmm.GaussianHMM, train_scaled: np.ndarray,
                            test_scaled: np.ndarray, rebalance_idx: np.ndarray) -> np.ndarray:
    """
    THE key leakage-safety function for regime inference at test time.

    For each requested rebalance position `i` in the test fold (0-indexed
    within the fold), decodes the most likely hidden-state PATH using only
    `train_scaled` (already-fit model's training data, for HMM context) plus
    `test_scaled[:i+1]` (the test observations UP TO AND INCLUDING day i) —
    and takes only the LAST state of that decoded path.

    This is deliberately NOT the cheaper `model.predict(test_scaled)` batch
    call, which would run Viterbi smoothing over the ENTIRE test fold at
    once — meaning the regime label assigned to day 5 of a 126-day test fold
    could, in principle, be influenced by what happens on day 120 of that
    same fold. That is a real, if subtle, form of lookahead bias. Decoding
    the growing prefix instead guarantees the label at day i is a function
    of data with index <= i only, exactly the standard from Section 7 of
    the project guide.

    `rebalance_idx` lets the caller only pay this (more expensive) causal
    decode on actual rebalance dates (e.g. weekly) rather than every single
    day, which is both realistic (nobody rebalances tick-by-tick) and
    keeps runtime reasonable.
    """
    labels = np.full(len(rebalance_idx), -1, dtype=int)
    for j, i in enumerate(rebalance_idx):
        prefix = np.vstack([train_scaled, test_scaled[: i + 1]])
        state_path = model.predict(prefix)  # Viterbi, but only over causal data
        labels[j] = state_path[-1]
    return labels


def batch_predict_regimes(model: hmm.GaussianHMM, test_scaled: np.ndarray) -> np.ndarray:
    """
    Faster, NON-causal-within-fold alternative (Viterbi smoothing over the
    whole test fold at once). Kept only for a side-by-side runtime/accuracy
    comparison in the notebook (see 'Causal vs batch decoding' section) —
    NOT used for the reported backtest results.
    """
    return model.predict(test_scaled)


def transition_matrix_df(model: hmm.GaussianHMM, label_map: dict[int, str]) -> pd.DataFrame:
    """Returns the fitted transition matrix with human-readable regime names
    as both row and column labels, ordered Bull/Bear/Crisis for readability."""
    n = model.n_components
    names = [label_map[s] for s in range(n)]
    tm = pd.DataFrame(model.transmat_, index=names, columns=names)
    order = [r for r in REGIME_NAMES if r in tm.index]
    tm = tm.loc[order, order]
    return tm
