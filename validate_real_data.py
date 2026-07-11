"""
validate_real_data.py
----------------------
A genuine real-data sanity check, separate from the main synthetic --demo
smoke test.

WHAT THIS IS
------------
`data/real_market_data/NIFTY50_1990_2019_verified.csv` is REAL, VERIFIED
historical Nifty 50 daily OHLC data (1990-07-03 -> 2019-05-03, 6,974 trading
days), sourced from a public GitHub dataset and independently cross-checked
by hand against known history before being trusted:

    - 04 Jan 2010 close: 5232.20   (matches real Nifty ~5200-5300 in Jan 2010)
    - Nov 2008 closes:   ~2550-2800 (matches the real 2008 GFC crash low)
    - 03 May 2019 close: 11712.25  (matches real Nifty in early May 2019)

This script runs the ACTUAL regime-detection code from `src/regime_hmm.py`
and `src/features.py` (not a reimplementation, not a mock) on this real
series, and produces a real regime overlay chart at
`outputs/real_data_validation.png` — including the acid test: does the
model, using ONLY equity price/volatility features and no manual labelling,
correctly flag the Nov 2008 financial crisis as "Crisis"?

WHAT THIS IS NOT
----------------
This is a single-asset (EQUITY only) validation of the regime-DETECTION
component only. It does NOT run the full BONDS/GOLD walk-forward portfolio
backtest — that requires `LTGILTBEES.NS`, `GOLDBEES.NS`, and `^INDIAVIX`
daily history, which yfinance can pull live, but which no fetchable,
robots-permitting, non-JS-rendered public source made available for
offline retrieval when this repo was built (Yahoo Finance's history pages
require JS rendering; Stooq disallows automated fetching; NSE's own site
requires a browser session). Run `python main.py` (no `--demo`) with a
live internet connection to get the genuine full 3-asset backtest.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import features as feat_mod
from src import regime_hmm

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "real_market_data",
                          "NIFTY50_1990_2019_verified.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "real_data_validation.png")


def load_real_nifty() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    return df


def build_equity_only_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Equity-only analogue of features.build_feature_frame (no VIX/bonds
    available for this historical stretch, so those columns are dropped;
    momentum + volatility are computed with the exact same code path as
    the main pipeline)."""
    log_ret = np.log(prices["Close"]).diff()
    feat = pd.DataFrame(index=prices.index)
    feat["eq_log_ret"] = log_ret
    for w in feat_mod.MOMENTUM_WINDOWS:
        feat[f"mom_{w}d"] = prices["Close"].pct_change(w)
    for w in feat_mod.VOL_WINDOWS:
        feat[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
    return feat.dropna()


def main():
    print("[validate_real_data] Loading REAL, verified Nifty 50 data "
          "(1990-2019)...")
    prices = load_real_nifty()
    print(f"[validate_real_data] {len(prices)} real trading days: "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    feat = build_equity_only_features(prices)
    cols = ["eq_log_ret", "mom_21d", "mom_63d", "vol_21d"]
    X = feat[cols]

    mu, sigma = feat_mod.fit_scaler(X)  # full-sample fit is fine here —
    # this is a single-shot descriptive validation of the HMM/labelling
    # logic on real data, NOT a backtest, so there is no walk-forward
    # train/test boundary to respect.
    X_scaled = feat_mod.apply_scaler(X, mu, sigma).values

    model = regime_hmm.fit_hmm(X_scaled, n_states=3)
    states = model.predict(X_scaled)
    label_map = regime_hmm.label_states(model, feat, states,
                                         vol_col="vol_21d", mom_col="mom_21d")
    regimes = pd.Series([label_map[s] for s in states], index=feat.index)

    # --- The acid test: was Nov 2008 (the real GFC crash) flagged Crisis? ---
    gfc_window = regimes.loc["2008-10-01":"2008-11-30"]
    gfc_crisis_share = (gfc_window == "Crisis").mean() if len(gfc_window) else float("nan")
    print(f"\n[validate_real_data] Oct-Nov 2008 (real GFC crash) classified "
          f"as 'Crisis' on {gfc_crisis_share:.1%} of days "
          f"({len(gfc_window)} real trading days evaluated).")

    dotcom_window = regimes.loc["2000-03-01":"2001-09-30"]
    dotcom_crisis_share = (dotcom_window == "Crisis").mean() if len(dotcom_window) else float("nan")
    print(f"[validate_real_data] 2000-2001 (real dot-com bust) classified "
          f"as 'Crisis' on {dotcom_crisis_share:.1%} of days "
          f"({len(dotcom_window)} real trading days evaluated).")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    px = prices.loc[feat.index, "Close"]
    ax.plot(px.index, px.values, color="black", lw=0.7, zorder=3)
    for label, color in regime_hmm.REGIME_COLORS.items():
        mask = regimes == label
        ax.scatter(px.index[mask], px.values[mask], s=4, color=color, label=label, zorder=2)
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2008-12-31"),
               color="grey", alpha=0.15)
    ax.annotate("2008 GFC\n(real crash)", xy=(pd.Timestamp("2008-10-15"), px.max() * 0.15),
                fontsize=9, ha="center")
    ax.set_yscale("log")
    ax.set_title("REAL DATA — Nifty 50, 1990-2019 (verified) — HMM Regimes (equity-only features)")
    ax.set_ylabel("Nifty 50 (log scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\n[validate_real_data] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
