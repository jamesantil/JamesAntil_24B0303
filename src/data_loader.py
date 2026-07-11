"""
data_loader.py
--------------
Pulls and aligns the multi-asset price universe used throughout the project.

Universe (chosen to represent three genuinely different asset classes, as required
by the problem statement, all tradeable on / linked to Indian markets):

    EQUITY  ^NSEI          Nifty 50 index                (growth / risk-on)
    BONDS   LTGILTBEES.NS  Nippon India ETF Long Term Gilt (rate-sensitive duration asset)
    GOLD    GOLDBEES.NS    Nippon India ETF Gold BeES     (safe-haven / inflation hedge)
    VIX     ^INDIAVIX      India VIX                      (fear gauge, feature only,
                                                             NOT a tradeable portfolio asset)

Why these specific tickers
---------------------------
- ^NSEI is the standard broad-market Indian equity benchmark.
- LTGILTBEES.NS (Nippon India ETF Long Term Gilt, ISIN INF204KB1882, tracks the
  Nifty 8-13yr G-Sec Index) is a long-duration government-bond ETF, so it actually
  behaves like
  a "bond" (rallies when rates fall / equities panic) rather than a cash-like
  instrument. If it's ever unavailable from Yahoo, the loader automatically falls
  back to LIQUIDBEES.NS (a liquid/money-market ETF) so the pipeline never silently
  breaks — this fallback is logged loudly so you know which one was actually used.
- GOLDBEES.NS is the most liquid, longest-running gold ETF on the NSE.
- ^INDIAVIX is used purely as a regime-detection FEATURE (Section: features.py) —
  it is intentionally excluded from the optimizable portfolio, because VIX itself
  is not a directly investable spot asset.

A note on `auto_adjust`
------------------------
We keep `auto_adjust=True` (yfinance default) so prices are already adjusted for
dividends/splits. Using unadjusted closes would inject fake jumps around dividend
dates - its own subtle source of bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EQUITY_TICKER = "^NSEI"
BOND_TICKER_PRIMARY = "LTGILTBEES.NS"
BOND_TICKER_FALLBACK = "LIQUIDBEES.NS"
GOLD_TICKER = "GOLDBEES.NS"
VIX_TICKER = "^INDIAVIX"

ASSET_NAMES = ["EQUITY", "BONDS", "GOLD"]


def _download_one(ticker: str, start: str, end: str) -> pd.Series:
    """Download a single ticker's adjusted close as a named Series. Returns
    an empty Series (not an exception) on failure, so callers can decide
    what to do (e.g. try a fallback ticker)."""
    import yfinance as yf

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float, name=ticker)
        # yfinance sometimes returns a MultiIndex column frame even for one ticker
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].iloc[:, 0]
        else:
            close = df["Close"]
        close.name = ticker
        return close.dropna()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[data_loader] WARNING: failed to download {ticker}: {exc}")
        return pd.Series(dtype=float, name=ticker)


def load_universe(start: str = "2010-01-01", end: str = "2025-01-01") -> tuple[pd.DataFrame, pd.Series]:
    """
    Downloads EQUITY, BONDS, GOLD (the tradeable portfolio universe) and VIX
    (feature-only) from Yahoo Finance, aligns them on a common trading-day
    index via an inner join, and returns:

        prices  : DataFrame[EQUITY, BONDS, GOLD]   (adjusted close, aligned)
        vix     : Series                            (India VIX level, aligned to prices.index)

    Raises RuntimeError if the core equity/gold series can't be fetched at all
    (almost certainly a connectivity issue) so failures are loud, not silent.
    """
    equity = _download_one(EQUITY_TICKER, start, end)
    equity.name = "EQUITY"

    bonds = _download_one(BOND_TICKER_PRIMARY, start, end)
    if bonds.empty:
        print(f"[data_loader] '{BOND_TICKER_PRIMARY}' unavailable, "
              f"falling back to '{BOND_TICKER_FALLBACK}'.")
        bonds = _download_one(BOND_TICKER_FALLBACK, start, end)
    bonds.name = "BONDS"

    gold = _download_one(GOLD_TICKER, start, end)
    gold.name = "GOLD"

    vix = _download_one(VIX_TICKER, start, end)
    vix.name = "VIX"

    if equity.empty or gold.empty or bonds.empty:
        raise RuntimeError(
            "Could not download the core price universe from Yahoo Finance. "
            "Check your internet connection, or re-run with use_synthetic=True "
            "in main.py / the notebook to exercise the pipeline offline."
        )

    prices = pd.concat([equity, bonds, gold], axis=1, join="inner").sort_index()
    prices = prices.dropna()

    vix = vix.reindex(prices.index).ffill()  # VIX occasionally has isolated NaNs; ffill, never bfill

    print(f"[data_loader] Loaded {len(prices)} aligned trading days "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")
    print(f"[data_loader] NaN check post-align:\n{prices.isna().sum().to_string()}")

    return prices, vix


def generate_synthetic_universe(start: str = "2010-01-01", end: str = "2025-01-01",
                                 seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    OFFLINE FALLBACK ONLY — NOT used for the reported results.

    Generates a synthetic EQUITY/BONDS/GOLD/VIX universe with regime-switching
    dynamics (calm periods, a couple of engineered crashes with vol spikes and
    a flight-to-quality bond/gold rally), purely so the *pipeline* (this repo's
    code) can be smoke-tested end-to-end without a network connection.

    Used only when `main.py --demo` / `use_synthetic=True` is passed explicitly.
    The actual submitted results in `outputs/` are generated from real
    `load_universe()` data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Hidden "true" regime path used only to generate synthetic data (never
    # exposed to the model — it must recover something like this on its own).
    regime = np.zeros(n, dtype=int)  # 0=Bull, 1=Bear, 2=Crisis
    t = 0
    while t < n:
        r = rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15])
        length = rng.integers(40, 260)
        regime[t:t + length] = r
        t += length

    mu_map = {0: 0.0006, 1: -0.0003, 2: -0.0020}     # equity daily drift by regime
    sig_map = {0: 0.008, 1: 0.013, 2: 0.032}          # equity daily vol by regime

    eq_ret = np.array([rng.normal(mu_map[r], sig_map[r]) for r in regime])
    # Bonds/gold: mild negative correlation to equity in Crisis (flight to quality)
    bond_ret = np.empty(n)
    gold_ret = np.empty(n)
    for i, r in enumerate(regime):
        beta_b = {0: 0.05, 1: -0.05, 2: -0.35}[r]
        beta_g = {0: -0.05, 1: 0.10, 2: 0.40}[r]
        bond_ret[i] = 0.00015 + beta_b * eq_ret[i] + rng.normal(0, 0.003)
        gold_ret[i] = 0.00020 + beta_g * eq_ret[i] + rng.normal(0, 0.006)

    vix_level = 12 + 200 * pd.Series(eq_ret).rolling(21, min_periods=1).std().values
    vix_level = np.clip(vix_level + rng.normal(0, 1.0, n), 8, 90)

    equity_px = 100 * np.exp(np.cumsum(eq_ret))
    bond_px = 100 * np.exp(np.cumsum(bond_ret))
    gold_px = 100 * np.exp(np.cumsum(gold_ret))

    prices = pd.DataFrame(
        {"EQUITY": equity_px, "BONDS": bond_px, "GOLD": gold_px}, index=dates
    )
    vix = pd.Series(vix_level, index=dates, name="VIX")

    print(f"[data_loader] SYNTHETIC universe generated: {n} days "
          f"({dates[0].date()} -> {dates[-1].date()}). NOT real market data.")
    return prices, vix
