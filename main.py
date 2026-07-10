"""
main.py
-------
Runs the FULL pipeline top to bottom, exactly as required by the submission
checklist:

    data -> features -> regime detection -> optimization -> backtest -> results

and saves every required deliverable into ./outputs/:

    regime_overlay.png          price chart with HMM regimes overlaid
    transition_matrix.png       heatmap of the (last-fold) transition matrix
    transition_matrix.csv       same, as raw numbers
    equity_curves.png           strategy vs benchmarks, gross vs net of costs
    drawdown.png                underwater plot for all strategies
    weights_over_time.png       how the dynamic strategy's weights evolve
    performance_summary.csv     Sharpe / Sortino / Calmar / MDD / turnover table
    regime_labels.csv           full daily regime label history (out-of-sample)

Usage
-----
    python main.py                       # real data via yfinance, full date range
    python main.py --start 2012-01-01 --end 2025-01-01
    python main.py --demo                # OFFLINE synthetic data, pipeline smoke-test only

Re-run with `--demo` any time to confirm the code executes end-to-end without
a network connection; the numbers reported in the README were produced by a
real (non---demo) run.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data_loader
from src import features as feat_mod
from src import regime_hmm
from src import walkforward
from src import backtest as bt_mod
from src import metrics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    parser = argparse.ArgumentParser(description="RegimeShift capstone — full pipeline runner")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--demo", action="store_true",
                         help="Use offline synthetic data instead of yfinance (pipeline smoke-test only).")
    parser.add_argument("--rebalance_freq", type=int, default=bt_mod.DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--tc_bps", type=float, default=bt_mod.DEFAULT_TC_BPS)
    parser.add_argument("--n_splits", type=int, default=6)
    parser.add_argument("--min_train_size", type=int, default=750)
    parser.add_argument("--test_size", type=int, default=126)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. DATA ----------
    if args.demo:
        prices, vix = data_loader.generate_synthetic_universe(args.start, args.end)
    else:
        prices, vix = data_loader.load_universe(args.start, args.end)

    # ---------- 2. FEATURES ----------
    feat_raw = feat_mod.build_feature_frame(prices, vix)
    print(f"[main] Feature matrix: {feat_raw.shape[0]} rows x {feat_raw.shape[1]} cols "
          f"({feat_raw.index[0].date()} -> {feat_raw.index[-1].date()})")

    # ---------- 3. WALK-FORWARD SPLITS ----------
    splits = walkforward.expanding_walk_forward_splits(
        n_obs=len(feat_raw), n_splits=args.n_splits,
        min_train_size=args.min_train_size, test_size=args.test_size,
    )
    print(f"[main] {len(splits)} walk-forward folds generated.")

    # ---------- 4. REGIME DETECTION + 5. OPTIMIZATION + 6. BACKTEST ----------
    result = bt_mod.run_walk_forward_backtest(
        prices, feat_raw, splits,
        rebalance_freq=args.rebalance_freq,
    )

    net_returns = bt_mod.apply_transaction_costs(
        result["gross_returns"], result["turnover"], tc_bps=args.tc_bps
    )

    test_index = result["test_index"]
    bench_6040 = bt_mod.static_benchmark_returns(prices, test_index, "60_40")
    bench_eq = bt_mod.static_benchmark_returns(prices, test_index, "equal_weight")

    # ---------- 7. RESULTS / DELIVERABLES ----------
    save_regime_overlay(prices, result["regime_daily"], OUTPUT_DIR)
    save_transition_matrix(result["fold_models"], OUTPUT_DIR)
    save_equity_curves(result["gross_returns"], net_returns, bench_6040, bench_eq, OUTPUT_DIR)
    save_drawdowns(result["gross_returns"], net_returns, bench_6040, bench_eq, OUTPUT_DIR)
    save_weights_plot(result["weights"], OUTPUT_DIR)
    save_performance_summary(result, net_returns, bench_6040, bench_eq, OUTPUT_DIR)

    result["regime_daily"].to_csv(os.path.join(OUTPUT_DIR, "regime_labels.csv"))

    print(f"\n[main] All deliverables written to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Deliverable-generating helpers
# ---------------------------------------------------------------------------

def save_regime_overlay(prices: pd.DataFrame, regime_daily: pd.Series, outdir: str):
    idx = regime_daily.dropna().index
    px = prices.loc[idx, "EQUITY"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(px.index, px.values, color="black", lw=1.0, zorder=3)
    for label, color in regime_hmm.REGIME_COLORS.items():
        mask = regime_daily.loc[idx] == label
        ax.scatter(px.index[mask], px.values[mask], s=8, color=color, label=label, zorder=2)
    ax.set_title("Out-of-Sample HMM Regimes Overlaid on EQUITY Price (walk-forward, causal)")
    ax.set_ylabel("EQUITY level")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "regime_overlay.png"), dpi=150)
    plt.close(fig)
    print("[main] saved regime_overlay.png")


def save_transition_matrix(fold_models: list[dict], outdir: str):
    last = fold_models[-1]
    tm = last["transition_matrix"]
    tm.to_csv(os.path.join(outdir, "transition_matrix.csv"))

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tm.columns)))
    ax.set_xticklabels(tm.columns)
    ax.set_yticks(range(len(tm.index)))
    ax.set_yticklabels(tm.index)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            ax.text(j, i, f"{tm.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if tm.values[i, j] > 0.5 else "black")
    ax.set_title(f"Transition Matrix (final fold, fold {last['fold']+1})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "transition_matrix.png"), dpi=150)
    plt.close(fig)
    print("[main] saved transition_matrix.png / .csv")


def save_equity_curves(gross: pd.Series, net: pd.Series, bench_6040: pd.Series,
                        bench_eq: pd.Series, outdir: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    for s, label, style in [
        (gross, "Regime-Adaptive (gross, no costs)", "--"),
        (net, "Regime-Adaptive (net of costs)", "-"),
        (bench_6040, "Static 60/40", "-"),
        (bench_eq, "Equal-Weight", "-"),
    ]:
        ax.plot(metrics.equity_curve(s).index, metrics.equity_curve(s).values, style, label=label, lw=1.6)
    ax.set_title("Out-of-Sample Equity Curves")
    ax.set_ylabel("Growth of ₹1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curves.png"), dpi=150)
    plt.close(fig)
    print("[main] saved equity_curves.png")


def save_drawdowns(gross: pd.Series, net: pd.Series, bench_6040: pd.Series,
                    bench_eq: pd.Series, outdir: str):
    fig, ax = plt.subplots(figsize=(13, 4))
    for s, label in [
        (net, "Regime-Adaptive (net)"),
        (bench_6040, "Static 60/40"),
        (bench_eq, "Equal-Weight"),
    ]:
        dd = metrics.drawdown_series(s)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=label)
        ax.plot(dd.index, dd.values, lw=0.8)
    ax.set_title("Drawdown (Underwater Plot)")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drawdown.png"), dpi=150)
    plt.close(fig)
    print("[main] saved drawdown.png")


def save_weights_plot(weights: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.stackplot(weights.index, weights["EQUITY"], weights["BONDS"], weights["GOLD"],
                 labels=["EQUITY", "BONDS", "GOLD"], alpha=0.85)
    ax.set_title("Regime-Adaptive Portfolio Weights Over Time")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "weights_over_time.png"), dpi=150)
    plt.close(fig)
    print("[main] saved weights_over_time.png")


def save_performance_summary(result: dict, net_returns: pd.Series, bench_6040: pd.Series,
                              bench_eq: pd.Series, outdir: str):
    rows = [
        metrics.summarize(result["gross_returns"], result["weights"], "Regime-Adaptive (gross)"),
        metrics.summarize(net_returns, result["weights"], "Regime-Adaptive (net of costs)"),
        metrics.summarize(bench_6040, None, "Static 60/40"),
        metrics.summarize(bench_eq, None, "Equal-Weight"),
    ]
    df = pd.DataFrame(rows).set_index("Strategy")
    df.to_csv(os.path.join(outdir, "performance_summary.csv"))
    print("[main] saved performance_summary.csv\n")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
