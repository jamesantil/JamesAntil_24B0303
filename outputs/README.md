# About the files in this folder

These artifacts (`regime_overlay.png`, `transition_matrix.png/.csv`,
`equity_curves.png`, `drawdown.png`, `weights_over_time.png`,
`performance_summary.csv`, `regime_labels.csv`) were committed by running:

```bash
python main.py --demo
```

i.e. against the **offline synthetic data generator** in
`src/data_loader.generate_synthetic_universe`, not real Yahoo Finance data —
this sandbox environment does not have outbound internet access to Yahoo
Finance, so this is the only way to commit a fully-run, reproducible set of
artifacts alongside the code.

**To regenerate these with real NSE/gold/gilt/VIX data**, simply run:

```bash
python main.py
```

(no `--demo` flag) from a machine with normal internet access. This will
overwrite every file in this folder with the real-data version, using the
exact same pipeline — `src/` code path is identical either way, only the
data source (`data_loader.load_universe` vs.
`data_loader.generate_synthetic_universe`) differs. The README.md at the
project root documents this explicitly as well.

The synthetic generator (see its docstring) creates a regime-switching
EQUITY/BONDS/GOLD/VIX universe with engineered calm periods and a few
crash-with-vol-spike-and-flight-to-quality episodes, specifically so the
full pipeline — HMM fitting, causal decoding, regime-conditional cvxpy
optimization, transaction costs, walk-forward evaluation — can be exercised
and sanity-checked end-to-end without a live data connection. It is not
intended, and should not be read, as a claim about real market performance.
