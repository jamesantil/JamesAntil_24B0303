"""
RegimeShift — Macro-Aware Tactical Asset Allocation Engine
Summer of Quant — Advanced Capstone

This package contains the full pipeline:
  data_loader   -> pull & align multi-asset prices (yfinance)
  features      -> momentum / volatility / VIX feature engineering
  regime_hmm    -> Gaussian HMM fitting, Viterbi decoding, regime labelling
  optimizer     -> cvxpy convex portfolio optimizers, one per regime
  walkforward   -> leakage-safe expanding-window train/test split generator
  backtest      -> the walk-forward backtest engine (causal regime decoding,
                    regime-conditional rebalancing, transaction costs)
  metrics       -> Sharpe / Sortino / Calmar / drawdown / turnover
"""
