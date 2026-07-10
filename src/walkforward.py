"""
walkforward.py
---------------
Leakage-safe train/test split generator for time series. This is the
operational core of "avoid lookahead bias": every fold's model is fit ONLY
on the training slice, and only ever evaluated on the test slice that comes
strictly after it in time. Adjacent folds slide forward; nothing is shuffled.

Two flavors, both provided:

    expanding_walk_forward_splits  - training window grows every fold
                                      (always starts at obs 0). Use when you
                                      believe more history never hurts.
    rolling_walk_forward_splits    - training window is a fixed size that
                                      slides forward (older data drops off).
                                      Use if you're worried about regime
                                      drift making very old data actively
                                      misleading.

We use the EXPANDING variant as the default for this project (see README):
market regimes recur (a well-fit Crisis emission distribution from 2020 is
still useful context in 2022), and our dataset isn't long enough to
comfortably afford throwing away history in a rolling window.
"""

from __future__ import annotations

import numpy as np


def expanding_walk_forward_splits(n_obs: int, n_splits: int = 6,
                                   min_train_size: int = 750,
                                   test_size: int = 126) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Yields (train_idx, test_idx) index-array pairs.

    n_obs           total number of observations in the dataset
    n_splits        target number of folds (fewer will be returned if the
                     data runs out first)
    min_train_size  size of the very first training window (~3 years of
                     daily data by default - an HMM needs a reasonable
                     amount of history to estimate 3 states' worth of
                     emission distributions and a transition matrix)
    test_size       size of each test window (~6 months by default)
    """
    splits = []
    start_test = min_train_size
    for _ in range(n_splits):
        if start_test >= n_obs:
            break
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, min(start_test + test_size, n_obs))
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
        start_test += test_size
    return splits


def rolling_walk_forward_splits(n_obs: int, n_splits: int = 6,
                                 train_size: int = 750,
                                 test_size: int = 126) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed-size training window that slides forward (older data drops off
    as newer data enters). Provided as an alternative / robustness check —
    see the notebook's 'Rolling vs Expanding' comparison cell."""
    splits = []
    start_train = 0
    for _ in range(n_splits):
        end_train = start_train + train_size
        start_test = end_train
        end_test = min(start_test + test_size, n_obs)
        if start_test >= n_obs or end_test <= start_test:
            break
        train_idx = np.arange(start_train, end_train)
        test_idx = np.arange(start_test, end_test)
        splits.append((train_idx, test_idx))
        start_train += test_size
    return splits
