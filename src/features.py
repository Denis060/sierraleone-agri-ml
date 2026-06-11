"""
features.py
-----------
No-leakage feature engineering for rice-yield prediction.

THE ONE RULE: every predictor for year *t* must be knowable *before year t
begins*. That means:
  • NO same-year feature of any kind (no Rice_Production in year t, etc.).
  • Cross-crop signals enter only as their year *t-1* value (lag-1).
  • Rolling means are computed on PAST years only — we shift(1) BEFORE rolling,
    so the window for year t covers t-1, t-2, … and never t itself.
  • We do NOT build a rice production-per-hectare feature: that is essentially
    the target (yield) and would be leakage.

Deterministic calendar features (linear trend, shock dummies) are allowed
because they are known in advance for every year.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import os
import pandas as pd

from . import config

# Column names of the two naive baselines (both are valid no-leakage features).
PERSISTENCE_COL = 'Rice_Yield_lag1'    # predict t = observed yield at t-1
ROLLING_COL     = 'Rice_Yield_roll3'   # predict t = mean of t-1, t-2, t-3


def build_features(wide: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build the modeling frame from the wide crop matrix.

    Parameters
    ----------
    wide : pd.DataFrame
        Year-indexed wide matrix from data_prep.load_and_pivot().

    Returns
    -------
    pd.DataFrame
        Year-indexed frame holding the target column (config.TARGET) plus every
        engineered, leakage-free predictor. No same-year crop values appear.
    """
    wide = wide.sort_index()
    year = wide.index.to_series()
    feat = pd.DataFrame(index=wide.index)

    # ── 1. Lag-1 of EVERY crop × element column ──────────────────────────────
    # This is the only channel through which any crop value (including rice's
    # own production/area) may enter — always shifted back one full year.
    for col in wide.columns:
        feat[f'{col}_lag1'] = wide[col].shift(1)

    # ── 2. Rice-yield autoregressive lags ────────────────────────────────────
    # lag1 already created above; add lag2 and lag3.
    feat['Rice_Yield_lag2'] = wide[config.TARGET].shift(2)
    feat['Rice_Yield_lag3'] = wide[config.TARGET].shift(3)

    # ── 3. Past-only rolling means (shift BEFORE rolling) ────────────────────
    past_yield = wide[config.TARGET].shift(1)            # strictly < year t
    feat['Rice_Yield_roll3'] = past_yield.rolling(3).mean()
    feat['Rice_Yield_roll5'] = past_yield.rolling(5).mean()

    # ── 4. Deterministic calendar features (known a-priori) ──────────────────
    feat['year_trend']  = year - year.min()
    feat['Ebola']       = year.between(*config.EBOLA_YEARS).astype(int)
    feat['COVID']       = year.between(*config.COVID_YEARS).astype(int)
    feat['FeedSalone']  = (year >= config.FEED_SALONE_FROM).astype(int)

    # ── Attach target and drop incomplete (early) rows ───────────────────────
    feat[config.TARGET] = wide[config.TARGET]
    feature_cols = [c for c in feat.columns if c != config.TARGET]

    before = len(feat)
    feat = feat.dropna(subset=feature_cols + [config.TARGET])
    after = len(feat)

    # Guard: confirm no same-year crop column slipped in.
    leak = [c for c in feature_cols
            if c in wide.columns]   # a bare crop_element name == same-year value
    assert not leak, f'Leakage: same-year columns present -> {leak}'

    if verbose:
        print(f'[features] {len(feature_cols)} predictors | '
              f'usable years {feat.index.min()}–{feat.index.max()} '
              f'({after} rows; dropped {before - after} early rows)')

    return feat


def get_feature_cols(feat: pd.DataFrame) -> list:
    """Return the predictor column names (everything except the target)."""
    return [c for c in feat.columns if c != config.TARGET]


def save_features(feat: pd.DataFrame) -> str:
    os.makedirs(config.PROC_DIR, exist_ok=True)
    path = os.path.join(config.PROC_DIR, 'model_frame.csv')
    feat.to_csv(path)
    print(f'[features] saved {path}')
    return path
