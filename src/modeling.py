"""
modeling.py
-----------
Model factory: the three ML regressors (all shallow / regularized) plus the two
naive baselines, all usable inside the same walk-forward loop.

Baselines are not "fit" — they read a precomputed, leakage-free feature column
(last year's yield, or the past-3-year mean), so they slot into the validation
framework exactly like the ML models.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from . import config
from . import features as F


def build_models() -> dict:
    """Return the three (untrained) ML regressors, conservatively configured."""
    return {
        'XGBoost':           XGBRegressor(**config.XGB_PARAMS),
        'GradientBoosting':  GradientBoostingRegressor(**config.GBM_PARAMS),
        'RandomForest':      RandomForestRegressor(**config.RF_PARAMS),
    }


def build_linear_model() -> dict:
    """
    A plain linear-regression robustness baseline: standardize features then
    Ridge (alpha=1). Scaling + a light L2 penalty keep it well-posed given the
    rainfall features are collinear (e.g. the anomaly is an affine transform of
    the growing-season total). Returns a fresh estimator each call.
    """
    return {'Ridge (linear)': make_pipeline(StandardScaler(),
                                             Ridge(alpha=1.0))}


class ColumnBaseline:
    """
    A 'model' whose prediction is simply the value of one precomputed feature
    column (e.g. last year's rice yield). Implements predict() so the walk-forward
    loop can treat it uniformly. It ignores training data entirely.
    """

    def __init__(self, column: str):
        self.column = column

    def predict_from_frame(self, X_test_frame) -> np.ndarray:
        return X_test_frame[self.column].to_numpy(dtype=float)


def build_baselines() -> dict:
    """Return the two naive baselines, keyed by display name."""
    return {
        'Baseline: persistence (t-1)':   ColumnBaseline(F.PERSISTENCE_COL),
        'Baseline: 3-yr rolling mean':    ColumnBaseline(F.ROLLING_COL),
    }
