"""
evaluation.py
-------------
Walk-forward (expanding-window) validation — the ONLY validation used here.

For each test year y in TEST_YEARS:
    train on every usable year strictly before y   (2000-2017 -> 2018, etc.)
    predict the single held-out year y
Metrics (R², RMSE, MAE) are computed once, across the full set of held-out
predictions (7 points). No random splits, no shuffling, no peeking forward.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from . import config
from . import features as F
from . import modeling


def _metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        'R2':   r2_score(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
    }


def walk_forward(feat: pd.DataFrame,
                 test_years: list = None,
                 verbose: bool = True) -> tuple:
    """
    Run expanding-window validation for all ML models and baselines.

    Returns
    -------
    (results_df, preds_df)
        results_df : metrics per model (R2, RMSE, MAE), sorted best RMSE first.
        preds_df   : one row per test year with the actual and every prediction.
    """
    test_years = test_years or config.TEST_YEARS
    feature_cols = F.get_feature_cols(feat)

    ml_models = modeling.build_models()
    baselines = modeling.build_baselines()
    model_names = list(ml_models) + list(baselines)

    # year -> {actual, <model>: pred, ...}
    rows = {y: {'Year': y, 'Actual': float(feat.loc[y, config.TARGET])}
            for y in test_years}

    for y in test_years:
        train = feat[feat.index < y]
        test  = feat[feat.index == y]
        X_train, y_train = train[feature_cols], train[config.TARGET]
        X_test = test[feature_cols]

        # ML models: fit on the expanding window, predict the one held-out year.
        for name, model in ml_models.items():
            model.fit(X_train.to_numpy(), y_train.to_numpy())
            rows[y][name] = float(model.predict(X_test.to_numpy())[0])

        # Baselines: read the precomputed leakage-free column for the test year.
        for name, base in baselines.items():
            rows[y][name] = float(base.predict_from_frame(X_test)[0])

    preds_df = pd.DataFrame([rows[y] for y in test_years]).set_index('Year')

    # Metrics across all held-out years.
    results = []
    for name in model_names:
        m = _metrics(preds_df['Actual'], preds_df[name])
        results.append({'Model': name, **m})
    results_df = (pd.DataFrame(results)
                  .sort_values('RMSE')
                  .reset_index(drop=True))

    if verbose:
        print(f'\n[evaluation] walk-forward over {test_years[0]}–{test_years[-1]} '
              f'({len(test_years)} held-out years)\n')
        print(results_df.to_string(index=False,
              formatters={'R2': '{:.3f}'.format,
                          'RMSE': '{:.1f}'.format,
                          'MAE': '{:.1f}'.format}))

    return results_df, preds_df


def best_ml_model_name(results_df: pd.DataFrame) -> str:
    """Lowest-RMSE model among the ML models (excludes baselines)."""
    ml = results_df[~results_df['Model'].str.startswith('Baseline')]
    return ml.sort_values('RMSE').iloc[0]['Model']


def walk_forward_featuresets(frame: pd.DataFrame,
                             feature_sets: dict,
                             test_years: list = None,
                             verbose: bool = True) -> tuple:
    """
    Run the identical expanding-window walk-forward for several feature sets,
    so their metrics are directly comparable (same rows, same folds, same
    fixed models). Baselines are computed once (feature-set independent).

    Parameters
    ----------
    frame : pd.DataFrame
        Year-indexed frame containing the target and every column referenced by
        any feature set, plus the baseline columns.
    feature_sets : dict
        {set_name: [feature columns]} — e.g. {'crop-lags': [...], 'climate': [...]}.

    Returns
    -------
    (results_df, preds_df)
        results_df : one row per (Feature Set, Model) with R2/RMSE/MAE, plus the
                     two baselines (Feature Set == '—').
        preds_df   : year-indexed predictions, columns '<Model> [<set>]' and the
                     two baselines, plus 'Actual'.
    """
    test_years = test_years or config.TEST_YEARS
    ml_factory = modeling.build_models
    baselines  = modeling.build_baselines()

    results, preds = [], {'Actual': frame.loc[test_years, config.TARGET]}

    # ── ML models under each feature set ──────────────────────────────────────
    for set_name, cols in feature_sets.items():
        for model_name in ml_factory():
            yhat = {}
            for y in test_years:
                train = frame[frame.index < y]
                test  = frame[frame.index == y]
                model = ml_factory()[model_name]
                model.fit(train[cols].to_numpy(), train[config.TARGET].to_numpy())
                yhat[y] = float(model.predict(test[cols].to_numpy())[0])
            ypred = pd.Series(yhat).reindex(test_years)
            preds[f'{model_name} [{set_name}]'] = ypred
            m = _metrics(preds['Actual'].to_numpy(), ypred.to_numpy())
            results.append({'Feature Set': set_name, 'Model': model_name, **m})

    # ── Baselines (once) ──────────────────────────────────────────────────────
    for name, base in baselines.items():
        ypred = pd.Series(
            {y: float(base.predict_from_frame(frame[frame.index == y])[0])
             for y in test_years}).reindex(test_years)
        preds[name] = ypred
        m = _metrics(preds['Actual'].to_numpy(), ypred.to_numpy())
        results.append({'Feature Set': '—', 'Model': name, **m})

    results_df = (pd.DataFrame(results)
                  .sort_values('RMSE').reset_index(drop=True))
    preds_df = pd.DataFrame(preds)
    preds_df.index.name = 'Year'

    if verbose:
        print(f'\n[evaluation] walk-forward {test_years[0]}–{test_years[-1]}, '
              f'{len(feature_sets)} feature sets x 3 models + 2 baselines\n')
        print(results_df.to_string(index=False,
              formatters={'R2': '{:.3f}'.format, 'RMSE': '{:.1f}'.format,
                          'MAE': '{:.1f}'.format}))

    return results_df, preds_df
