"""
evaluation.py
-------------
Metrics and performance evaluation for the
Sierra Leone Agricultural ML project.

Provides test-set evaluation, k-fold cross-validation, and feature-importance
extraction for trained tree-ensemble models.

Author: Ibrahim Denis Fofanah
Affiliation: Pace University / RiseAfrica Foundation for STEM and Innovation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_models(models: dict,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> tuple:
    """
    Evaluate all models on the test set.

    Parameters
    ----------
    models : dict
        Mapping of {name: trained model}.
    X_test, y_test : np.ndarray
        Test features and targets.

    Returns
    -------
    tuple : (results_df, predictions)
        results_df : pd.DataFrame sorted by R² (R², RMSE, MAE per model)
        predictions : dict of {name: y_pred}
    """
    results = []
    predictions = {}

    print('\n=== MODEL PERFORMANCE ON TEST SET ===')
    print(f'{"Model":<22} {"R²":>8} {"RMSE":>10} {"MAE":>10}')
    print('-' * 55)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        results.append({'Model': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae})
        print(f'{name:<22} {r2:>8.4f} {rmse:>10.2f} {mae:>10.2f}')

    results_df = pd.DataFrame(results).sort_values('R²', ascending=False).reset_index(drop=True)
    return results_df, predictions


def cross_validate_models(models: dict,
                          X: np.ndarray,
                          y: np.ndarray,
                          cv: int = 5) -> pd.DataFrame:
    """
    Run k-fold cross-validation (R²) for all models.

    Parameters
    ----------
    models : dict
        Mapping of {name: model}.
    X, y : np.ndarray
        Full feature matrix and target vector.
    cv : int
        Number of folds (default: 5).

    Returns
    -------
    pd.DataFrame
        Cross-validation results sorted by mean R².
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = []

    print(f'\n=== {cv}-FOLD CROSS-VALIDATION ===')
    print(f'{"Model":<22} {"Mean R²":>10} {"Std R²":>10}')
    print('-' * 45)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        cv_results.append({
            'Model': name,
            'CV_R2_Mean': scores.mean(),
            'CV_R2_Std':  scores.std(),
            'CV_R2_Min':  scores.min(),
            'CV_R2_Max':  scores.max(),
        })
        print(f'{name:<22} {scores.mean():>10.4f} {scores.std():>10.4f}')

    return pd.DataFrame(cv_results).sort_values('CV_R2_Mean', ascending=False).reset_index(drop=True)


def get_feature_importance(model,
                           feature_names: list,
                           model_name: str = '') -> pd.DataFrame:
    """
    Extract and rank feature importances from a trained tree-ensemble model.

    Parameters
    ----------
    model : fitted model
        Model exposing a ``feature_importances_`` attribute.
    feature_names : list
        Feature names aligned with the model's input columns.
    model_name : str
        Optional label stored alongside the importances.

    Returns
    -------
    pd.DataFrame
        Features ranked by importance, with a percentage share column.
    """
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_,
        'Model': model_name,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    fi_df['Importance_Pct'] = (fi_df['Importance'] / fi_df['Importance'].sum() * 100).round(2)
    return fi_df
