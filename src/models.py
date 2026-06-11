"""
models.py
---------
Model definitions, training, and persistence for the
Sierra Leone Agricultural ML project.

Trains Random Forest, XGBoost, and Gradient Boosting regressors.
Feature preparation lives in feature_engineering.py and metrics in evaluation.py.

Author: Ibrahim Denis Fofanah
Affiliation: Pace University / RiseAfrica Foundation for STEM and Innovation
"""

import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# ── Model Configurations ──────────────────────────────────────────────────────

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 2,
    'random_state': 42,
    'n_jobs': -1,
}

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
}

GBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42,
}


def build_models() -> dict:
    """Instantiate the three (untrained) ensemble regressors."""
    return {
        'Random Forest': RandomForestRegressor(**RF_PARAMS),
        'XGBoost':       XGBRegressor(**XGB_PARAMS),
        'GBM':           GradientBoostingRegressor(**GBM_PARAMS),
    }


def train_all_models(X_train, y_train, models: dict = None) -> dict:
    """
    Train Random Forest, XGBoost, and Gradient Boosting models.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and targets.
    models : dict, optional
        Pre-built {name: estimator} mapping. Defaults to build_models().

    Returns
    -------
    dict
        Mapping of {name: trained model}.
    """
    models = models or build_models()
    trained = {}

    print('\nTraining models...')
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f'  ✓ {name} trained')

    return trained


def save_model(model, model_name: str, output_dir: str) -> None:
    """Save a trained model to ``<output_dir>/<model_name>.pkl``."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pkl")
    joblib.dump(model, filepath)
    print(f"[✓] Model saved: {filepath}")


def load_model(filepath: str):
    """Load a saved model from disk."""
    model = joblib.load(filepath)
    print(f"[✓] Model loaded: {filepath}")
    return model
