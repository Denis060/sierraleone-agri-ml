"""
feature_engineering.py
----------------------
Feature construction and preprocessing for the
Sierra Leone Agricultural ML project.

Builds lag, rolling, shock, and trend features from the analysis dataset,
and prepares the (X, y) matrices used for model training.

Author: Ibrahim Denis Fofanah
Affiliation: Pace University / RiseAfrica Foundation for STEM and Innovation
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal, lag, rolling, shock, and efficiency features for ML modeling.

    Features added (when source columns are present):
    - Rice yield lags (1–3), 3- and 5-year rolling means, year-on-year % change
    - Cassava yield lag and year-on-year % change
    - Rice area lag and percentage change
    - Rice production efficiency (production per hectare)
    - Year trend, decade, and shock-period indicators
      (Ebola 2014–2016, COVID 2020–2021, Feed Salone >= 2023, Post-2010)
    - Crop diversity index (count of crops with a recorded yield each year)

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset from data_loader.build_analysis_dataset().

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features added.
    """
    ml_df = df.copy().sort_values('Year').reset_index(drop=True)

    # ── Rice yield features (primary target) ─────────────────────────────────
    if 'Rice_Yield' in ml_df.columns:
        ml_df['Rice_Yield_Lag1']     = ml_df['Rice_Yield'].shift(1)
        ml_df['Rice_Yield_Lag2']     = ml_df['Rice_Yield'].shift(2)
        ml_df['Rice_Yield_Lag3']     = ml_df['Rice_Yield'].shift(3)
        ml_df['Rice_Yield_Rolling3'] = ml_df['Rice_Yield'].rolling(3).mean()
        ml_df['Rice_Yield_Rolling5'] = ml_df['Rice_Yield'].rolling(5).mean()
        ml_df['Rice_Yield_YoY']      = ml_df['Rice_Yield'].pct_change() * 100

    # ── Cassava yield features ───────────────────────────────────────────────
    if 'Cassava_Yield' in ml_df.columns:
        ml_df['Cassava_Yield_Lag1'] = ml_df['Cassava_Yield'].shift(1)
        ml_df['Cassava_YoY']        = ml_df['Cassava_Yield'].pct_change() * 100

    # ── Rice area features ───────────────────────────────────────────────────
    if 'Rice_Area harvested' in ml_df.columns:
        ml_df = ml_df.rename(columns={'Rice_Area harvested': 'Rice_Area'})
    if 'Rice_Area' in ml_df.columns:
        ml_df['Rice_Area_Lag1']   = ml_df['Rice_Area'].shift(1)
        ml_df['Rice_Area_Change'] = ml_df['Rice_Area'].pct_change() * 100

    # ── Production efficiency ────────────────────────────────────────────────
    if 'Rice_Production' in ml_df.columns and 'Rice_Area' in ml_df.columns:
        ml_df['Rice_Prod_Per_Ha'] = (
            ml_df['Rice_Production'] / ml_df['Rice_Area'].replace(0, np.nan)
        )

    # ── Temporal & shock indicators ──────────────────────────────────────────
    ml_df['Year_Trend']   = ml_df['Year'] - ml_df['Year'].min()
    ml_df['Decade']       = (ml_df['Year'] // 10) * 10
    ml_df['Ebola_Period'] = ml_df['Year'].between(2014, 2016).astype(int)
    ml_df['COVID_Period'] = ml_df['Year'].between(2020, 2021).astype(int)
    ml_df['FeedSalone']   = (ml_df['Year'] >= 2023).astype(int)
    ml_df['Post2010']     = (ml_df['Year'] >= 2010).astype(int)

    # ── Multi-crop diversity index ───────────────────────────────────────────
    yield_crop_cols = [c for c in ml_df.columns
                       if c.endswith('_Yield') and 'Lag' not in c
                       and 'Rolling' not in c and 'YoY' not in c]
    ml_df['Crop_Diversity_Yield'] = ml_df[yield_crop_cols].notna().sum(axis=1)

    n_new = ml_df.shape[1] - df.shape[1]
    print(f"[✓] Feature engineering: {ml_df.shape[1]} columns ({n_new} new features)")
    return ml_df


def prepare_features(df: pd.DataFrame,
                     target_col: str,
                     drop_cols: list = None) -> tuple:
    """
    Prepare the feature matrix X and target vector y for modeling.

    Steps:
    - Drop the target and non-feature columns (e.g. 'Year')
    - Keep numeric columns only
    - Drop rows where the target is missing
    - Fill remaining feature NaNs with the column median

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset (after add_temporal_features).
    target_col : str
        Name of the target column (e.g. 'Rice_Yield').
    drop_cols : list, optional
        Additional columns to exclude from the features.

    Returns
    -------
    tuple : (X, y, feature_names)
        X : np.ndarray feature matrix
        y : np.ndarray target vector
        feature_names : list of str
    """
    drop = ['Year'] + (drop_cols or [])

    X = df.drop(columns=[target_col] + [c for c in drop if c in df.columns],
                errors='ignore')
    X = X.select_dtypes(include=[np.number])

    mask = df[target_col].notna()
    X = X[mask].copy()
    y = df.loc[mask, target_col].copy()

    X = X.fillna(X.median())

    feature_names = X.columns.tolist()
    print(f"[✓] Features: {len(feature_names)} | Samples: {len(y)}")
    print(f"    Target: {target_col} | Range: {y.min():.1f} – {y.max():.1f}")
    return X.values, y.values, feature_names
