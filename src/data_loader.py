"""
data_loader.py
--------------
Data loading and cleaning functions for the Sierra Leone Agricultural ML project.

Author: Ibrahim Denis Fofanah
Affiliation: Pace University / RiseAfrica Foundation for STEM and Innovation
"""

import pandas as pd
import numpy as np
import os


# ── Key crops of interest for Sierra Leone ────────────────────────────────────
KEY_CROPS = [
    'Rice',
    'Cassava, fresh',
    'Maize (corn)',
    'Groundnuts, excluding shelled',
    'Oil palm fruit',
    'Sweet potatoes',
    'Sorghum',
    'Yams',
    'Cocoa beans',
    'Plantains and cooking bananas',
]

STAPLE_CROPS = ['Rice', 'Cassava, fresh', 'Maize (corn)', 'Sweet potatoes']


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw FAOSTAT CSV data.

    Parameters
    ----------
    filepath : str
        Path to the raw FAOSTAT CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"[✓] Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Years: {df['Year'].min()} – {df['Year'].max()}")
    print(f"    Unique items: {df['Item'].nunique()}")
    print(f"    Elements: {df['Element'].unique().tolist()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the raw FAOSTAT dataframe.

    Steps:
    - Rename columns for readability
    - Drop unnecessary columns
    - Filter to Sierra Leone only (safety check)
    - Handle missing values
    - Standardize crop names

    Parameters
    ----------
    df : pd.DataFrame
        Raw FAOSTAT dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    # Rename columns
    df = df.rename(columns={
        'Area': 'Country',
        'Item': 'Crop',
        'Element': 'Metric',
        'Year': 'Year',
        'Value': 'Value',
        'Unit': 'Unit',
        'Flag Description': 'DataQuality',
    })

    # Keep only relevant columns
    keep_cols = ['Country', 'Crop', 'Metric', 'Year', 'Value', 'Unit', 'DataQuality']
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Safety check: Sierra Leone only
    df = df[df['Country'] == 'Sierra Leone'].copy()

    # Drop rows with missing values
    missing_before = df['Value'].isna().sum()
    df = df.dropna(subset=['Value'])
    if missing_before > 0:
        print(f"[!] Dropped {missing_before} rows with missing values")

    # Standardize crop names (shorter aliases)
    crop_rename = {
        'Cassava, fresh': 'Cassava',
        'Maize (corn)': 'Maize',
        'Groundnuts, excluding shelled': 'Groundnuts',
        'Sweet potatoes': 'Sweet Potato',
        'Plantains and cooking bananas': 'Plantains',
    }
    df['Crop'] = df['Crop'].replace(crop_rename)

    print(f"[✓] Cleaned data: {df.shape[0]:,} rows")
    return df


def pivot_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pivot data to wide format: rows = years, columns = crops.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned long-format dataframe.
    metric : str
        One of 'Yield', 'Production', 'Area harvested'.

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe with Year as index.
    """
    df_metric = df[df['Metric'] == metric].copy()
    wide = df_metric.pivot_table(
        index='Year',
        columns='Crop',
        values='Value',
        aggfunc='first'
    )
    wide = wide.sort_index()
    print(f"[✓] Pivoted '{metric}': {wide.shape[0]} years × {wide.shape[1]} crops")
    return wide


def get_crop_timeseries(df: pd.DataFrame, crop: str, metric: str = 'Yield') -> pd.Series:
    """
    Extract a single crop time series.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned long-format dataframe.
    crop : str
        Crop name (use cleaned names e.g. 'Rice', 'Cassava').
    metric : str
        'Yield', 'Production', or 'Area harvested'.

    Returns
    -------
    pd.Series
        Time series indexed by Year.
    """
    subset = df[(df['Crop'] == crop) & (df['Metric'] == metric)].copy()
    subset = subset.set_index('Year')['Value'].sort_index()
    return subset


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the main analysis dataset:
    - Wide format with yield, production, and area for key crops
    - Year-on-year change features
    - Rolling averages

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned long-format dataframe.

    Returns
    -------
    pd.DataFrame
        Analysis-ready dataset.
    """
    yield_wide = pivot_wide(df, 'Yield')
    prod_wide = pivot_wide(df, 'Production')
    area_wide = pivot_wide(df, 'Area harvested')

    # Rename columns with metric suffix
    yield_wide.columns = [f"{c}_Yield" for c in yield_wide.columns]
    prod_wide.columns  = [f"{c}_Production" for c in prod_wide.columns]
    area_wide.columns  = [f"{c}_Area" for c in area_wide.columns]

    # Merge on Year
    analysis = pd.concat([yield_wide, prod_wide, area_wide], axis=1)
    analysis.index.name = 'Year'
    analysis = analysis.reset_index()

    print(f"[✓] Analysis dataset: {analysis.shape[0]} rows × {analysis.shape[1]} columns")
    return analysis


def save_processed(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """
    Save processed dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to save.
    output_dir : str
        Output directory path.
    filename : str
        Output filename (e.g. 'analysis_dataset.csv').
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[✓] Saved: {filepath}")
