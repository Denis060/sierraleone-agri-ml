"""
data_prep.py
------------
Load the raw FAOSTAT long-format CSV and reshape it into a clean wide matrix:
one row per year, one column per (crop × element).

Only the nine complete crops and three core elements are kept; everything else
(aggregates, processed products, sparse crops) is dropped here, once.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import os
import pandas as pd

from . import config


def load_and_pivot(raw_csv: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load the raw CSV and return a wide, year-indexed dataframe.

    Returns
    -------
    pd.DataFrame
        Index = Year (int, ascending). Columns = '<Crop>_<Element>'
        (e.g. 'Rice_Yield', 'Cassava_Production', 'Maize_Area_harvested').
    """
    raw_csv = raw_csv or config.RAW_CSV
    df = pd.read_csv(raw_csv, encoding='utf-8-sig')

    # Keep only the whitelisted crops and the three core elements.
    df = df[df['Item'].isin(config.CROP_MAP) & df['Element'].isin(config.ELEMENT_MAP)].copy()

    # Standardize names.
    df['Crop']    = df['Item'].map(config.CROP_MAP)
    df['Element'] = df['Element'].map(config.ELEMENT_MAP)
    df['Year']    = df['Year'].astype(int)
    df['Value']   = pd.to_numeric(df['Value'], errors='coerce')

    # Wide pivot: one row per year, one column per crop × element.
    wide = df.pivot_table(index='Year', columns=['Crop', 'Element'],
                          values='Value', aggfunc='first')
    wide.columns = [f'{crop}_{element}' for crop, element in wide.columns]
    wide = wide.sort_index()

    if verbose:
        n_missing = int(wide.isna().sum().sum())
        print(f'[data_prep] {wide.shape[0]} years x {wide.shape[1]} columns '
              f'({len(config.CROP_MAP)} crops x {len(config.ELEMENT_MAP)} elements)')
        print(f'[data_prep] year range {wide.index.min()}–{wide.index.max()} | '
              f'missing cells: {n_missing}')
        if config.TARGET not in wide.columns:
            raise ValueError(f'Target {config.TARGET} not found after pivot.')

    return wide


def save_wide(wide: pd.DataFrame) -> str:
    """Persist the wide matrix to data/processed/ for transparency."""
    os.makedirs(config.PROC_DIR, exist_ok=True)
    path = os.path.join(config.PROC_DIR, 'wide_crop_matrix.csv')
    wide.to_csv(path)
    print(f'[data_prep] saved {path}')
    return path


if __name__ == '__main__':
    w = load_and_pivot()
    save_wide(w)
    print(w.head())
