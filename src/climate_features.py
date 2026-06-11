"""
climate_features.py
-------------------
Turn the raw climate cache into modeling features.

  • CHIRPS rasters  -> national monthly rainfall over Sierra Leone (rasterio + the
    ADM0 boundary), read straight from the .gz via GDAL's /vsigzip/.
  • NASA POWER JSON -> national monthly T2M and PRECTOTCORR (5-point average).
  • Annual climate features for the yield model (same-year permitted — these are
    exogenous and observed before/during harvest).

Climatology for the rainfall anomaly and the monthly deficit is computed ONLY
from years before the test window (config.CLIM_BASELINE_YEARS = 2000–2017), so
no look-ahead leakage enters the features.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import os
import json
import glob
import calendar
import numpy as np
import pandas as pd

from . import config

CHIRPS_MONTHLY_CSV = os.path.join(config.CLIMATE_DIR, 'chirps_sle_monthly.csv')
POWER_MONTHLY_CSV  = os.path.join(config.CLIMATE_DIR, 'nasa_power_monthly.csv')


# ── CHIRPS -> national monthly rainfall ───────────────────────────────────────
def build_chirps_monthly(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Mean CHIRPS rainfall over Sierra Leone for every month (mm/month)."""
    if os.path.exists(CHIRPS_MONTHLY_CSV) and not force:
        if verbose:
            print(f'[chirps] cached -> {CHIRPS_MONTHLY_CSV}')
        return pd.read_csv(CHIRPS_MONTHLY_CSV)

    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd

    boundary = gpd.read_file(os.path.join(config.CLIMATE_DIR, 'sle_adm0.geojson'))
    geoms = [g.__geo_interface__ for g in boundary.geometry]

    records = []
    for year in config.YEARS:
        for month in range(1, 13):
            gz = os.path.join(config.CHIRPS_DIR,
                              f'chirps-v2.0.{year}.{month:02d}.tif.gz')
            if not os.path.exists(gz):
                records.append({'year': year, 'month': month, 'rain_mm': np.nan})
                continue
            with rasterio.open(f'/vsigzip/{gz}') as src:
                out, _ = mask(src, geoms, crop=True)
                arr = out[0].astype('float64')
                nodata = src.nodata if src.nodata is not None else -9999
                valid = arr[(arr != nodata) & (arr >= 0)]
                records.append({'year': year, 'month': month,
                                'rain_mm': float(valid.mean()) if valid.size else np.nan})
        if verbose:
            print(f'[chirps] processed {year}')

    df = pd.DataFrame(records)
    os.makedirs(config.CLIMATE_DIR, exist_ok=True)
    df.to_csv(CHIRPS_MONTHLY_CSV, index=False)
    if verbose:
        print(f'[chirps] saved {CHIRPS_MONTHLY_CSV} '
              f'({df["rain_mm"].notna().sum()} months)')
    return df


# ── NASA POWER -> national monthly T2M + PRECTOTCORR ──────────────────────────
def build_power_monthly(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Average T2M and PRECTOTCORR across the 5 grid points -> monthly series."""
    if os.path.exists(POWER_MONTHLY_CSV) and not force:
        if verbose:
            print(f'[power] cached -> {POWER_MONTHLY_CSV}')
        return pd.read_csv(POWER_MONTHLY_CSV)

    raw_dir = os.path.join(config.CLIMATE_DIR, 'nasa_power_raw')
    files = sorted(glob.glob(os.path.join(raw_dir, 'point_*.json')))
    frames = []
    for f in files:
        params = json.load(open(f))['properties']['parameter']
        rows = []
        for ym, t2m in params['T2M'].items():
            month = int(ym[4:6])
            if 1 <= month <= 12:          # drop the YYYY13 annual aggregate
                rows.append({'year': int(ym[:4]), 'month': month,
                             'T2M': t2m,
                             'PRECTOTCORR': params['PRECTOTCORR'][ym]})
        frames.append(pd.DataFrame(rows))

    allpts = pd.concat(frames)
    df = (allpts.groupby(['year', 'month'], as_index=False)
                .agg(T2M=('T2M', 'mean'), PRECTOTCORR=('PRECTOTCORR', 'mean')))
    df.to_csv(POWER_MONTHLY_CSV, index=False)
    if verbose:
        print(f'[power] saved {POWER_MONTHLY_CSV} ({len(df)} months, '
              f'{len(files)} points averaged)')
    return df


# ── Cross-check: CHIRPS vs POWER precipitation ────────────────────────────────
def chirps_vs_power_correlation(chirps_m, power_m) -> float:
    """
    Pearson r between CHIRPS monthly totals and NASA POWER PRECTOTCORR converted
    to monthly totals (mm/day x days-in-month). Sanity cross-check only.
    """
    p = power_m.copy()
    p['days'] = p.apply(lambda r: calendar.monthrange(int(r['year']),
                                                      int(r['month']))[1], axis=1)
    p['power_total'] = p['PRECTOTCORR'] * p['days']
    m = chirps_m.merge(p[['year', 'month', 'power_total']], on=['year', 'month'])
    m = m.dropna(subset=['rain_mm', 'power_total'])
    return float(m['rain_mm'].corr(m['power_total']))


# ── Annual climate features ───────────────────────────────────────────────────
def build_climate_annual(verbose: bool = True) -> tuple:
    """
    Assemble year-indexed climate features.

    Returns
    -------
    (feat_df, meta)
        feat_df : year-indexed DataFrame with 7 climate features.
        meta    : dict with the CHIRPS/POWER precip correlation.
    """
    chirps_m = build_chirps_monthly(verbose=verbose)
    power_m  = build_power_monthly(verbose=verbose)
    corr = chirps_vs_power_correlation(chirps_m, power_m)
    if verbose:
        print(f'[crosscheck] CHIRPS vs NASA-POWER monthly precip Pearson r = {corr:.3f}')

    gs, early, peak = (config.GROWING_SEASON_MONTHS,
                       config.EARLY_SEASON_MONTHS, config.PEAK_SEASON_MONTHS)

    def season_sum(df, months, col):
        s = (df[df['month'].isin(months)]
             .groupby('year')[col].sum())
        return s

    rain = chirps_m
    gs_total = season_sum(rain, gs, 'rain_mm').rename('gs_rain_total')
    early_r  = season_sum(rain, early, 'rain_mm').rename('early_rain')
    peak_r   = season_sum(rain, peak, 'rain_mm').rename('peak_rain')

    # Growing-season mean temperature (May–Oct).
    gs_temp = (power_m[power_m['month'].isin(gs)]
               .groupby('year')['T2M'].mean().rename('gs_temp_mean'))

    feat = pd.concat([gs_total, early_r, peak_r, gs_temp], axis=1).sort_index()

    # Climatology strictly from pre-test-window years (2000–2017).
    c0, c1 = config.CLIM_BASELINE_YEARS
    base_mask = (feat.index >= c0) & (feat.index <= c1)
    clim_mean = feat.loc[base_mask, 'gs_rain_total'].mean()
    clim_std  = feat.loc[base_mask, 'gs_rain_total'].std()
    feat['gs_rain_anom'] = (feat['gs_rain_total'] - clim_mean) / clim_std

    # Max 1-month growing-season deficit vs that month's climatological mean.
    monthly_gs = rain[rain['month'].isin(gs)]
    base_clim = (monthly_gs[(monthly_gs['year'] >= c0) & (monthly_gs['year'] <= c1)]
                 .groupby('month')['rain_mm'].mean())
    def max_deficit(grp):
        deficits = base_clim.reindex(grp['month'].values).values - grp['rain_mm'].values
        return float(np.nanmax(deficits))
    feat['max_month_deficit'] = monthly_gs.groupby('year').apply(
        max_deficit, include_groups=False)

    # One lagged climate signal: prior-year growing-season rainfall.
    feat['gs_rain_total_lag1'] = feat['gs_rain_total'].shift(1)

    feat.index.name = 'Year'
    return feat, {'chirps_power_corr': corr,
                  'clim_mean_gs_rain': float(clim_mean),
                  'clim_std_gs_rain': float(clim_std)}


CLIMATE_FEATURE_COLS = [
    'gs_rain_total', 'early_rain', 'peak_rain', 'gs_rain_anom',
    'max_month_deficit', 'gs_temp_mean', 'gs_rain_total_lag1',
]


if __name__ == '__main__':
    f, meta = build_climate_annual()
    print(meta)
    print(f.round(1))
