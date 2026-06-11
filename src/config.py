"""
config.py
---------
Central configuration for the Sierra Leone rice-yield prediction pipeline.

Everything that defines *what* the pipeline does — the crop whitelist, the
target, the validation horizon, the shock windows, and the (deliberately
conservative) model settings — lives here so the rest of the code stays generic.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV     = os.path.join(ROOT_DIR, 'data', 'raw',
                           'FAOSTAT_SierraLeone_CropData_2000_2024.csv')
PROC_DIR    = os.path.join(ROOT_DIR, 'data', 'processed')
OUT_DIR     = os.path.join(ROOT_DIR, 'outputs')
FIG_DIR     = os.path.join(OUT_DIR, 'figures')

# ── Crop whitelist ────────────────────────────────────────────────────────────
# The nine crops with complete 2000–2024 coverage across all three elements.
# Yams (3/25 yrs), every aggregate ("…, primary", "… n.e.c.") and every
# processed product (beer, palm oil, sugar, molasses, groundnut oil, tea, …)
# are deliberately EXCLUDED.
CROP_MAP = {
    'Rice':                          'Rice',
    'Cassava, fresh':                'Cassava',
    'Maize (corn)':                  'Maize',
    'Groundnuts, excluding shelled': 'Groundnuts',
    'Oil palm fruit':                'Oil_palm_fruit',
    'Sweet potatoes':                'Sweet_potatoes',
    'Sorghum':                       'Sorghum',
    'Cocoa beans':                   'Cocoa_beans',
    'Plantains and cooking bananas': 'Plantains',
}

ELEMENT_MAP = {
    'Production':      'Production',
    'Area harvested':  'Area_harvested',
    'Yield':           'Yield',
}

# ── Target ────────────────────────────────────────────────────────────────────
TARGET = 'Rice_Yield'   # kg/ha

# ── Walk-forward validation horizon ───────────────────────────────────────────
# Expanding window: for each test year, train on every complete year strictly
# before it. 2018 is the first year with enough history (5-yr rolling mean needs
# 2013–2017), giving 7 out-of-sample predictions: 2018 … 2024.
TEST_YEARS = list(range(2018, 2025))   # [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ── Deterministic shock windows (knowable a-priori, not leakage) ──────────────
EBOLA_YEARS      = (2014, 2016)
COVID_YEARS      = (2020, 2021)
FEED_SALONE_FROM = 2023

# ── Model settings — shallow & regularized on purpose (n ≈ 25 is tiny) ────────
# No grid search. These are fixed, conservative defaults; we do NOT tune them
# to chase a higher score.
RANDOM_STATE = 42

XGB_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'reg_lambda': 1,
    'random_state': RANDOM_STATE,
    'objective': 'reg:squarederror',
    'verbosity': 0,
}

GBM_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'random_state': RANDOM_STATE,
}

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'max_features': 0.5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# ── Climate data (exogenous extension) ────────────────────────────────────────
CLIMATE_DIR = os.path.join(ROOT_DIR, 'data', 'climate')
CHIRPS_DIR  = os.path.join(CLIMATE_DIR, 'chirps')

YEARS = list(range(2000, 2025))   # 2000–2024 inclusive

CHIRPS_BASE_URL   = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/'
GEOBOUNDARIES_URL = 'https://www.geoboundaries.org/api/current/gbOpen/SLE/ADM0/'
NASA_POWER_URL    = 'https://power.larc.nasa.gov/api/temporal/monthly/point'

# ~5-point grid covering Sierra Leone (lat 7.0–9.9, lon −13.3 to −10.3),
# averaged into one national monthly series.
NASA_POINTS = [
    (8.5, -11.8),   # centre
    (7.5, -12.5),   # SW
    (9.5, -12.5),   # NW
    (7.5, -10.8),   # SE
    (9.5, -10.8),   # NE
]

# Growing-season month windows (Sierra Leone single rainy season).
GROWING_SEASON_MONTHS = [5, 6, 7, 8, 9, 10]   # May–October
EARLY_SEASON_MONTHS   = [5, 6]                # May–June
PEAK_SEASON_MONTHS    = [7, 8, 9]             # July–September

# Climatology baseline for the rainfall anomaly — strictly pre-test-window
# (years before 2018) so the anomaly carries no look-ahead leakage.
CLIM_BASELINE_YEARS = (2000, 2017)
