# Can Machine Learning Forecast Rice Yields in Data-Constrained Settings?
## Satellite Climate Data, National Crop Statistics, and Lessons from Sierra Leone

**Author:** Ibrahim Denis Fofanah
**Affiliation:** Seidenberg School of Computer Science & Information Systems, Pace University, New York · RiseAfrica Foundation for STEM and Innovation, Sierra Leone
**Email:** IF57774N@pace.edu

---

## Overview

A reproducible pipeline that asks whether **rice yield (kg/ha)** in Sierra Leone
can be forecast from publicly available data, under **strict anti-leakage
discipline** and **walk-forward validation only**, benchmarked against naive
baselines.

The answer comes in two parts, both reported plainly:

1. **Crop statistics alone: no.** Trained on 25 years of FAOSTAT crop data
   (2000–2024), no ML model beats a simple persistence baseline (predict this
   year = last year).
2. **Adding free satellite climate data: yes.** With CHIRPS rainfall and NASA
   POWER temperature aggregated to national growing-season features, a
   climate-only XGBoost cuts forecast error by **one third** vs. persistence
   (RMSE 284 vs. 428 kg/ha) — a gain that holds for a linear model too and is
   robust to dropping the anomalous 2018 season.

The dominant predictor is **May–June (planting-season) rainfall**, observable in
CHIRPS months before harvest — the basis for a near-zero-cost early-warning
capability for Sierra Leone's Ministry of Agriculture and Food Security.

Honest boundaries, documented rather than hidden: **no model anticipated the
2018 yield collapse** (institutional, not climatic, in origin), and the record
yields of 2020–2022 occurred in *below-average* rainfall years, consistent with
input-driven policy gains. Full write-ups:
[outputs/model_report.md](outputs/model_report.md) (v1) and
[outputs/model_report_v2.md](outputs/model_report_v2.md) (v2 + robustness).

> An earlier version of this pipeline — same-year features, FAOSTAT aggregates,
> random 70/30 split — produced an apparent R² of 0.96. Every component of that
> number was leakage; it is preserved in git history (tag `pre-rebuild`) as a
> documented case study in how flawed validation designs flatter small-sample
> agricultural ML.

---

## Project structure

```
sierraleone-agri-ml/
├── data/
│   ├── raw/                          # FAOSTAT_SierraLeone_CropData_2000_2024.csv (do not modify)
│   ├── processed/                    # wide_crop_matrix.csv, model_frame.csv (generated)
│   └── climate/                      # derived monthly CSVs + SLE boundary (CHIRPS raster cache gitignored)
├── src/
│   ├── config.py                     # crop whitelist, target, horizon, fixed model params
│   ├── data_prep.py                  # load long CSV → wide year × (crop_element) matrix
│   ├── features.py                   # no-leakage crop/autoregressive features
│   ├── download_climate.py           # CHIRPS rasters + geoBoundaries + NASA POWER API
│   ├── climate_features.py           # growing-season rainfall/temperature features
│   ├── modeling.py                   # ML model factory + naive baselines + Ridge check
│   ├── evaluation.py                 # expanding-window walk-forward validation
│   ├── visualize.py                  # all figures
│   ├── run_pipeline.py               # v1: crop-lags experiment
│   └── run_pipeline_v2.py            # v2: climate / crop+climate / robustness
├── outputs/
│   ├── figures/                      # actual_vs_predicted, shap_summary, rainfall_vs_yield, …
│   ├── results_table.csv             # v1 results
│   ├── results_climate_comparison.csv# v2 combined results (all feature sets)
│   ├── per_year_errors.csv           # per-year actual/predicted/|error|, all models
│   ├── walk_forward_predictions.csv
│   ├── model_report.md               # v1 honest write-up
│   └── model_report_v2.md            # v2 honest write-up + robustness checks
├── requirements.txt
└── README.md
```

---

## Data sources (all free, no registration)

| Source | Content | Access |
|---|---|---|
| FAOSTAT (QCL) | Crop area/yield/production, Sierra Leone 2000–2024 | <https://www.fao.org/faostat/en/#data/QCL> |
| CHIRPS v2.0 | Monthly precipitation rasters, Africa | <https://www.chc.ucsb.edu/data/chirps> |
| NASA POWER | Monthly T2M temperature (AG community) | <https://power.larc.nasa.gov> |
| geoBoundaries | Sierra Leone ADM0 boundary | <https://www.geoboundaries.org> |

CHIRPS and NASA POWER monthly precipitation cross-check at **r = 0.73**.

---

## Data preparation

- FAOSTAT long format is pivoted to **one row per year**, columns =
  `<Crop>_<Element>` (e.g. `Rice_Yield`, `Cassava_Production`).
- **Only nine complete crops** (full 25-year coverage on all three elements):
  Rice, Cassava, Maize, Groundnuts, Oil palm fruit, Sweet potatoes, Sorghum,
  Cocoa beans, Plantains.
- **Excluded:** Yams (3/25 years present), all FAOSTAT aggregates
  (`Cereals, primary` is 87–93% rice; r = 0.998 with the target — pure
  leakage), and all processed products.
- **Target:** `Rice_Yield` (kg/ha).

## Feature engineering — strict no-leakage rules

Every *crop-derived* predictor for year *t* must be knowable **before year *t*
begins**:

1. **No same-year crop features.** Cross-crop signals enter only as **lag-1**.
2. **Rice-yield lags 1–3** plus 3- and 5-year rolling means computed on past
   years only (`shift(1)` *before* `rolling()`).
3. **No rice production-per-hectare feature** — it reconstructs the target.
4. Calendar/shock features: year trend, Ebola (2014–16), COVID (2020–21),
   Feed Salone (2023+) dummies.

**Climate features are exogenous** (measured independently of yield, available
in-season, before harvest) and therefore permitted at year *t*: growing-season
total rainfall (May–Oct), early-season rainfall (May–Jun), peak-season rainfall
(Jul–Sep), standardized rainfall anomaly (climatology from 2000–2017 only),
max one-month deficit, mean growing-season temperature, lag-1 growing-season
rainfall.

A runtime assertion fails the build if any bare same-year crop column reaches
the predictor matrix.

## Validation — walk-forward only

Expanding window, **no random splits anywhere**: train 2000→2017 ⇒ predict
2018; train 2000→2018 ⇒ predict 2019; … through 2024 = **7 out-of-sample
forecasts**, each made with prior information only.

## Models

Conservative, **fixed** settings (n ≈ 25 — no grid search):

| Model | Settings |
|---|---|
| XGBoost | `max_depth=2, lr=0.05, n_estimators=100, subsample=0.8, reg_lambda=1` |
| GradientBoosting | `max_depth=2, lr=0.05, n_estimators=100, subsample=0.8` |
| RandomForest | `max_depth=3, n_estimators=300, min_samples_leaf=2, max_features=0.5` |
| Ridge (robustness) | linear check on climate features, identical protocol |
| Baseline — persistence | predict `t` = rice yield at `t−1` |
| Baseline — rolling mean | predict `t` = mean of past 3 years |

## Results (walk-forward, 2018–2024, 7 held-out years)

| Feature set | Model | R² | RMSE | MAE |
|---|---|---:|---:|---:|
| **Climate only** | **XGBoost** | **0.542** | **284.1** | **216.3** |
| Climate only | GradientBoosting | 0.465 | 306.8 | 242.7 |
| Climate only | Ridge (linear) | 0.222 | 370.2 | 304.0 |
| Climate only | RandomForest | 0.162 | 384.2 | 304.6 |
| — | Baseline: persistence (t−1) | −0.039 | 427.8 | 341.3 |
| Crop + climate | RandomForest | −0.097 | 439.5 | 420.5 |
| Crop + climate | GradientBoosting | −0.181 | 456.1 | 427.2 |
| Crop + climate | XGBoost | −0.191 | 458.0 | 427.0 |
| Crop lags only | RandomForest | −0.247 | 468.5 | 454.7 |
| — | Baseline: 3-yr rolling mean | −0.565 | 524.9 | 511.4 |
| Crop lags only | XGBoost | −0.591 | 529.4 | 497.2 |
| Crop lags only | GradientBoosting | −0.630 | 535.8 | 496.7 |

Key reading:

- **Only climate-only configurations beat persistence.** Crop lags — alone or
  added to climate — make every model worse (overfitting at n ≈ 15/fold:
  *more features hurt*).
- The climate advantage is earned at **turning points** (2019: XGB error 171
  vs. persistence 789; 2023: 33 vs. 601), where persistence lags by
  construction. Excluding 2018 entirely *widens* the gap (182 vs. 438).
- **2018 is the honest failure:** yield collapsed to 786 kg/ha; the climate
  model predicted 1,391. The collapse was institutional (post-Ebola
  disorganization, fertilizer access, area contraction), invisible to rainfall
  data.
- R² over 7 points is high-variance; read RMSE/MAE vs. persistence as the
  primary metrics. Per-year detail: `outputs/per_year_errors.csv`.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_climate   # one-time: fetch CHIRPS rasters + NASA POWER + boundary
python -m src.run_pipeline       # v1: crop-lags experiment
python -m src.run_pipeline_v2    # v2: all feature sets + robustness (uses the cached climate data)
```

> macOS note: XGBoost needs the OpenMP runtime — `brew install libomp`.
> The `download_climate` step fetches ~300 CHIRPS monthly rasters (~1.3 GB,
> cached under `data/climate/` and gitignored); later runs reuse the cache.

## Citation

```
Fofanah, I.D. (2026, under review). Can Machine Learning Forecast Rice Yields
in Data-Constrained Settings? Satellite Climate Data, National Crop Statistics,
and Lessons from Sierra Leone.
```

(An arXiv link will be added upon preprint posting.)

## License

MIT.

---

*"African data scientists should train on African data to solve African problems."*
— Ibrahim Denis Fofanah, RiseAfrica Foundation
