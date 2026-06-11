# Rice Yield Prediction in Sierra Leone — A Leakage-Free Modeling Pipeline

**Author:** Ibrahim Denis Fofanah
**Affiliation:** Seidenberg School of Computer Science & Information Systems, Pace University, New York · RiseAfrica Foundation for STEM and Innovation, Sierra Leone
**Email:** IF57774N@pace.edu

---

## Overview

A small, honest, reproducible pipeline that predicts **rice yield (kg/ha)** in
Sierra Leone from 25 years of FAOSTAT crop data (2000–2024), benchmarked against
naive baselines under **strict anti-leakage discipline** and **walk-forward
validation only**.

The headline finding is reported plainly, not flattered: on this short national
series, **no machine-learning model beats a simple persistence baseline**
(predict this year = last year). See [outputs/model_report.md](outputs/model_report.md).

---

## Project structure

```
sierraleone-agri-ml/
├── data/
│   ├── raw/                         # FAOSTAT_SierraLeone_CropData_2000_2024.csv (do not modify)
│   └── processed/                   # wide_crop_matrix.csv, model_frame.csv (generated)
├── src/
│   ├── config.py                    # crop whitelist, target, horizon, model params
│   ├── data_prep.py                 # load long CSV → wide year × (crop_element) matrix
│   ├── features.py                  # no-leakage feature engineering
│   ├── modeling.py                  # ML model factory + naive baselines
│   ├── evaluation.py                # expanding-window walk-forward validation
│   ├── visualize.py                 # actual-vs-pred, feature importance, SHAP
│   └── run_pipeline.py              # orchestrator (entry point)
├── outputs/
│   ├── figures/                     # actual_vs_predicted / feature_importance / shap_summary .png
│   ├── results_table.csv            # R², RMSE, MAE for all models + baselines
│   ├── walk_forward_predictions.csv # per-year actual vs predictions
│   └── model_report.md              # honest written summary of design + results
├── requirements.txt
└── README.md
```

---

## Data preparation

- FAOSTAT long format (`Item`, `Element`, `Year`, `Value`, …) is pivoted to **one
  row per year**, columns = `<Crop>_<Element>` (e.g. `Rice_Yield`,
  `Cassava_Production`).
- **Only nine complete crops** (full 25-year coverage on all three elements):
  Rice, Cassava (`Cassava, fresh`), Maize (`Maize (corn)`), Groundnuts
  (`Groundnuts, excluding shelled`), Oil palm fruit, Sweet potatoes, Sorghum,
  Cocoa beans, Plantains (`Plantains and cooking bananas`).
- **Excluded:** Yams (3/25 years), all aggregates (`Cereals, primary`,
  `Fruit Primary`, anything `… n.e.c.`), and all processed products (beer, palm
  oil, molasses, sugar, groundnut oil, tea, …).
- **Target:** `Rice_Yield` (kg/ha).

## Feature engineering — strict no-leakage rules

Every predictor for year *t* must be knowable **before year *t* begins**:

1. **No same-year features of any kind.** Cross-crop signals enter only as their
   **lag-1** value (e.g. `Sweet_potatoes_Production_lag1`).
2. **Rice-yield lags 1, 2, 3**, plus **3- and 5-year rolling means computed on
   past years only** (`shift(1)` *before* `rolling()`).
3. **No rice production-per-hectare feature** — it reconstructs the target.
4. Deterministic calendar features: linear **year trend**, **Ebola** (2014–16),
   **COVID** (2020–21), **Feed Salone** (2023+) dummies.

A runtime assertion fails the build if any bare same-year crop column ever
appears among the predictors.

## Validation — walk-forward only

Expanding window, **no random splits**: train 2000→2017 ⇒ predict 2018; train
2000→2018 ⇒ predict 2019; … through 2024. That yields **7 out-of-sample
predictions**, across which R², RMSE, and MAE are computed.

## Models

Conservative, **fixed** settings (n ≈ 25 is tiny — no grid search):

| Model | Settings |
|---|---|
| XGBoost | `max_depth=2, lr=0.05, n_estimators=100, subsample=0.8, reg_lambda=1` |
| GradientBoosting | `max_depth=2, lr=0.05, n_estimators=100, subsample=0.8` |
| RandomForest | `max_depth=3, n_estimators=300, min_samples_leaf=2, max_features=0.5` |
| Baseline — persistence | predict `t` = rice yield at `t-1` |
| Baseline — rolling mean | predict `t` = mean of past 3 years |

## Results (walk-forward, 2018–2024)

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| **Baseline: persistence (t-1)** | **-0.039** | **427.8** | **341.3** |
| RandomForest | -0.247 | 468.5 | 454.7 |
| Baseline: 3-yr rolling mean | -0.565 | 524.9 | 511.4 |
| XGBoost | -0.591 | 529.4 | 497.2 |
| GradientBoosting | -0.630 | 535.8 | 496.7 |

**No ML model beats persistence.** With only ~13–18 training rows, the tree
ensembles cannot out-predict the strong year-to-year autocorrelation that "last
year's yield" already captures. R² across 7 points is high-variance (and can be
negative); RMSE/MAE in kg/ha are the more interpretable metrics. Models were not
tuned and features were not added to chase a higher score.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_pipeline      # prints the results table; writes outputs/
```

> macOS note: XGBoost needs the OpenMP runtime — `brew install libomp`.

## Data source

FAOSTAT — Crops and Livestock Products, Sierra Leone (2000–2024).
<https://www.fao.org/faostat/en/#data/QCL>

## Citation

```
Fofanah, I.D. (2026, in preparation). Machine Learning Approaches to Crop Yield
Prediction in Smallholder Farming Communities: Evidence from Sierra Leone.
```

## License

MIT.

---

*"African data scientists should train on African data to solve African problems."*
— Ibrahim Denis Fofanah, RiseAfrica Foundation
