# Machine Learning Approaches to Crop Yield Prediction and Post-Harvest Loss Reduction in Smallholder Farming Communities: Evidence from Sierra Leone

**Author:** Ibrahim Denis Fofanah  
**Affiliation 1:** Seidenberg School of Computer Science & Information Systems, Pace University, New York, USA  
**Affiliation 2:** RiseAfrica Foundation for STEM and Innovation, Sierra Leone  
**Email:** if74741p@pace.edu  

---

## Overview

This repository contains the full research codebase, data pipeline, and analysis for the paper:

> *"Machine Learning Approaches to Crop Yield Prediction and Post-Harvest Loss Reduction in Smallholder Farming Communities: Evidence from Sierra Leone"*

This study presents the **first integrated machine learning framework** designed specifically to address crop yield prediction and post-harvest loss risk scoring among smallholder farming communities in Sierra Leone, using 25 years of FAOSTAT agricultural data (2000–2024).

Findings are situated within the context of Sierra Leone's **Feed Salone Strategy 2023–2030** and contribute directly to the national agenda of doubling rice production and achieving food sovereignty by 2030.

---

## Project Structure

```
sierraleone-agri-ml/
│
├── data/
│   ├── raw/                          # Original FAOSTAT data (do not modify)
│   └── processed/                    # Cleaned, engineered datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory Data Analysis (EDA)
│   ├── 02_feature_engineering.ipynb  # Feature construction and preprocessing
│   ├── 03_model_training.ipynb       # RF, XGBoost, GBM training & comparison
│   ├── 04_shap_analysis.ipynb        # SHAP explainability analysis
│   └── 05_visualizations.ipynb       # All paper figures and charts
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Data loading and cleaning functions
│   ├── feature_engineering.py        # Feature construction functions
│   ├── models.py                     # Model training and evaluation functions
│   ├── evaluation.py                 # Metrics and performance evaluation
│   └── visualization.py             # Plotting and figure generation
│
├── outputs/
│   ├── figures/                      # All generated figures (PNG, PDF)
│   ├── models/                       # Saved trained models (.pkl)
│   └── reports/                      # Performance reports (CSV, JSON)
│
├── docs/
│   └── paper/                        # LaTeX paper source files
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---

## Data Source

**Primary Dataset:** FAOSTAT Crops and Livestock Products — Sierra Leone  
**Coverage:** 2000–2024 (25 years)  
**Elements:** Area Harvested (ha), Yield (kg/ha), Production Quantity (tonnes)  
**Crops:** Rice, Cassava, Maize, Groundnuts, Oil Palm, Sweet Potato, Sorghum, Yams, Cocoa, Plantains, and more  
**Source:** Food and Agriculture Organization of the United Nations  
**URL:** https://www.fao.org/faostat/en/#data/QCL  

---

## Models

Three ensemble machine learning models are trained and compared:

| Model | Type | Key Strength |
|---|---|---|
| Random Forest (RF) | Ensemble — Bagging | Robust, handles mixed features, interpretable |
| XGBoost | Ensemble — Boosting | High accuracy, efficient, handles missing data |
| Gradient Boosting (GBM) | Ensemble — Boosting | Flexible loss functions, strong on structured data |

**Evaluation Metrics:** R², RMSE, MAE  
**Explainability:** SHAP (SHapley Additive exPlanations)

---

## Research Questions

1. Which ML algorithm most accurately predicts crop yield outcomes for Sierra Leone's key staple crops (2000–2024)?
2. What are the primary predictors of yield variability and production decline in Sierra Leone's agricultural sector?
3. How can ML-derived insights directly inform the implementation of Sierra Leone's Feed Salone Strategy 2023–2030?

---

## How to Run

### Option 1 — Google Colab (Recommended)
Open any notebook in the `notebooks/` folder directly in Google Colab. No installation required.

### Option 2 — Local Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sierraleone-agri-ml.git
cd sierraleone-agri-ml

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Citation

If you use this code or data in your research, please cite:

```
Fofanah, I.D. (2026). Machine Learning Approaches to Crop Yield Prediction 
and Post-Harvest Loss Reduction in Smallholder Farming Communities: 
Evidence from Sierra Leone. arXiv preprint.
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

Data sourced from FAOSTAT (Food and Agriculture Organization of the United Nations).  
This research is conducted in support of Sierra Leone's Feed Salone Strategy 2023–2030  
and the RiseAfrica Foundation's mission to train African data scientists on African datasets.

---

*"African data scientists should train on African data to solve African problems."*  
— Ibrahim Denis Fofanah, RiseAfrica Foundation
