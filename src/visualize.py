"""
visualize.py
------------
Figures for the rice-yield pipeline:
  • actual vs. walk-forward predicted yield by year (all models)
  • feature importance for the best model
  • SHAP summary for the best model

All figures are saved as PNG (300 dpi) to outputs/figures/.

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from . import config

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})

PRIMARY = '#1A6B47'


def _save(fig, name):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'[viz] saved {path}')


def plot_actual_vs_pred(preds_df):
    """Line plot: observed rice yield vs each model's walk-forward predictions."""
    fig, ax = plt.subplots(figsize=(11, 6))
    years = preds_df.index

    ax.plot(years, preds_df['Actual'], color='black', lw=2.8, marker='o',
            ms=7, label='Actual', zorder=5)

    model_cols = [c for c in preds_df.columns if c != 'Actual']
    cmap = plt.cm.tab10(np.linspace(0, 1, len(model_cols)))
    for col, color in zip(model_cols, cmap):
        style = '--' if col.startswith('Baseline') else '-'
        ax.plot(years, preds_df[col], style, color=color, lw=1.8,
                marker='s', ms=4, alpha=0.85, label=col)

    ax.set_title('Rice Yield — Actual vs. Walk-Forward Predictions\n'
                 'Sierra Leone, expanding-window out-of-sample (2018–2024)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rice Yield (kg/ha)')
    ax.set_xticks(years)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    _save(fig, 'actual_vs_predicted.png')
    return fig


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Horizontal bar chart of the best model's top-N feature importances."""
    imp = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals  = imp[order][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names, vals, color=PRIMARY, edgecolor='white')
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f'{v:.3f}', va='center', fontsize=8.5)
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}\n'
                 '(fit on full data, leakage-free features)')
    ax.set_xlabel('Importance')
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    _save(fig, 'feature_importance.png')
    return fig


def plot_rainfall_vs_yield(years, gs_rain, rice_yield, fname='rainfall_vs_yield.png'):
    """Dual-axis: growing-season rainfall vs rice yield by year (e.g. is 2018's
    yield crash aligned with a rainfall deficit?)."""
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.bar(years, gs_rain, color='#2980B9', alpha=0.45, label='Growing-season rainfall')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Growing-season rainfall, May–Oct (mm)', color='#1F5F8B')
    ax1.tick_params(axis='y', labelcolor='#1F5F8B')

    ax2 = ax1.twinx()
    ax2.plot(years, rice_yield, color=PRIMARY, lw=2.6, marker='o', ms=6,
             label='Rice yield')
    ax2.set_ylabel('Rice yield (kg/ha)', color=PRIMARY)
    ax2.tick_params(axis='y', labelcolor=PRIMARY)
    ax2.grid(False)

    if 2018 in list(years):
        ax1.axvline(2018, color='#C0392B', ls='--', lw=1.3, alpha=0.7)
        ax1.text(2018, ax1.get_ylim()[1] * 0.96, ' 2018', color='#C0392B',
                 fontsize=9, va='top')

    ax1.set_title('Growing-Season Rainfall vs. Rice Yield — Sierra Leone (2000–2024)')
    ax1.set_xticks(list(years)[::2])
    fig.tight_layout()
    _save(fig, fname)
    return fig


def plot_shap_summary(model, X, feature_names, model_name):
    """SHAP summary (beeswarm) for the best model on the full dataset."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=15, show=False)
    plt.title(f'SHAP Summary — {model_name}', fontsize=13,
              fontweight='bold', pad=12)
    plt.tight_layout()
    _save(fig, 'shap_summary.png')
    return fig
