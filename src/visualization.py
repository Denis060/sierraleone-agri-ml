"""
visualization.py
----------------
All plotting and figure generation functions for the
Sierra Leone Agricultural ML project.

Author: Ibrahim Denis Fofanah
Affiliation: Pace University / RiseAfrica Foundation for STEM and Innovation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Style Configuration ────────────────────────────────────────────────────────
COLORS = {
    'primary':    '#1A6B47',   # Forest green
    'secondary':  '#D4A017',   # Gold
    'accent':     '#4CAF72',   # Lime green
    'dark':       '#0D3B2E',   # Dark forest
    'light':      '#E0F4F4',   # Light teal
    'danger':     '#C0392B',   # Red for shocks
    'neutral':    '#7F8C8D',   # Gray
}

CROP_COLORS = {
    'Rice':        '#1A6B47',
    'Cassava':     '#D4A017',
    'Maize':       '#4CAF72',
    'Groundnuts':  '#E67E22',
    'Oil palm fruit': '#8E44AD',
    'Sweet Potato':'#2980B9',
    'Sorghum':     '#C0392B',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'font.family':       'sans-serif',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def save_figure(fig, filename: str, output_dir: str = 'outputs/figures',
                dpi: int = 300) -> None:
    """Save figure as high-resolution PNG and PDF."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    print(f"[✓] Figure saved: {filename}")


def plot_rice_yield_trend(df: pd.DataFrame,
                          output_dir: str = 'outputs/figures') -> plt.Figure:
    """
    Plot Rice yield trend 2000-2024 with shock annotations.
    This is Figure 1 in the paper.
    """
    rice = df[['Year', 'Rice_Yield']].dropna()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Main line
    ax.plot(rice['Year'], rice['Rice_Yield'],
            color=COLORS['primary'], linewidth=2.5,
            marker='o', markersize=5, label='Rice Yield (kg/ha)')

    # Fill under the curve
    ax.fill_between(rice['Year'], rice['Rice_Yield'],
                    alpha=0.12, color=COLORS['primary'])

    # Shock annotations
    shocks = {
        2014: ('Ebola\nCrisis', '#C0392B'),
        2018: ('Yield\nCrash', '#E67E22'),
        2020: ('COVID-19', '#8E44AD'),
        2023: ('Feed\nSalone\nLaunch', '#1A6B47'),
    }
    for year, (label, color) in shocks.items():
        y_val = rice[rice['Year'] == year]['Rice_Yield'].values
        if len(y_val) > 0:
            ax.axvline(x=year, color=color, linestyle='--',
                       alpha=0.5, linewidth=1.2)
            ax.annotate(label,
                        xy=(year, y_val[0]),
                        xytext=(year + 0.3, y_val[0] + 120),
                        fontsize=8.5, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # National average line
    mean_yield = rice['Rice_Yield'].mean()
    ax.axhline(y=mean_yield, color=COLORS['secondary'],
               linestyle=':', linewidth=1.5,
               label=f'24-yr Average: {mean_yield:.0f} kg/ha')

    ax.set_title('Rice Yield Trends in Sierra Leone (2000–2024)\n'
                 'with Key Agricultural and Economic Shock Periods',
                 pad=15)
    ax.set_xlabel('Year')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_xlim(rice['Year'].min() - 0.5, rice['Year'].max() + 0.5)
    ax.legend(loc='upper left', framealpha=0.9)

    # Source annotation
    ax.annotate('Source: FAOSTAT (2025). Sierra Leone Crops and Livestock Products.',
                xy=(0.01, -0.12), xycoords='axes fraction',
                fontsize=8, color=COLORS['neutral'], style='italic')

    fig.tight_layout()
    save_figure(fig, 'fig1_rice_yield_trend', output_dir)
    return fig


def plot_multi_crop_yield(df: pd.DataFrame,
                          crops: list = None,
                          output_dir: str = 'outputs/figures') -> plt.Figure:
    """
    Plot yield trends for multiple key crops.
    This is Figure 2 in the paper.
    """
    if crops is None:
        crops = ['Rice', 'Cassava', 'Maize', 'Groundnuts', 'Sweet Potato']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, crop in enumerate(crops):
        col = f'{crop}_Yield'
        if col not in df.columns:
            col = f'{crop.replace(" ", "_")}_Yield'
        if col not in df.columns:
            continue

        data = df[['Year', col]].dropna()
        color = CROP_COLORS.get(crop, COLORS['primary'])

        axes[i].plot(data['Year'], data[col],
                     color=color, linewidth=2,
                     marker='o', markersize=4)
        axes[i].fill_between(data['Year'], data[col],
                              alpha=0.1, color=color)
        axes[i].set_title(f'{crop}')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Yield (kg/ha)')

        # Trend line
        z = np.polyfit(data['Year'], data[col], 1)
        p = np.poly1d(z)
        axes[i].plot(data['Year'], p(data['Year']),
                     color=COLORS['secondary'], linestyle='--',
                     linewidth=1.2, alpha=0.7, label='Trend')
        axes[i].legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(crops), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Crop Yield Trends in Sierra Leone (2000–2024)\n'
                 'Key Staple and Cash Crops',
                 fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()
    save_figure(fig, 'fig2_multi_crop_yields', output_dir)
    return fig


def plot_model_comparison(results_df: pd.DataFrame,
                          output_dir: str = 'outputs/figures') -> plt.Figure:
    """
    Plot model performance comparison bar chart.
    This is Figure 3 in the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [('R²', True), ('RMSE', False), ('MAE', False)]
    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    for i, (metric, higher_better) in enumerate(metrics):
        if metric not in results_df.columns:
            continue

        sorted_df = results_df.sort_values(metric, ascending=not higher_better)
        bars = axes[i].bar(sorted_df['Model'], sorted_df[metric],
                           color=bar_colors, edgecolor='white',
                           linewidth=0.8, alpha=0.9)

        # Value labels on bars
        for bar, val in zip(bars, sorted_df[metric]):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + (bar.get_height() * 0.02),
                         f'{val:.4f}' if metric == 'R²' else f'{val:.2f}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        axes[i].set_title(f'{metric}\n{"(Higher = Better)" if higher_better else "(Lower = Better)"}')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=15)

        # Highlight best
        best_idx = sorted_df[metric].idxmax() if higher_better else sorted_df[metric].idxmin()
        best_model = results_df.loc[best_idx, 'Model']
        best_pos = sorted_df['Model'].tolist().index(best_model)
        bars[best_pos].set_edgecolor(COLORS['dark'])
        bars[best_pos].set_linewidth(2)

    fig.suptitle('Model Performance Comparison\nRandom Forest vs XGBoost vs Gradient Boosting',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig3_model_comparison', output_dir)
    return fig


def plot_feature_importance(fi_df: pd.DataFrame,
                            top_n: int = 15,
                            model_name: str = '',
                            output_dir: str = 'outputs/figures') -> plt.Figure:
    """
    Plot top-N feature importances as a horizontal bar chart.
    This is Figure 4 in the paper.
    """
    top = fi_df.head(top_n).sort_values('Importance')

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [COLORS['primary'] if i >= len(top) - 3
              else COLORS['accent']
              for i in range(len(top))]

    bars = ax.barh(top['Feature'], top['Importance'],
                   color=colors, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, top['Importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8.5)

    ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
    ax.set_title(f'Top {top_n} Feature Importances\n{model_name}',
                 pad=12)
    ax.set_xlim(0, top['Importance'].max() * 1.15)

    fig.tight_layout()
    save_figure(fig, f'fig4_feature_importance_{model_name.replace(" ", "_").lower()}',
                output_dir)
    return fig


def plot_actual_vs_predicted(y_test: np.ndarray,
                             predictions: dict,
                             output_dir: str = 'outputs/figures') -> plt.Figure:
    """
    Plot actual vs predicted values for all models.
    This is Figure 5 in the paper.
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.7, color=colors[i],
                   edgecolors='white', linewidth=0.5, s=60)

        # Perfect prediction line
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.2,
                alpha=0.6, label='Perfect Prediction')

        # R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.92, f'R² = {r2:.4f}',
                transform=ax.transAxes, fontsize=11,
                fontweight='bold', color=colors[i])

        ax.set_title(name)
        ax.set_xlabel('Actual Yield (kg/ha)')
        ax.set_ylabel('Predicted Yield (kg/ha)')
        ax.legend(fontsize=8)

    fig.suptitle('Actual vs Predicted Crop Yield\nModel Comparison',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, 'fig5_actual_vs_predicted', output_dir)
    return fig


def plot_production_overview(df: pd.DataFrame,
                             output_dir: str = 'outputs/figures') -> plt.Figure:
    """
    Plot total production overview for key crops.
    This is Figure 6 / context figure.
    """
    prod_cols = [c for c in df.columns if c.endswith('_Production')]

    # Most recent available value per crop (forward-fill so sparsely reported
    # crops still contribute their latest figure), then keep the top crops.
    df_sorted = df.sort_values('Year')
    latest_year = int(df_sorted['Year'].max())
    latest_vals = df_sorted[prod_cols].ffill().iloc[-1]

    pairs = [(v, c.replace('_Production', ''))
             for c, v in latest_vals.items() if pd.notna(v) and v > 0]
    pairs = sorted(pairs, reverse=True)[:12]   # top 12 by production
    values, crop_labels = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
                  COLORS['dark']] * 5
    bars = ax.bar(crop_labels, values,
                  color=bar_colors[:len(crop_labels)],
                  edgecolor='white', linewidth=0.8, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + bar.get_height() * 0.02,
                f'{val/1000:.0f}K t',
                ha='center', va='bottom', fontsize=8.5)

    ax.set_title(f'Agricultural Production by Crop — Sierra Leone ({latest_year})\n'
                 'Top Crops by Production Quantity (tonnes)',
                 pad=12)
    ax.set_ylabel('Production (tonnes)')
    ax.tick_params(axis='x', rotation=30)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    ax.annotate('Source: FAOSTAT (2025).',
                xy=(0.01, -0.18), xycoords='axes fraction',
                fontsize=8, color=COLORS['neutral'], style='italic')

    fig.tight_layout()
    save_figure(fig, 'fig6_production_overview', output_dir)
    return fig
