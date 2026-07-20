"""
Shared comparison plots + CSV export for the automatic-vs-GT results.

`full_report(runs, out_dir)` takes a dict {label: DataFrame | .xlsx/.csv path} (one entry
per approach/model), and writes to out_dir:
  - comparison_percase.csv   : every case row, tagged with 'approach'  (for your own plots)
  - comparison_summary.csv   : mean/median/std per approach            (read the exact DSC/MD/FA
                               numbers when the violins are too close to eyeball)
  - dice_violin.png          : LV Dice by approach
  - ip_hausdorff_violin.png  : insertion-point Hausdorff (misses reported separately)
  - gtpred_<METRIC>.png      : MD / FA / DWI GT-vs-Prediction split violins (median_boxplot.ipynb style)

Used by CompareApproaches.ipynb and by SmartHealthTestingHannum.ipynb's final section.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DICE = 'Dice Score Original Label 1'
HD2, HD3 = 'Hausdorff Distance Label 2', 'Hausdorff Distance Label 3'
HD_MISS = 1250.0   # process_folders failsafe for a MISSED insertion point (1000 x 1.25)
CASE = 'Case ID'
METRIC_LABELS = {
    'MD':  'Mean Diffusivity [$\\mu m^2 / ms$]',
    'FA':  'Fractional Anisotropy',
    'DWI': 'DWI signal [a.u.]',
}


def load_runs(runs):
    """runs: {label: DataFrame | .xlsx/.csv path} -> (tidy DataFrame with 'approach', order list)."""
    frames, order = [], []
    for label, src in runs.items():
        if isinstance(src, pd.DataFrame):
            df = src.copy()
        elif str(src).endswith('.xlsx'):
            df = pd.read_excel(src)
        else:
            df = pd.read_csv(src)
        df.insert(0, 'approach', label)
        frames.append(df)
        order.append(label)
    alldf = pd.concat(frames, ignore_index=True)
    alldf['approach'] = pd.Categorical(alldf['approach'], categories=order, ordered=True)
    return alldf, order


def _violin(ax, groups, labels, ylabel, title, ylim=None):
    vals = [np.asarray(g, float) for g in groups]
    vals = [v[np.isfinite(v)] for v in vals]
    parts = ax.violinplot(vals, showmedians=True, showextrema=False, widths=0.8)
    for b in parts['bodies']:
        b.set_alpha(0.45)
    parts['cmedians'].set_color('k'); parts['cmedians'].set_linewidth(2)
    for i, v in enumerate(vals):
        if len(v):
            ax.scatter(np.random.normal(i + 1, 0.05, len(v)), v, s=12, alpha=0.5, color='k', zorder=3)
            ax.text(i + 1, np.median(v), f'{np.median(v):.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


def dice_violin(alldf, order, out_dir):
    fig, ax = plt.subplots(figsize=(1.8 * len(order) + 3, 5))
    _violin(ax, [alldf[alldf.approach == a][DICE].values for a in order], order,
            'Dice (LV, label 1)', 'LV Dice — automatic vs GT, by approach', ylim=(0, 1))
    plt.tight_layout(); p = os.path.join(out_dir, 'dice_violin.png'); plt.savefig(p, dpi=200); plt.show()
    return p


def ip_hausdorff_violin(alldf, order, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(1.8 * len(order) + 4, 5))
    for ax, hd, name in zip(axes, [HD2, HD3], ['anterior (label 2)', 'inferior (label 3)']):
        if hd not in alldf.columns:
            ax.set_visible(False); continue
        groups = []
        for a in order:
            v = alldf[alldf.approach == a][hd].values
            print(f'  {a} | IP {name}: {(v >= HD_MISS).mean() * 100:.0f}% missed')
            groups.append(v[v < HD_MISS])          # detected points only
        _violin(ax, groups, order, 'Hausdorff [mm]', f'IP {name} (detected only)')
    plt.tight_layout(); p = os.path.join(out_dir, 'ip_hausdorff_violin.png'); plt.savefig(p, dpi=200); plt.show()
    return p


def gt_pred_violins(alldf, order, out_dir):
    """MD / FA / DWI GT-vs-Prediction split violins, exact median_boxplot.ipynb styling."""
    import seaborn as sns
    from matplotlib.lines import Line2D
    paths = []
    pairs = []
    for c in alldf.columns:
        m = re.match(r'GT_Median_(\w+)$', c)
        if m and f'Pred_median_{m.group(1)}' in alldf.columns:
            pairs.append((m.group(1), c, f'Pred_median_{m.group(1)}'))
    for metric, gt_col, pr_col in pairs:
        melted = alldf[[CASE, 'approach', gt_col, pr_col]].melt(
            id_vars=[CASE, 'approach'], value_vars=[gt_col, pr_col], var_name='Type', value_name='Median')
        melted['Type'] = melted['Type'].replace({gt_col: 'GT', pr_col: 'Pred'})
        approaches = [a for a in order if a in set(melted['approach'].astype(str))]
        pred_colors = {a: plt.cm.tab10(i % 10) for i, a in enumerate(approaches)}

        plt.figure(figsize=(2.4 * len(approaches) + 2, 4), dpi=300)
        plt.title(f'{metric} — GT vs Prediction', fontsize=16)
        sns.violinplot(x='approach', y='Median', hue='Type', data=melted, split=True,
                       inner='quartile', palette={'GT': 'grey', 'Pred': 'black'},
                       cut=0, width=0.5, order=approaches)
        for idx, a in enumerate(approaches):
            for patch in plt.gca().collections[2 * idx:2 * idx + 2]:
                patch.set_facecolor('grey' if patch.get_paths()[0].vertices[:, 0].min() < idx else pred_colors[a])
                patch.set_edgecolor('black'); patch.set_alpha(0.6)
        for idx, a in enumerate(approaches):
            d = melted[melted['approach'] == a]
            gt = d[d['Type'] == 'GT'].sort_values(CASE)['Median'].values
            pr = d[d['Type'] == 'Pred'].sort_values(CASE)['Median'].values
            plt.scatter(np.full(len(gt), idx - 0.1) + np.random.uniform(-0.02, 0.02, len(gt)), gt,
                        color='black', s=10, alpha=0.7)
            plt.scatter(np.full(len(pr), idx + 0.1) + np.random.uniform(-0.02, 0.02, len(pr)), pr,
                        color='black', s=10, alpha=0.7)
            for g, prd in zip(gt, pr):
                plt.plot([idx - 0.1, idx + 0.1], [g, prd], color='black', alpha=0.3)
        plt.xticks(fontsize=12, rotation=15, ha='right')
        plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=13); plt.xlabel('')
        plt.legend(handles=[Line2D([0], [0], color='grey', lw=6, label='GT')], fontsize=11)
        plt.tight_layout(); p = os.path.join(out_dir, f'gtpred_{metric}.png')
        plt.savefig(p, dpi=300); plt.show(); paths.append(p)
    return paths


def full_report(runs, out_dir):
    """Write all CSVs + PNGs for a set of runs. Returns (tidy DataFrame, summary DataFrame)."""
    os.makedirs(out_dir, exist_ok=True)
    alldf, order = load_runs(runs)
    alldf.to_csv(os.path.join(out_dir, 'comparison_percase.csv'), index=False)

    metrics = [c for c in [DICE, 'F1 Label 1', 'Precision', 'Recall', HD2, HD3,
                           'GT_Median_MD', 'Pred_median_MD', 'GT_Median_FA', 'Pred_median_FA',
                           'GT_Median_DWI', 'Pred_median_DWI', 'Avg. HD Epi', 'Avg. HD Endo']
               if c in alldf.columns]
    summary = alldf.groupby('approach', observed=True)[metrics].agg(['mean', 'median', 'std'])
    summary.to_csv(os.path.join(out_dir, 'comparison_summary.csv'))
    print('=== summary (mean / median / std per approach) ===')
    print(summary.round(3))
    print()

    dice_violin(alldf, order, out_dir)
    ip_hausdorff_violin(alldf, order, out_dir)
    gt_pred_violins(alldf, order, out_dir)
    print(f'\nsaved CSVs + PNGs to {out_dir}')
    return alldf, summary
