"""
Shared comparison plots + CSV export for the automatic-vs-GT results.

`full_report(runs, out_dir)` takes a dict {label: DataFrame | .xlsx/.csv path} (one entry
per approach/model), and writes to out_dir:
  - comparison_percase.csv   : every case row, tagged with 'approach'  (for your own plots)
  - comparison_summary.csv   : mean/median/std per approach            (read the exact DSC/MD/FA
                               numbers when the violins are too close to eyeball)
  - dice_violin.png          : LV Dice by approach
  - ip_hausdorff_violin.png  : insertion-point Hausdorff (misses reported separately)
  - gtpred_<METRIC>.png      : MD / FA / DWI (+ helix HAP / HAR if present) GT-vs-Prediction split violins
                               (median_boxplot.ipynb style)

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
    'HAP': 'Helix angle pitch [deg / wall]',
    'HAR': 'Helix angle range [deg]',
}
METRIC_TITLES = {'HAP': 'Helix angle pitch', 'HAR': 'Helix angle range'}


def _fig_width(n):
    return max(5.5, 2.6 * n + 1.5)   # wide enough for the title + two-line run labels


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
            ax.text(i + 1.3, np.median(v), f'median\n{np.median(v):.2f}', ha='left', va='center', fontsize=9)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


def dice_violin(alldf, order, out_dir):
    """LV Dice per approach, in the median_boxplot.ipynb paper aesthetic (seaborn, tab10,
    inner quartiles, jittered per-case points, median annotated). Single distribution
    (Dice already IS the GT-vs-pred agreement, so no split)."""
    import seaborn as sns
    d = alldf[['approach', DICE]].dropna()
    approaches = [a for a in order if a in set(d['approach'].astype(str))]
    colors = {a: plt.cm.tab10(i % 10) for i, a in enumerate(approaches)}

    plt.figure(figsize=(_fig_width(len(approaches)), 4.8), dpi=300)
    plt.title('LV Dice — automatic vs manual', fontsize=15)
    sns.violinplot(x='approach', y=DICE, data=d, order=approaches, inner='quartile',
                   palette=colors, cut=0, width=0.6)
    for patch in plt.gca().collections:
        patch.set_alpha(0.6); patch.set_edgecolor('black')
    for idx, a in enumerate(approaches):
        vals = d[d['approach'] == a][DICE].values
        plt.scatter(np.full(len(vals), idx) + np.random.uniform(-0.08, 0.08, len(vals)), vals,
                    color='black', s=12, alpha=0.6, zorder=3)
        plt.text(idx + 0.33, np.median(vals), f'median\n{np.median(vals):.3f}', ha='left', va='center', fontsize=9)
    plt.ylim(0.6, 1); plt.xlim(-0.6, len(approaches) - 0.1)
    plt.ylabel('Dice (LV myocardium)', fontsize=13); plt.xlabel('')
    plt.xticks(fontsize=10); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); p = os.path.join(out_dir, 'dice_violin.png')
    plt.savefig(p, dpi=300); plt.show()
    return p


def ip_hausdorff_violin(alldf, order, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(2 * _fig_width(len(order)), 5))
    for ax, hd, name in zip(axes, [HD2, HD3], ['anterior (label 2)', 'inferior (label 3)']):
        if hd not in alldf.columns:
            ax.set_visible(False); continue
        groups = []
        for a in order:
            v = alldf[alldf.approach == a][hd].values
            print(f'  {a} | IP {name}: {(v >= HD_MISS).mean() * 100:.0f}% missed')
            groups.append(v[v < HD_MISS])          # detected points only
        _violin(ax, groups, order, 'distance to manual point [mm]', f'Insertion point, {name}')
    plt.tight_layout(); p = os.path.join(out_dir, 'ip_hausdorff_violin.png'); plt.savefig(p, dpi=200); plt.show()
    return p


def gt_pred_violins(alldf, order, out_dir):
    """MD / FA / DWI GT-vs-Prediction split violins, exact median_boxplot.ipynb styling."""
    import seaborn as sns
    paths = []
    pairs = []
    for c in alldf.columns:
        m = re.match(r'GT_(?:Median_)?(\w+)$', c)          # GT_Median_MD / GT_HAP
        if not m:
            continue
        pr = next((p for p in (f'Pred_median_{m.group(1)}', f'Pred_{m.group(1)}') if p in alldf.columns), None)
        if pr and alldf[c].notna().any():
            pairs.append((m.group(1), c, pr))
    for metric, gt_col, pr_col in pairs:
        melted = alldf[[CASE, 'approach', gt_col, pr_col]].melt(
            id_vars=[CASE, 'approach'], value_vars=[gt_col, pr_col], var_name='Type', value_name='Median')
        melted['Type'] = melted['Type'].replace({gt_col: 'GT', pr_col: 'Pred'})
        melted = melted.dropna(subset=['Median'])           # e.g. a run without helix results yet
        approaches = [a for a in order if a in set(melted['approach'].astype(str))]
        melted['approach'] = pd.Categorical(melted['approach'].astype(str), categories=approaches, ordered=True)
        pred_colors = {a: plt.cm.tab10(i % 10) for i, a in enumerate(approaches)}

        plt.figure(figsize=(_fig_width(len(approaches)), 4.8), dpi=300)
        plt.title(f'{METRIC_TITLES.get(metric, metric)} — manual vs automatic mask', fontsize=14)
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
        plt.xticks(fontsize=10)
        plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=12); plt.xlabel('')
        if plt.gca().get_legend():
            plt.gca().get_legend().remove()
        plt.figtext(0.5, 0.01, 'left half: manual segmentation (GT)   ·   right half: automatic segmentation',
                    ha='center', va='bottom', fontsize=10)
        plt.tight_layout(rect=(0, 0.05, 1, 1)); p = os.path.join(out_dir, f'gtpred_{metric}.png')
        plt.savefig(p, dpi=300); plt.show(); paths.append(p)
    return paths


def full_report(runs, out_dir):
    """Write all CSVs + PNGs for a set of runs. Returns (tidy DataFrame, summary DataFrame)."""
    os.makedirs(out_dir, exist_ok=True)
    alldf, order = load_runs(runs)
    alldf.to_csv(os.path.join(out_dir, 'comparison_percase.csv'), index=False)

    metrics = [c for c in [DICE, 'F1 Label 1', 'Precision', 'Recall', HD2, HD3,
                           'GT_Median_MD', 'Pred_median_MD', 'GT_Median_FA', 'Pred_median_FA',
                           'GT_Median_DWI', 'Pred_median_DWI', 'Avg. HD Epi', 'Avg. HD Endo',
                           'GT_HAP', 'Pred_HAP', 'GT_HAR', 'Pred_HAR']
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


# ===========================================================================
#  Per-case GT-vs-Prediction overlays (full view + cropped), LV mask + IPs
#  Same colour convention as eval_util.plot_original_with_masks:
#    GT   -> LV yellow, IP red (+ marker)      Pred -> LV cyan, IP blue (x marker)
# ===========================================================================
from matplotlib.colors import ListedColormap as _LC
from matplotlib.patches import Patch as _Patch

_PRED_CMAP = _LC([(0, 0, 0, 0), (0, 1, 1, 0.45), (0, 0, 1, 0.55), (0, 0, 1, 0.55)])  # bg, LV, IP2, IP3
_GT_CMAP   = _LC([(0, 0, 0, 0), (1, 1, 0, 0.45), (1, 0, 0, 0.55), (1, 0, 0, 0.55)])
_LEGEND = [_Patch(facecolor=(1, 1, 0, 0.6), label='GT LV'),
           _Patch(facecolor=(0, 1, 1, 0.6), label='Pred LV'),
           _Patch(facecolor=(1, 0, 0, 0.7), label='GT IP'),
           _Patch(facecolor=(0, 0, 1, 0.7), label='Pred IP')]


def _norm(img):
    img = np.asarray(img, float)
    lo, hi = np.nanmin(img), np.nanmax(img)
    return (img - lo) / (hi - lo) if hi > lo else img


def _ip_markers(ax, gt_mask, pred_mask):
    from scipy.ndimage import center_of_mass
    for lbl in (2, 3):
        if np.any(gt_mask == lbl):
            y, x = center_of_mass(gt_mask == lbl); ax.scatter(x, y, c='red', marker='+', s=90, linewidths=2)
        if np.any(pred_mask == lbl):
            y, x = center_of_mass(pred_mask == lbl); ax.scatter(x, y, c='blue', marker='x', s=70, linewidths=2)


def save_case_overlay(out_png, full_img, full_gt, full_pred, crop_img, crop_gt, crop_pred, title=''):
    """Two-panel GT-vs-pred overlay: full-FoV (original spacing) + cropped 256. Saves a PNG."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, (im, gt, pr, sub) in zip(ax, [(full_img, full_gt, full_pred, 'full view (original spacing)'),
                                         (crop_img, crop_gt, crop_pred, 'cropped 256')]):
        a.imshow(_norm(im), cmap='gray')
        a.imshow(np.ma.masked_where(gt == 0, gt), cmap=_GT_CMAP, vmin=0, vmax=3, interpolation='none')
        a.imshow(np.ma.masked_where(pr == 0, pr), cmap=_PRED_CMAP, vmin=0, vmax=3, interpolation='none')
        _ip_markers(a, gt, pr)
        a.set_title(sub, fontsize=11); a.axis('off')
    ax[0].legend(handles=_LEGEND, loc='lower right', fontsize=8, framealpha=0.6)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out_png
