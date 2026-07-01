"""
hap_analysis.py
===============

Standalone, notebook-facing Helix Angle Pitch (HAP) / Helix Angle Range (HAR)
analysis for the cDTI project. Companion to ``yolo_pipeline.py``.

Purpose
-------
Pull the per-mask HAP/HAR computation that currently lives inline in
``hap.ipynb`` (cells 1 & 3) into clean, testable functions so you can run it
*once for the ground-truth mask and once for the predicted mask* and compare,
the same way MD/FA are compared. This file does NOT modify CarDpy -- it calls
CarDpy's ``cDTI_recon`` / ``Endo2Epi_Grid`` exactly as the notebook does and only
cleans up the regression / book-keeping around them.

What HAP / HAR are (as computed here)
-------------------------------------
* Helix Angle (HA): per-voxel angle of the primary eigenvector, ~+60 deg at the
  endocardium trending to ~-60 deg at the epicardium across the wall.
* Transmural depth (Endo->Epi grid): a 0..1 coordinate, 0 = endocardium,
  1 = epicardium.
* HAP (pitch) = slope of a linear fit HA vs depth = d(HA)/d(depth), in
  deg per full wall (depth 0->1). Physiologically negative (HA decreases
  outward), typically tens of degrees in magnitude.
* HAR (range) = HA(endo) - HA(epi) = -HAP for a depth axis spanning exactly
  [0, 1]. Reported because it is sometimes easier to interpret.

SHALLOW-HAP DIAGNOSIS (verified against CarDpy source)
------------------------------------------------------
CarDpy already returns HA in **degrees** (``Microstructure_Angle_Projections``
ends every angle with ``* 180 / np.pi``), so a "too shallow" HAP is NOT a
radian/degree unit bug. The two real causes, in order:

1. **Using the RAW ``HA`` map instead of the filtered ``HASF`` (or ``HALF``).**
   The raw helix angle is computed with ``math.asin(...)`` and a polarity check,
   which leaves sign-ambiguous voxels near the endo/epi borders (e.g. an
   epicardial fiber that should read -60 deg showing +60). CarDpy's
   ``Spatial_Helix_Angle_Filtering`` / ``Linear_Helix_Angle_Filtering`` exist
   precisely to flip those mis-polarized voxels (see their "Autoadjust
   Epicardium/Endocardium" sign flips). Wrong-sign voxels at the wall extremes
   drag the regression slope toward zero -> shallow HAP. ``cDTI_recon`` returns
   ``HA`` (raw), ``HALF`` (linear-filtered), and ``HASF`` (spatial-filtered);
   the notebook currently fits the raw ``HA``. Prefer ``HASF``.

   Quick check in the notebook -- fit both and compare:
       raw  = helix_angle_pitch_per_slice(M['HA'],   depth, mask)['HAP']
       filt = helix_angle_pitch_per_slice(M['HASF'], depth, mask)['HAP']
   ``filt`` should be visibly steeper (closer to the expected ~-45).

2. **Depth axis not spanning [0, 1]** (e.g. left in interpolation-index units
   0..200, then clipped to 1 so most voxels collapse to a single depth).
   ``helix_angle_pitch_per_slice`` warns if the grid is outside [0, 1].
   For reference, CarDpy's ``Endo2Epi_Grid`` returns endo~=0, epi~=1 (cubic
   ``griddata``), matching the orientation used here.

Units are therefore assumed to be **degrees** and are NOT auto-converted. If you
ever feed radians on purpose, pass ``assume_units="rad"``; ``"auto"`` enables the
old range-based guess (kept opt-in to avoid a false positive on a near-flat
slice).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# Heuristic threshold: |HA| never exceeds ~pi for radian data but reaches ~90
# for degree data. Anything whose max magnitude sits below this is treated as
# radians.
_RADIAN_MAX_ABS = 3.2  # a hair above pi


def ensure_degrees(ha: np.ndarray, assume: Optional[str] = None) -> np.ndarray:
    """Return HA in **degrees**.

    CarDpy emits HA in degrees, so the default (``assume=None``) is a no-op.
    Pass ``"rad"`` to force a radian->degree conversion, or ``"auto"`` to infer
    from the value range (kept opt-in: a near-flat slice could otherwise be
    misread as radians and wrongly scaled by 57.3x).
    """
    ha = np.asarray(ha, dtype=float)
    if assume in (None, "deg"):
        return ha
    if assume == "rad":
        return np.degrees(ha)
    if assume == "auto":
        finite = ha[np.isfinite(ha)]
        if finite.size and np.nanmax(np.abs(finite)) <= _RADIAN_MAX_ABS:
            warnings.warn(
                "HA looks like RADIANS (max |HA| <= %.1f); converting to degrees."
                % _RADIAN_MAX_ABS, stacklevel=2)
            return np.degrees(ha)
        return ha
    raise ValueError("assume must be None, 'deg', 'rad', or 'auto'")


def _fit_slope(depth: np.ndarray, ha_deg: np.ndarray, outlier_stdev: Optional[float]):
    """Least-squares slope/intercept of ha_deg ~ depth, optional outlier trim.

    Returns ``(slope, intercept, n_used)``. ``slope`` is HAP (deg per unit
    depth). NaNs are dropped. If ``outlier_stdev`` is given, points whose
    residual exceeds that many std-devs from the first fit are dropped and the
    fit is repeated once (matches the spirit of CarDpy's linear outlier filter).
    """
    m = np.isfinite(depth) & np.isfinite(ha_deg)
    d, y = depth[m], ha_deg[m]
    if d.size < 2 or np.ptp(d) == 0:
        return np.nan, np.nan, int(d.size)

    slope, intercept = np.polyfit(d, y, 1)
    if outlier_stdev is not None and d.size > 3:
        resid = y - (slope * d + intercept)
        keep = np.abs(resid) <= outlier_stdev * np.std(resid)
        if keep.sum() >= 2 and np.ptp(d[keep]) > 0:
            slope, intercept = np.polyfit(d[keep], y[keep], 1)
            d = d[keep]
    return float(slope), float(intercept), int(d.size)


def helix_angle_pitch_per_slice(ha: np.ndarray,
                                depth_grid: np.ndarray,
                                mask: Optional[np.ndarray] = None,
                                slice_axis: int = 2,
                                assume_units: Optional[str] = None,
                                outlier_stdev: Optional[float] = None) -> dict:
    """Per-slice HAP and HAR from a helix-angle map and an Endo->Epi depth grid.

    Pure (no CarDpy): give it the arrays and it returns the fits. This is the
    cleaned-up replacement for the regression loop in ``hap.ipynb`` cell 3.

    Parameters
    ----------
    ha : array, helix angle per voxel (deg or rad; auto-detected).
    depth_grid : array, same shape, transmural depth in [0, 1] (endo=0, epi=1).
    mask : optional bool/int array; if given, only voxels where ``mask`` is
        truthy and depth is finite are used (pass the *same* myocardial mask you
        built the grid from).
    slice_axis : axis indexing slices (default 2, i.e. arrays are X x Y x Z).
    assume_units : None | "deg" | "rad" forwarded to :func:`ensure_degrees`.
    outlier_stdev : optional residual-trim threshold (e.g. 1.0).

    Returns
    -------
    dict with arrays over slices: ``HAP`` (slope, deg/wall), ``HAR``
    (HA_endo - HA_epi, deg), ``intercept``, ``n`` (voxels used per slice).
    """
    ha_deg = ensure_degrees(ha, assume=assume_units)
    depth = np.asarray(depth_grid, dtype=float)

    if depth_grid is not None:
        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size and (finite_depth.min() < -0.01 or finite_depth.max() > 1.01):
            warnings.warn(
                "Endo->Epi depth grid is outside [0, 1] (min=%.3f, max=%.3f). A "
                "shallow HAP often means the depth axis is not normalized to the "
                "wall. Normalize depth to [0, 1] before fitting." %
                (float(finite_depth.min()), float(finite_depth.max())),
                stacklevel=2,
            )

    if mask is not None:
        valid = np.asarray(mask).astype(bool)
        depth = np.where(valid, depth, np.nan)

    ha_deg = np.moveaxis(ha_deg, slice_axis, 0)
    depth = np.moveaxis(depth, slice_axis, 0)

    HAP, HAR, intercept, n = [], [], [], []
    for s in range(ha_deg.shape[0]):
        d = depth[s].ravel()
        y = ha_deg[s].ravel()
        slope, b, used = _fit_slope(d, y, outlier_stdev)
        HAP.append(slope)
        # HA(endo=0) - HA(epi=1) evaluated on the fitted line == -slope
        HAR.append((b + slope * 0.0) - (b + slope * 1.0))
        intercept.append(b)
        n.append(used)

    return {
        "HAP": np.asarray(HAP),
        "HAR": np.asarray(HAR),
        "intercept": np.asarray(intercept),
        "n": np.asarray(n),
    }


# ---------------------------------------------------------------------------
# CarDpy wrapper: mask + eigenvectors  ->  HAP/HAR  (one call per mask)
# ---------------------------------------------------------------------------
def run_hap_for_mask(myocardial_mask: np.ndarray,
                     eigenvectors,
                     num_interp_points: int = 200,
                     smoothness_level: str = "Native",
                     helix_angle_filter_settings: Optional[dict] = None,
                     ha_key: str = "HASF",
                     assume_units: Optional[str] = None,
                     outlier_stdev: Optional[float] = None) -> dict:
    """Run CarDpy's cDTI recon for one mask, then compute per-slice HAP/HAR.

    This reproduces ``hap.ipynb`` cell 1's CarDpy calls (``cDTI_recon`` +
    ``Endo2Epi_Grid``) and feeds the result into
    :func:`helix_angle_pitch_per_slice`. CarDpy is imported lazily so this module
    imports fine without the T7 drive mounted.

    ``ha_key`` selects which helix-angle map to fit: ``"HASF"`` (spatial-filtered,
    recommended -- corrects the polarity-ambiguous voxels that make the raw map's
    HAP shallow), ``"HALF"`` (linear-filtered), or ``"HA"`` (raw, what the current
    notebook uses).

    Returns the per-slice dict plus the chosen ``HA`` and ``depth`` grids and the
    (possibly smoothed) mask CarDpy used, so the notebook can still plot.
    """
    from cardpy.Data_Processing.cDTI import cDTI_recon, Endo2Epi_Grid  # lazy

    if helix_angle_filter_settings is None:
        helix_angle_filter_settings = {
            "Linear Filter: Outlier StDev": 1,
            "Spatial Filter: Wall Depth Factor": 0.25,
            "Spatial Filter: Kernel Size": 5,
        }

    Cardiac_DTI_Metrics, Epi, Endo, Mask = cDTI_recon(
        myocardial_mask, eigenvectors, num_interp_points,
        smoothness_level, helix_angle_filter_settings,
    )

    # NOTE: use np.array_equal -- `if a == b:` on arrays raises ValueError
    # ("truth value of an array ... is ambiguous"), which is a latent bug in the
    # current notebook cell 1.
    if not np.array_equal(myocardial_mask, Mask):
        warnings.warn(
            "CarDpy smoothed the mask (smoothness_level=%r). HAP is computed on "
            "the smoothed mask; use the same mask for MD/FA to stay consistent."
            % smoothness_level, stacklevel=2,
        )

    mask_f = Mask.astype(float)
    mask_f[mask_f == 0] = np.nan
    depth = np.clip(Endo2Epi_Grid(Mask) * mask_f, 0.0, 1.0)

    result = helix_angle_pitch_per_slice(
        Cardiac_DTI_Metrics[ha_key], depth, mask=Mask,
        slice_axis=2, assume_units=assume_units, outlier_stdev=outlier_stdev,
    )
    result["ha_key"] = ha_key
    result["HA"] = Cardiac_DTI_Metrics[ha_key]
    result["depth"] = depth
    result["mask_used"] = Mask
    result["metrics"] = Cardiac_DTI_Metrics
    return result


def compare_gt_pred(gt_mask: np.ndarray,
                    pred_mask: np.ndarray,
                    eigenvectors,
                    **kwargs) -> dict:
    """Run HAP/HAR for GT and predicted masks and report per-slice differences.

    Drop-in for the "repeat for predicted mask / compare" TODO in cell 3.
    Returns ``{"gt": ..., "pred": ..., "HAP_diff": ..., "HAR_diff": ...}``.
    """
    gt = run_hap_for_mask(gt_mask, eigenvectors, **kwargs)
    pred = run_hap_for_mask(pred_mask, eigenvectors, **kwargs)
    return {
        "gt": gt,
        "pred": pred,
        "HAP_diff": pred["HAP"] - gt["HAP"],
        "HAR_diff": pred["HAR"] - gt["HAR"],
    }


if __name__ == "__main__":  # pragma: no cover - illustrative
    # Tiny synthetic sanity check (no CarDpy needed): a wall whose HA goes
    # linearly +60 -> -60 across normalized depth must yield HAP = -120, HAR=+120.
    Z = 1
    depth = np.linspace(0, 1, 50).reshape(50, 1, 1).repeat(1, 1)
    ha = 60 - 120 * depth  # degrees
    out = helix_angle_pitch_per_slice(ha, depth, slice_axis=2)
    print("HAP:", out["HAP"], "HAR:", out["HAR"])  # -> [-120.] [120.]
