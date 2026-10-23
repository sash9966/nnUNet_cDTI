# Automated Whole-Heart cDTI LV Segmentation — Baseline Model

A fully automated pipeline for **left-ventricular (LV) myocardium segmentation on cardiac diffusion
tensor imaging (cDTI)**, plus right-ventricular **insertion-point (RVIP)** detection — designed to
drop into CarDpy for downstream cDTI metrics (MD, FA, helix angle).

> **Status:** segmentation model (LV + RVIP) released; helix angle on the predicted masks validated against
> the manual masks (pitch −61.0 vs −61.7 °/wall, median per-slice difference 4.2 °/wall). Open questions for
> co-author review are listed in the `master` README.

Pipeline: **YOLO heart crop → resize 256 → nnUNet ResEnc-M LV segmentation (5-fold) → YOLO RVIP
detection → un-crop to original FoV.**

---

## Intended use
Hand a new case folder (per slice: average DWI + mean diffusivity) to the pipeline and get back an
LV myocardium mask (+ insertion points) in the original image space, ready for CarDpy's cDTI analysis.

**Baseline model = 2-channel (MD + DWI).** We found two contrasts are sufficient (see results), so the
baseline keeps inputs minimal: only the **average DWI** and **mean diffusivity (MD)** maps are required.

### Inputs (per slice, after the YOLO crop + resize to 256×256)
| channel | image | normalization |
|---|---|---|
| 0 | Average diffusion-weighted image (DWI) | min–max to [0,1] |
| 1 | Mean diffusivity (MD) | divide by 4.0 |

The YOLO **crop** model runs on the average DWI only (1 channel). The YOLO **RVIP** model runs on the
average DWI (single-contrast).

---

## Expected performance
Held-out **SmartHealth test set: 39 slices, 6 volunteers (46, 47, 49, 50, 51, 52)**, never seen in
training (excluded via `splits_final.json`). 5-fold ensemble, `nnUNetResEncUNetMPlans`.

| metric | **Baseline (MD+DWI, 2ch)** |
|---|---|
| LV Dice | **0.909** |
| Precision | 0.889 |
| Recall | 0.932 |
| F1 | 0.909 |
| MD median agreement (\|%diff\|) | ~1.0% |
| FA median agreement (\|%diff\|) | ~1.8% |
| DWI median agreement (\|%diff\|) | ~2.9% |
| Insertion-point Hausdorff (median) | ~2.3–2.4 mm (mean ~5–6 mm: a few slices miss by 30–55 mm) |
| Epicardial / endocardial avg. HD | 0.29 / 0.18 |

### Why the 2-channel baseline (data vs contrasts)
We compared three training regimes on the same holdout:

| run | LV Dice | Recall | notes |
|---|---|---|---|
| SmartHealth only, 4 contrasts | 0.902 | 0.923 | baseline cohort |
| SmartHealth + DirVsAverages, 4 contrasts | 0.909 | 0.936 | **+data** |
| SmartHealth + DirVsAverages, **MD+DWI (2ch)** | 0.909 | 0.932 | **shipped baseline** |

**More data gives a small, consistent gain; dropping from 4 contrasts to MD+DWI costs essentially
nothing.** So the baseline uses more data + only the two most informative contrasts.

---

## Weights
Weights are **not included** in this repository (the 5 nnUNet folds are ~848 MB each, ~4.2 GB total).
They are shared separately on request. You need:

```
nnunet_lv/                       # LV segmentation (nnUNet v2, ResEnc-M, 2-channel MD+DWI), 5-fold ensemble
  fold_0/ ... fold_4/checkpoint_final.pth
  plans.json
  dataset.json
yolo/
  crop.pt                        # heart-crop detector (crop_D111)
  rvip.pt                        # insertion-point detector (rvip_D107, single-contrast avg)
```

---

## How to run
**Environment:** `nnunetv2`, `ultralytics`, `nibabel`, `opencv-python`, `cardpy` (for cDTI metrics).

```bash
export nnUNet_results=/path/to/nnUNet_results   # folder containing Dataset310_*/
export nnUNet_compile=false
```

nnUNet expects this exact layout under `nnUNet_results` (keep the folder names):

```
nnUNet_results/Dataset310_HannumSmartHealthandDirVsAvgsLVMDandAvg/
  nnUNetTrainerDA5_100epochsNoCompile__nnUNetResEncUNetMPlans__2d/
    dataset.json  plans.json  fold_0/ ... fold_4/checkpoint_final.pth
```

The notebook imports `yolo_pipeline.py`, `eval_util.py` and `comparison_plots.py` from the repo root.

The runnable reference is **`PaperRun_cDTI_Pipeline.ipynb`** with `COLLAB_MODE = True`: set your data
paths + the downloaded weights in the `COLLAB` dict, keep `RUN_CROP/RUN_LV/RUN_IP = True`, and set
`RUN_METRICS/RUN_HELIX/RUN_PLOTS = False` for pure inference. It produces LV masks (and insertion
points) per case in the original field of view — which CarDpy then consumes for MD/FA/helix-angle.

Minimal flow per case:
1. YOLO crop on the average DWI → square crop → resize to 256.
2. Normalize channels (DWI min–max, MD /4) → 2-channel input.
3. `nnUNetv2_predict -d <id> -c 2d -tr nnUNetTrainerDA5_100epochsNoCompile -p nnUNetResEncUNetMPlans -f 0 1 2 3 4` → LV mask.
4. YOLO RVIP on the average DWI → insertion points.
5. Un-crop the mask back to the original FoV (using the YOLO crop box).

---

## Training data
Hannum-annotated cDTI (Stanford CMR): **SmartHealth + Directions-vs-Averages** cohorts.
Test volunteers (46, 47, 49, 50, 51, 52) were held out of every training fold.
Trainer: `nnUNetTrainerDA5_100epochsNoCompile` (DA5 augmentation, torch.compile disabled), 100 epochs.

## Limitations
- Single-center (Stanford/Hannum), single acquisition protocol (DiVO_15_04). Generalization to other
  scanners/protocols is unvalidated.
- Expects the average DWI + MD maps already reconstructed (the pipeline does not do tensor fitting).
- Insertion-point detector is a weak single-class YOLO; use its points as an initialization.
- Distances in mm assume 1.25 mm pixels (`VOXEL_SPACING_MM`): the NIfTI exports only carry a placeholder
  1 × 1 mm spacing, so the true per-cohort spacing still has to be confirmed.
- Helix angle (`RUN_HELIX`) needs the raw `NifTis` (`Interpolated.nii/.bvals/.bvecs/.header`) per volunteer in
  `RAW_DIFFUSION_ROOT`; GT helix uses the native full-size manual mask (`04_Segmentation_Masks`).

## Citation / contact
Stocker et al., automated whole-heart cDTI segmentation (ISMRM).
