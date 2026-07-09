# Automated cDTI Whole-Heart Pipeline — Project Overview

Automated analysis pipeline for **cardiac diffusion tensor imaging (cDTI)**. It replaces
a manual, per-slice workflow with an automatic chain: locate the heart, segment the left
ventricle (LV), find the right-ventricular insertion points (RVIPs), map everything back to
the original image spacing, and then compute the physical DTI metrics (MD, FA, helix-angle
pitch) — so the automatic pipeline can be compared, on held-out test cases, against manual
ground truth and against a second reader (inter-reader study).

> **Status / how to read this:** this is the standing reference for the project. Section
> [Known issues & fixes](#known-issues--fixes-fingerprints) is a running log of the traps we
> hit and how to recognize them. Anything marked **⚠ CONFIRM** is a path/ID the human owner
> must pin down before a run.

---

## 1. High-level pipeline

```
                       ORIGINAL cDTI acquisition (full FoV, native spacing)
                                        │
             ┌──────────────────────────┴───────────────────────────┐
             │  STAGE 1 — CROP (YOLO)                                │
             │  detect whole-heart box in the full image             │
             └──────────────────────────┬───────────────────────────┘
                                        │ square bounding box
             ┌──────────────────────────┴───────────────────────────┐
             │  STAGE 2 — CROP + RESAMPLE                             │
             │  crop each modality to the box, resize to 256×256,     │
             │  apply per-channel normalization → _0000.._0003        │
             └──────────────────────────┬───────────────────────────┘
                          ┌──────────────┴───────────────┐
       ┌──────────────────┴─────────┐        ┌───────────┴───────────────────┐
       │ STAGE 3 — LV SEG (nnUNet)  │        │ STAGE 4 — RVIP (YOLO)         │
       │ myocardium mask @256       │        │ 2 insertion points @256       │
       └──────────────────┬─────────┘        └───────────┬───────────────────┘
                          └──────────────┬───────────────┘
             ┌──────────────────────────┴───────────────────────────┐
             │  STAGE 5 — COMBINE + RESAMPLE BACK                     │
             │  merge masks, invert the crop+resize to return to      │
             │  ORIGINAL spacing (proper transforms)                  │
             └──────────────────────────┬───────────────────────────┘
             ┌──────────────────────────┴───────────────────────────┐
             │  STAGE 6-8 — ANALYSIS (in original spacing)           │
             │  • Dice / segmentation metrics vs GT                    │
             │  • MD / FA on the underlying physical values            │
             │  • Helix-Angle-Pitch (HAP) via CarDpy DTI recon         │
             │  • inter-reader study (Hannum vs Cork readers)          │
             └────────────────────────────────────────────────────────┘
```

**Why crop first:** the heart is a small fraction of the full FoV. Cropping to a square ROI
and resampling to 256×256 gives the segmentation model a consistent, zoomed-in input and is
the drop-in replacement for the old Stage-1 nnUNet crop model.

---

## 2. Repository & branch layout

Two active branches hold **different** parts of the project (this trips people up):

| Branch | Holds | Notes |
|---|---|---|
| `local` | data-prep scripts (`data_image_mask_making/`), `nnunetv2/` core | slice-cap fix, normalization fix, `polylr` PyTorch fix live here |
| `crop-alignment-and-hap-fixes` | `yolo_pipeline.py`, `train_yolo_models.ipynb`, `inference.ipynb`, `hap.ipynb`, `hap_analysis.py` | the YOLO detectors, the full inference pipeline, and HAP |

- Remote: `origin` → `https://github.com/sash9966/nnUNet_cDTI.git` (moved from `nnUNet.git`).
- The server pulls per-branch; a fix is only visible on the branch it was committed to.

---

## 3. Data preparation (`data_image_mask_making/`)

Three scripts, one per nnUNet dataset. Each walks the raw acquisition tree, quality-filters,
and writes nnUNet-format `imagesTr`/`labelsTr` (4 channels per case).

| Script | Builds | Purpose |
|---|---|---|
| `NewFALVonly.py` | **Dataset100** `_HannumSmarthHealthDataLV` | LV myocardium → nnUNet segmentation |
| `NewFAIPOnly.py` | **Dataset105** `_HannumSmartHealthDataIPs` | RV insertion points → YOLO |
| `NewCropPrep.py` | **Dataset110** `_HannumSmarthHealthDataCrop` | whole-heart box → YOLO |

> Note the folder-name spelling: LV & crop use `SmarthHealth` (typo), IPs uses `SmartHealth`.
> Paths/globs must match exactly.

### Source data structure
```
<root>/Hannum(or Cork)/Volunteer_XX/Distortion_Corrected/DiVO_.._..(or MDDW..)/
    Detailed_Information.xlsx          # per-slice quality + crop coords + IP coords
    05_Segmentation_Images_CI/         # Cropped_*_Image_Slice_NNN.nii  (avg, MD, E1, FA)
    06_Segmentation_Masks_CI/          # Cropped_Segmentation_Slice_NNN.nii (LV/IP labels)
    02_Crop_Masks/ , 03_Segmentation_Images/  # (crop dataset: Square_Crop_Mask_*, avg image)
```

### Slice selection
- Every DiVO/MDDW folder holds **6–12 short-axis slices** (not 3).
- Slices are **discovered dynamically** from the mask folder and looped over; each slice is
  kept only if `Detailed_Information.xlsx` marks it `Good Image Quality`.
- Filenames are zero-padded `Slice_{i:03d}` so slices ≥10 resolve correctly.
- Result on the Hannum set: **96 → ~243 good slices (≈2.5×)**; volunteers whose slices 1–3
  were poor quality (e.g. 08, 10) are no longer dropped entirely.

### Channels / modalities (why each helps)
Each case is stored as four nnUNet channels:

| Channel | Modality | Rationale |
|---|---|---|
| `_0000` | Average diffusion-weighted image | Dominant anatomical/structural signal — carries most of the segmentation information |
| `_0001` | Mean Diffusivity (MD) | Tissue characterization; distinguishes myocardium/blood/effusion |
| `_0002` | Combined primary eigenvector `E1_x + E1_y` (in-plane fiber projection) | Fiber orientation cue — helps at myocardial boundaries and at the RV/LV junction (insertion points) |
| `_0003` | Fractional Anisotropy (FA) | Anisotropy / tissue integrity; already normalized [0,1] by definition |

### Normalization (per channel, applied in the prep script)
nnUNet is configured with **`noNorm`** (see §4), so **what the prep script writes is exactly
what the network sees** — the prep-side normalization is the real normalization.

| Channel | Transform | Range | Function |
|---|---|---|---|
| `_0000` avg | per-image min-max | [0,1] | `normalize_image` |
| `_0001` MD | fixed `/4` (physical MD ceiling) | [0,1] | `normalise_MD` |
| `_0002` eigvec | **fixed `/√2`** | [0,1] | `normalise_eigenvector` |
| `_0003` FA | none (already [0,1]) | [0,1] | — |

**Why `/√2` for the eigenvector (not `/2`, not min-max):** `E1` is a *unit* vector
(`x²+y²+z²=1`), so `E1_x + E1_y` is geometrically capped at **√2 ≈ 1.414**, not 2 — the
components can't both reach 1. Measured across all 306 source images: the sum hits 1.4142 in
**every** image, and 0/306 ever reach 2. A true 0 (a purely through-plane voxel) exists in
only ~46% of images and is background noise, so **min-max would anchor on an unreliable,
noise-driven floor**; the fixed √2 ceiling is present everywhere → stable and reproducible.
**This normalization must be identical in the prep scripts and in `inference.ipynb`** (it is,
as of the latest commits) — otherwise `noNorm` gives a train↔inference mismatch.

---

## 4. nnUNet (LV segmentation)

- **Datasets:** 100 (LV, the one actively trained), plus 105/110 exist but are now served by
  YOLO in the pipeline.
- **`noNorm`:** set in `dataset.json` `channel_names` because we normalize each modality in
  the prep script (informed, physical normalizations). Consequence: **exact train↔inference
  normalization consistency is mandatory** (there is no z-score to absorb differences).
- **Trainer:** `nnUNetTrainerDA5_100epochs` — DA5 = heavy data augmentation, chosen to
  increase data variability given the modest dataset size (100 epochs).
- **Encoder:** **ResEnc-M** (`nnUNetResEncUNetMPlans`). ResEnc-**L** was tried first and
  overfit hard (huge model on a few-hundred-slice 2D set → train loss floors fast, validation
  Dice rises then collapses into an "upside-down parabola"). ResEnc-M ≈ standard UNet capacity
  and is the right tier for this data size.
- **Splits:** ~**80 : 10 : 10** train/val/test.
  - **⚠ CONFIRM** split-file location: `nnUNet_preprocessed/Dataset100_.../splits_final.json`
    (human owner to confirm exact path).
  - **Critical:** `splits_final.json` is written **once and reused**, and preprocessing is
    **skipped if the folder exists**. After changing the raw data you MUST regenerate them,
    or training silently runs on the *old* case count.

### Run order (LV)
```bash
conda activate nnunet     # NOT base — base uses another user's nnUNet install
rm -rf nnUNet_preprocessed/Dataset100_...          # force fresh splits + preprocessing
nnUNetv2_plan_and_preprocess -d 100 -pl nnUNetPlannerResEncM --verify_dataset_integrity
nnUNetv2_train 100 2d 2 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerDA5_100epochs
#                    ^fold 2
```
Sanity check in the log: `This split has NNN training and MMM validation cases` should reflect
the **larger** dataset (~194/49 for 243 slices), not the old ~77/19.

### Timing / reporting
Per run under `nnUNet_results/Dataset.../fold_X/`: `progress.png` (panel 2 = epoch duration),
timestamped `training_log_*.txt`, and `checkpoint_*.pth` (carry `epoch_start/end_timestamps`
for total time). Best model = `checkpoint_best.pth` (peak pseudo-Dice), so a late collapse
doesn't cost you the usable weights.

---

## 5. YOLO detection models (`yolo_pipeline.py`, `train_yolo_models.ipynb`)

Two detectors replace the two nnUNet *detection* stages (crop + IPs); nnUNet is kept only for
LV segmentation.

| Model | Dataset | Output |
|---|---|---|
| **CROP** | Dataset110 | one whole-heart square box (`Heart=1`) |
| **RVIP** | Dataset105 | two RV insertion points (anterior/inferior) |

- **Channels:** baseline `(0,0,0)` = first contrast replicated to RGB. Multi-contrast
  extension `(0,1,2)` trains a separate `rvip_mc` model (§7 of the notebook).
- **RVIP is single-class:** both points detected as one `insertion_point` class; anterior vs
  inferior assigned **by geometry** afterwards (`assign_anterior_inferior`, anterior = higher
  row; `RVIP_FLIP_ASSIGN` inverts). The 2-class version failed (mAP≈0.06) because the points
  differ by *position*, not local appearance.
- **mAP is not the pipeline metric.** The pipeline takes the top-2 points; §8 evaluates the
  real metric — **per-point pixel distance** vs GT on the held-out test set.
- **Adapters for the pipeline:** `detect_crop_box_from_path → crop_box_to_mask`,
  `detect_rvips_from_path → rvips_to_mask` (labels 2/3). Use the **same `CHANNELS` at inference
  as at training**.

### Force a clean retrain (YOLO caches aggressively)
```bash
rm -f  yolo_datasets/*/labels/*.cache                 # stale label caches
rm -rf yolo_datasets/{rvip,rvip_mc,crop}/images \
       yolo_datasets/{rvip,rvip_mc,crop}/labels        # stale exported PNGs (prevents split leakage)
```
Then run `train_yolo_models.ipynb` **top-to-bottom** (the *export* cells rebuild the PNGs from
the new nii; training alone would reuse old PNGs). Confirm the printed sample count grew
(~2.5×) and `train: Scanning ... N images` shows N ≫ 62.

---

## 6. Full inference pipeline (`inference.ipynb` → target for a clean rebuild)

The current `inference.ipynb` is the most complete end-to-end pipeline but is crowded
(290-line cells, experimental clutter). The intended flow, stage by stage:

1. **Crop (YOLO)** — replaces the current nnUNet crop cell. `detect_crop_box_from_path` → box.
2. **Crop + resample** — crop each modality to the box, resize to 256×256, apply the
   per-channel normalization from §3 → `_0000.._0003`.
3. **LV segmentation (nnUNet)** — Dataset100 model on the 256 crop.
4. **RVIP (YOLO)** — replaces the current nnUNet IP cell. `detect_rvips_from_path` →
   `rvips_to_mask` (labels 2/3).
5. **Combine + resample back to ORIGINAL spacing** — merge LV (=1) + IPs (=2/3) and invert the
   crop+resize with the proper transforms so masks land back on the native-resolution image.
   **This step is essential for HAP** (the mask must align with the eigenvector field computed
   from the raw data). Verify where the un-crop happens (currently ambiguous — may be inside
   `eval_util.process_folders`).
6. **Segmentation metrics** — Dice etc. vs GT (`eval_util.process_folders`).
7. **Physical-value comparison** — pull the **original images** (from the folder the LV model
   was trained on, for a fair comparison) and compare MD / FA under the predicted vs GT masks.
8. **HAP** — §8 below.
9. **Inter-reader study** — Hannum reader vs Cork reader (§9 below).

- **⚠ CONFIRM** dataset IDs: the current notebook config uses `crop=62, lv=60, ips=61`
  (older FIMH datasets). Point these at the currently trained models (100 for LV, plus the
  YOLO weights) before running.
- **⚠ CONFIRM** the exact "original data" folder to pull physical values from — must be the
  same source the LV model trained on, for a fair comparison.
- Test set: driven from `imagesTs`/`labelsTs` (and/or the prepared `TestData` folders); the
  raw `Distortion_Corrected` images provide the native-spacing modalities for §7–8.

---

## 7. HAP — Helix-Angle-Pitch analysis (`hap.ipynb`, `hap_analysis.py`)

- **What it is:** for each slice, fit a line of helix angle vs endo→epi wall depth; the slope
  is the **HAP** (helix-angle pitch), the endpoint difference is the **HAR** (range). A good
  automatic segmentation should give HAP/HAR close to the GT segmentation's.
- **Dependency:** CarDpy (`cardpy/`) does the DTI reconstruction. HAP needs the **primary
  eigenvector field (E1)** from `DTI_recon` on the **raw diffusion data (bvals/bvecs)** — this
  is *not* produced by the segmentation pipeline, so per case you also load the raw series and
  reconstruct.
- **Refactored module:** `hap_analysis.py` packages the messy `hap.ipynb` prototype:
  - `run_hap_for_mask(myocardial_mask, eigenvectors, ha_key="HASF", ...)` — CarDpy `cDTI_recon`
    + per-slice pitch. Use **HASF** (spatially-filtered HA), not raw HA (raw has
    polarity-flipped border voxels → artificially shallow pitch).
  - `compare_gt_pred(gt_mask, pred_mask, eigenvectors)` — the drop-in for the "repeat for
    predicted mask / compare" TODO; returns per-slice HAP/HAR diffs.
- **Integration plan (next step):** keep `hap.ipynb`'s architecture but feed it the **inference
  LV mask** where it currently uses the GUI/GT mask — run HAP once for GT, once for the
  predicted mask (both in **original spacing**), then compare. Same code path, two masks.

---

## 8. Evaluation & inter-reader study

- **Test cases:** two held-out cohorts — **Hannum** and **Cork** — each with manual GT masks.
- **Automatic vs manual:** the pipeline's predicted masks vs the GT masks (Dice, MD, FA, HAP).
- **Inter-reader:** Hannum-reader GT vs Cork-reader GT via the same `process_folders` machinery
  (`inference.ipynb` cell 11, currently commented) — quantifies human-vs-human variability as
  the reference band the automatic method should fall within.
- All comparisons are done in **original spacing** on the underlying physical values.

---

## 9. Reproduction checklist (end to end)

1. **Prep** — set paths in the three `New*.py` scripts (LV paths OK; ⚠ point IP/crop at
   Smart_Health Dataset105/110), run each → regenerates `nnUNet_raw` datasets with all slices
   and the §3 normalizations.
2. **nnUNet LV** — regenerate `dataset.json` (with `noNorm`), delete stale
   `nnUNet_preprocessed/Dataset100`, `plan_and_preprocess` with ResEnc-M, train fold 2 (§4).
3. **YOLO** — copy Dataset105/110 to the server, clear caches + old exports, run
   `train_yolo_models.ipynb` top-to-bottom (§5).
4. **Inference** — point dataset IDs / weights at the trained models, run the pipeline (§6)
   over `imagesTs`/`labelsTs` for Hannum + Cork.
5. **Analysis** — Dice/MD/FA + HAP (`compare_gt_pred`) + inter-reader (§7–8).

---

## Known issues & fixes (fingerprints)

| Symptom | Cause | Fix |
|---|---|---|
| Pseudo-Dice **rises then collapses** ("upside-down parabola"), train loss floors in ~5 epochs | Model too big for the data (ResEnc-**L** on a few-hundred-slice 2D set) — variance, not a leak (a leak *inflates* val) | Use **ResEnc-M** (or standard UNet); confirm the dataset actually grew |
| Same parabola after "fixing" the data | Stale `splits_final.json` / skipped preprocessing → still the old case count | `rm -rf nnUNet_preprocessed/DatasetXXX`, re-`plan_and_preprocess`; check the "NNN training" line |
| Only 3 slices per volunteer used | `for i in range(1,4)` hard-cap in the prep scripts | dynamic slice discovery + `Slice_{i:03d}` (fixed) |
| `'NoneType' object is not callable` in `plan_and_preprocess` | Wrong conda env → running **another user's** nnUNet install (traceback path `/home/lethu/...`) | `conda activate nnunet`; verify `which` + `nnunetv2.__file__` point to `/home/sastocke/nnUNet` |
| `LRScheduler.__init__() takes 2–3 positional args but 4 were given` | PyTorch ≥2.4 removed `verbose`; `PolyLRScheduler` passed it | drop the trailing `False` in `nnunetv2/training/lr_scheduler/polylr.py` (done on `local`) |
| YOLO "doesn't see" new data | ultralytics label `.cache` + stale exported PNGs | delete caches + exported `images/labels`, re-run the export cells |
| Eigenvector channel off between train & inference | `noNorm` + prep used min-max / inference used `/2` | fixed **`/√2`** in both (done) |
| Values "all over the place" in Slicer | prep normalization was commented out (`# Use nnUnet normaliser!`) | re-enabled per-channel normalization (done) |

---

## Open TODOs

- [ ] Rebuild a clean `full_pipeline.ipynb` around the YOLO adapters + `hap_analysis`, with an
      explicit **un-crop → original spacing** step (so HAP aligns with the eigenvector field).
- [ ] Reconcile inference dataset IDs (60/61/62) → current trained models (100 + YOLO weights).
- [ ] Pin the exact "original data" source folder for the fair physical-value comparison.
- [ ] Wire `compare_gt_pred` into the pipeline (GT mask → predicted mask, same code path).
- [ ] Confirm HAP uses **HASF** and the mask is in original spacing before fitting.
- [ ] Re-generate all three datasets after the `/√2` normalization change before the next runs.
