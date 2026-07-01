"""
yolo_pipeline.py
================

Standalone, notebook-facing YOLO side-pipeline for the cDTI LV / RVIP project.

Why this module exists
----------------------
A reviewer asked us to replace the *Stage-1 nnUNet whole-heart crop* (and,
ideally, the RVIP detection) with a cheaper detector. YOLO is a natural fit:
the heart-crop is a single bounding box, and the two RV insertion points (RVIPs)
are well modelled as two *tiny* boxes whose centers we read off.

Design principle
----------------
YOLO here is a drop-in *emitter of the same artifacts the current nnUNet stages
already produce*, so the existing evaluation code (``eval_util.py``,
``calculate_scale_factor``, ``upscale_segmentation_to_original``) keeps working
**unchanged**:

* ``detect_crop_box``  -> a *square* (x_min, x_max, y_min, y_max) box, squared
  with the *exact same* logic as ``make_square_crop`` in ``inference.ipynb``.
* ``crop_box_to_mask`` -> a binary mask byte-for-byte compatible with the
  "cleaned" Stage-1 crop mask that the rest of the pipeline consumes.
* ``detect_rvips``     -> two points (or ``None`` on low confidence -> a *clean*
  failure instead of a wild 1250 mm outlier).
* ``rvips_to_mask``    -> disks written as labels **2** (anterior) and **3**
  (inferior), matching the combined-mask convention from
  ``process_mask_slices`` (label 1 = LV, 2 = anterior RVIP, 3 = inferior RVIP).

RVIP detection uses the **tiny-bbox + center** formulation (chosen over a single
keypoint): a box carries spatial extent and a confidence score, so a weak
detection lands *near* the right region and can be *rejected* rather than thrown
to a random corner of the image.

Coordinate conventions  (read this once, it removes all the confusion)
----------------------------------------------------------------------
The repo's ``make_square_crop`` does ``coords = np.argwhere(mask > 0)`` then
``x_min, y_min = coords.min(axis=0)``. So throughout *this* repo:

    "x"  ==  array axis 0  ==  row    (vertical)
    "y"  ==  array axis 1  ==  col    (horizontal)

YOLO, by contrast, speaks in image pixels where:

    YOLO "x" (cx, x1, x2)  ==  column  (horizontal)
    YOLO "y" (cy, y1, y2)  ==  row     (vertical)

Every conversion below is explicit about this swap. Boxes and points returned by
this module are always in the **repo convention** (x = row/axis-0,
y = col/axis-1) so they slot straight into the existing code.

Dependency
----------
``ultralytics`` (YOLOv8) is required for training/inference but is imported
*lazily* inside the functions that need it, so you can import this module (and
use the pure-numpy adapters / label exporters) without it installed.
Install with: ``pip install ultralytics`` (pulls torch, already present for
nnUNet). The default ``yolov8n.pt`` weights download on first use.
"""

from __future__ import annotations

import os
import glob
from typing import Iterable, Optional, Sequence

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Class-id conventions
# ---------------------------------------------------------------------------
# Two *separate* YOLO models are assumed (one for the crop, one for the RVIPs),
# matching the plan. If you ever unify them into a single model, shift the RVIP
# class ids to 1 and 2 and reserve 0 for the heart.
CROP_CLASS_ID = 0          # single class for the whole-heart crop model
ANTERIOR_CLASS_ID = 0      # RVIP model, class 0 -> anterior insertion point
INFERIOR_CLASS_ID = 1      # RVIP model, class 1 -> inferior insertion point

# Output combined-mask labels (matches process_mask_slices in data_image_mask_making/)
LV_LABEL = 1
ANTERIOR_LABEL = 2
INFERIOR_LABEL = 3


# ===========================================================================
# Geometry helpers (pure numpy, no YOLO needed)
# ===========================================================================
def make_square_crop(mask: np.ndarray, target_size: int = 256):
    """Square bounding box around the non-zero pixels of ``mask``.

    This is an exact re-implementation of ``make_square_crop`` from
    ``inference.ipynb`` (cell 2) so YOLO crops are squared *identically* to the
    nnUNet pipeline. Returns ``(x_min, x_max, y_min, y_max, scale_factor)`` in
    repo convention (x = axis 0 / row, y = axis 1 / col).
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("make_square_crop: mask is empty (no non-zero pixels).")
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1  # exclusive

    side_length = int(max(x_max - x_min, y_max - y_min))

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x_min = int(max(0, center_x - side_length // 2))
    x_max = int(x_min + side_length)
    y_min = int(max(0, center_y - side_length // 2))
    y_max = int(y_min + side_length)

    scale_factor = side_length / target_size
    return x_min, x_max, y_min, y_max, scale_factor


def _box_to_binary_mask(x_min, x_max, y_min, y_max, shape) -> np.ndarray:
    """Render a (repo-convention) box as a uint8 binary mask of ``shape``."""
    mask = np.zeros(shape, dtype=np.uint8)
    x_min = max(0, int(x_min)); y_min = max(0, int(y_min))
    x_max = min(shape[0], int(x_max)); y_max = min(shape[1], int(y_max))
    mask[x_min:x_max, y_min:y_max] = 1
    return mask


def _bbox_repo_to_yolo(x_min, x_max, y_min, y_max, img_h, img_w):
    """Repo box (x=row, y=col, exclusive max) -> normalized YOLO (cx, cy, w, h).

    Remember the swap: YOLO cx is horizontal (col / y-axis here), cy is vertical
    (row / x-axis here).
    """
    row_center = (x_min + x_max) / 2.0
    col_center = (y_min + y_max) / 2.0
    cx = col_center / img_w          # horizontal -> col
    cy = row_center / img_h          # vertical   -> row
    w = (y_max - y_min) / img_w      # width  spans cols
    h = (x_max - x_min) / img_h      # height spans rows
    return cx, cy, w, h


def _bbox_yolo_to_repo(cx, cy, w, h, img_h, img_w):
    """Normalized YOLO (cx, cy, w, h) -> repo box (x_min, x_max, y_min, y_max).

    Inverse of ``_bbox_repo_to_yolo``. Returns integer, exclusive-max bounds.
    """
    col_center = cx * img_w
    row_center = cy * img_h
    box_w = w * img_w
    box_h = h * img_h
    x_min = int(round(row_center - box_h / 2.0))
    x_max = int(round(row_center + box_h / 2.0))
    y_min = int(round(col_center - box_w / 2.0))
    y_max = int(round(col_center + box_w / 2.0))
    return x_min, x_max, y_min, y_max


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] uint8 (matches normalize_image in repo)."""
    image = np.asarray(image, dtype=np.float64)
    lo, hi = image.min(), image.max()
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return (image * 255.0).round().astype(np.uint8)


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume as a numpy array (squeezed to 2D when possible)."""
    data = nib.load(path).get_fdata()
    return np.squeeze(data)


# ===========================================================================
# B1. Label export  (existing ground truth  ->  YOLO format)
# ===========================================================================
# YOLO label files are one ``.txt`` per image, each line:
#     <class_id> <cx> <cy> <w> <h>      (all of cx,cy,w,h normalized to [0,1])
# and images live alongside in a parallel folder. write_data_yaml() ties it
# together.

def nifti_to_yolo_image(nifti_path: str, out_png_path: str) -> None:
    """Normalize a NIfTI slice and save it as an 8-bit PNG for YOLO training."""
    from PIL import Image  # lazy; Pillow ships with ultralytics

    arr = load_nifti(nifti_path)
    png = _normalize_to_uint8(arr)
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    Image.fromarray(png).save(out_png_path)


def crop_mask_to_yolo_label(crop_mask_path: str,
                            out_txt_path: str,
                            class_id: int = CROP_CLASS_ID) -> str:
    """GT square-crop mask -> a single-line YOLO label (the heart box).

    Reads ``Square_Crop_Mask_Slice_*.nii``, squares it with ``make_square_crop``
    (so training targets match exactly what we evaluate against), and writes the
    normalized YOLO line. Returns the written line.
    """
    mask = load_nifti(crop_mask_path)
    img_h, img_w = mask.shape[:2]
    x_min, x_max, y_min, y_max, _ = make_square_crop(mask)
    cx, cy, w, h = _bbox_repo_to_yolo(x_min, x_max, y_min, y_max, img_h, img_w)
    line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w") as fh:
        fh.write(line + "\n")
    return line


def combined_mask_to_rvip_label(combined_mask_path: str,
                                out_txt_path: str,
                                box_px: int = 8,
                                anterior_label: int = ANTERIOR_LABEL,
                                inferior_label: int = INFERIOR_LABEL,
                                anterior_class: int = ANTERIOR_CLASS_ID,
                                inferior_class: int = INFERIOR_CLASS_ID) -> list:
    """GT combined mask (labels 2 & 3) -> YOLO labels for the two RVIPs.

    Each insertion point is a single annotated region; we take its centroid and
    emit a fixed ``box_px`` x ``box_px`` box around it (tiny-bbox formulation).
    A point that is absent in the GT is simply skipped. Returns the written
    lines.
    """
    mask = load_nifti(combined_mask_path)
    img_h, img_w = mask.shape[:2]
    lines = []
    for label, cls in ((anterior_label, anterior_class),
                       (inferior_label, inferior_class)):
        coords = np.argwhere(mask == label)
        if coords.size == 0:
            continue  # this RVIP not annotated on this slice
        # centroid in repo convention (x = row, y = col)
        cx_repo, cy_repo = coords.mean(axis=0)
        half = box_px / 2.0
        x_min, x_max = cx_repo - half, cx_repo + half
        y_min, y_max = cy_repo - half, cy_repo + half
        cx, cy, w, h = _bbox_repo_to_yolo(x_min, x_max, y_min, y_max, img_h, img_w)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return lines


def export_dataset(samples: Iterable[dict],
                   out_dir: str,
                   kind: str = "crop",
                   split_key: str = "split",
                   box_px: int = 8) -> dict:
    """Materialize a YOLO dataset on disk from a list of sample dicts.

    Each ``sample`` is a dict::

        {
            "image_path": "<path to DWI_avg .nii>",      # the training image
            "mask_path":  "<path to GT mask .nii>",      # crop mask OR combined mask
            "name":       "Le_002_..._slice_001",        # unique stem
            "split":      "train" | "val" | "test",      # volunteer-level split
        }

    ``kind`` selects the label exporter: ``"crop"`` (whole-heart square box from a
    ``Square_Crop_Mask``) or ``"rvip"`` (two insertion-point boxes from a combined
    mask). Mirrors the existing volunteer-level split rather than a random one.

    Returns a small summary dict with per-split counts.
    """
    assert kind in ("crop", "rvip"), "kind must be 'crop' or 'rvip'"
    counts = {}
    for s in samples:
        split = s.get(split_key, "train")
        img_out = os.path.join(out_dir, "images", split, s["name"] + ".png")
        lbl_out = os.path.join(out_dir, "labels", split, s["name"] + ".txt")
        nifti_to_yolo_image(s["image_path"], img_out)
        if kind == "crop":
            crop_mask_to_yolo_label(s["mask_path"], lbl_out)
        else:
            combined_mask_to_rvip_label(s["mask_path"], lbl_out, box_px=box_px)
        counts[split] = counts.get(split, 0) + 1
    return counts


def write_data_yaml(out_dir: str, kind: str = "crop") -> str:
    """Write the ``data.yaml`` describing an exported dataset. Returns its path."""
    if kind == "crop":
        names = {CROP_CLASS_ID: "heart"}
    else:
        names = {ANTERIOR_CLASS_ID: "anterior_rvip",
                 INFERIOR_CLASS_ID: "inferior_rvip"}
    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(names.items()))
    yaml_path = os.path.join(out_dir, "data.yaml")
    content = (
        f"path: {os.path.abspath(out_dir)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_block}\n"
    )
    with open(yaml_path, "w") as fh:
        fh.write(content)
    return yaml_path


# ===========================================================================
# B2. Train
# ===========================================================================
def train_yolo(data_yaml: str,
               model: str = "yolov8n.pt",
               epochs: int = 100,
               imgsz: int = 256,
               project: str = "yolo_runs",
               name: str = "crop",
               **kwargs):
    """Thin wrapper over ``ultralytics.YOLO(...).train(...)``.

    Call once for the crop model and once for the RVIP model (with the matching
    ``data.yaml`` and a distinct ``name``). Returns the trained ``YOLO`` object;
    best weights are saved under ``<project>/<name>/weights/best.pt``.
    """
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
               project=project, name=name, **kwargs)
    return yolo


def load_model(weights_path: str):
    """Load a trained YOLO model from a ``.pt`` weights file."""
    from ultralytics import YOLO
    return YOLO(weights_path)


# ===========================================================================
# B3. Inference + adapters  (the notebook-facing API)
# ===========================================================================
def _predict_boxes(model, image: np.ndarray, conf: float = 0.25):
    """Run YOLO on a 2D array and return a list of (cls, confidence, cx,cy,w,h).

    Boxes are returned in *normalized YOLO* coords; callers convert to repo
    convention as needed. The input array is min-max normalized to an 8-bit
    3-channel image (YOLO expects RGB-like input).
    """
    png = _normalize_to_uint8(image)
    rgb = np.stack([png] * 3, axis=-1)  # (H, W, 3)
    results = model.predict(rgb, conf=conf, verbose=False)
    out = []
    img_h, img_w = image.shape[:2]
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls.item())
            confidence = float(b.conf.item())
            # ultralytics xywh is in pixels, x=col(horizontal), y=row(vertical)
            x_c, y_c, bw, bh = (float(v) for v in b.xywh[0].tolist())
            out.append((cls, confidence,
                        x_c / img_w, y_c / img_h, bw / img_w, bh / img_h))
    return out


def detect_crop_box(model, image: np.ndarray, conf: float = 0.25):
    """Detect the whole-heart crop box. Drop-in replacement for Stage-1 nnUNet.

    Picks the highest-confidence detection, then *squares* it with the repo's
    ``make_square_crop`` logic so the result is identical in form to the cleaned
    nnUNet crop. Returns ``(x_min, x_max, y_min, y_max, scale_factor)`` in repo
    convention, or ``None`` if nothing was detected.
    """
    dets = _predict_boxes(model, image, conf=conf)
    dets = [d for d in dets if d[0] == CROP_CLASS_ID]
    if not dets:
        return None
    _, _, cx, cy, w, h = max(dets, key=lambda d: d[1])  # highest confidence
    img_h, img_w = image.shape[:2]
    x_min, x_max, y_min, y_max = _bbox_yolo_to_repo(cx, cy, w, h, img_h, img_w)
    # Square it via a binary mask so the logic is byte-identical to the pipeline.
    raw = _box_to_binary_mask(x_min, x_max, y_min, y_max, image.shape[:2])
    return make_square_crop(raw)


def crop_box_to_mask(box, shape) -> np.ndarray:
    """Square box -> binary uint8 crop mask (same interface as the nnUNet crop).

    ``box`` may be the 5-tuple from ``detect_crop_box`` or a 4-tuple
    ``(x_min, x_max, y_min, y_max)``. The result is consumable directly by
    ``calculate_scale_factor`` / ``upscale_segmentation_to_original`` and can be
    saved to NIfTI to feed the existing evaluation unchanged.
    """
    x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
    return _box_to_binary_mask(x_min, x_max, y_min, y_max, shape)


def detect_rvips(model, image: np.ndarray, conf_thresh: float = 0.25) -> dict:
    """Detect the two RV insertion points.

    Returns ``{"anterior": (x, y) | None, "inferior": (x, y) | None}`` with points
    in repo convention (x = row/axis-0, y = col/axis-1). For each class we keep the
    highest-confidence box above ``conf_thresh`` and take its center; a missing or
    low-confidence point becomes ``None`` (a clean failure rather than a far-off
    outlier).
    """
    dets = _predict_boxes(model, image, conf=conf_thresh)
    img_h, img_w = image.shape[:2]
    result = {"anterior": None, "inferior": None}
    for key, cls in (("anterior", ANTERIOR_CLASS_ID), ("inferior", INFERIOR_CLASS_ID)):
        cands = [d for d in dets if d[0] == cls]
        if not cands:
            continue
        _, _, cx, cy, w, h = max(cands, key=lambda d: d[1])
        # center: cx is col (y here), cy is row (x here)
        x = cy * img_h   # row  / axis 0
        y = cx * img_w   # col  / axis 1
        result[key] = (float(x), float(y))
    return result


def rvips_to_mask(points: dict,
                  shape=(256, 256),
                  radius: int = 3,
                  anterior_label: int = ANTERIOR_LABEL,
                  inferior_label: int = INFERIOR_LABEL) -> np.ndarray:
    """Grow a disk around each detected RVIP -> labelled mask (labels 2 & 3).

    Reproduces the circular GT annotation style so DSC / Hausdorff / center-of-mass
    evaluation in ``eval_util.py`` works against the YOLO output with no changes.
    ``points`` is the dict from ``detect_rvips``; ``None`` points are skipped.

    Tip: to compare points directly instead of via masks, just use the centers
    from ``detect_rvips`` and report center-to-center distance.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    for key, label in (("anterior", anterior_label), ("inferior", inferior_label)):
        pt = points.get(key)
        if pt is None:
            continue
        x, y = pt  # x = row, y = col
        disk = (rows - x) ** 2 + (cols - y) ** 2 <= radius ** 2
        mask[disk] = label
    return mask


# ===========================================================================
# Convenience: discover GT files for export (optional, layout-aware helper)
# ===========================================================================
def find_crop_samples(main_folder: str,
                      annotator: str,
                      image_glob: str = "*Average_Diffusion_Weighted_Image_Slice_*.nii*",
                      crop_mask_name: str = "Square_Crop_Mask_Slice_{slice}.nii"):
    """Best-effort walker that pairs DWI_avg images with their GT square-crop masks.

    Mirrors the directory layout used in ``eval_util.process_folders``
    (``.../<annotator>/<volunteer>/Distortion_Corrected/<divo>/02_Crop_Masks/...``).
    Returns a list of ``sample`` dicts ready for ``export_dataset(kind='crop')``.
    Adjust the globs/patterns to your exact filenames if they differ -- this is a
    starting point, not a hard dependency.
    """
    samples = []
    pattern = os.path.join(main_folder, annotator, "*", "Distortion_Corrected",
                           "*", "02_Crop_Masks", crop_mask_name.replace("{slice}", "*"))
    for crop_mask_path in sorted(glob.glob(pattern)):
        divo_dir = os.path.dirname(os.path.dirname(crop_mask_path))
        slice_token = os.path.basename(crop_mask_path).split("_")[-1].split(".")[0]
        imgs = glob.glob(os.path.join(divo_dir, "**", image_glob), recursive=True)
        imgs = [p for p in imgs if slice_token in os.path.basename(p)]
        if not imgs:
            continue
        volunteer = os.path.basename(os.path.dirname(os.path.dirname(divo_dir)))
        divo = os.path.basename(divo_dir)
        name = f"{annotator}_{volunteer}_{divo}_slice_{slice_token}"
        samples.append({
            "image_path": imgs[0],
            "mask_path": crop_mask_path,
            "name": name,
            "split": "train",  # set per-volunteer split downstream
        })
    return samples


if __name__ == "__main__":  # pragma: no cover - illustrative, not run on import
    print(__doc__)
