"""
Build splits_final.json for a dataset that MERGES two cohorts:

  * SmartHealth   -> filenames 'Hannum_Volunteer_XX_...'          (volunteers ~6-52)
  * DirVsAverages -> filenames 'DirVsAvg...'/'DirVsAverages...'    (volunteers 1-12)

Strategy (test set is SmartHealth-only, so validation should mirror it):
  - SmartHealth is cross-validated by volunteer number into 5 folds (train/val).
  - ALL DirVsAverages cases go into EVERY fold's *training* set (auxiliary data to
    enrich variety); validation stays pure SmartHealth == the test distribution.

The cohort detector catches BOTH the 'Avg' and 'Averages' spellings, case-insensitive,
so no DirVsAverages case can slip through as SmartHealth or be dropped.
"""
import os
import json
import re


def specific_split_json_file(dataset_name):
    dataset_folder = os.path.join("/home/sastocke/nnUNet/nnUNet_raw", dataset_name, "imagesTr")

    # unique case ids (strip the _000X channel suffix)
    all_files = sorted({re.sub(r'_000[0-3]\.nii\.gz$', '', f)
                        for f in os.listdir(dataset_folder) if f.endswith(".nii.gz")})

    # ---- split the two cohorts by prefix; catch BOTH 'Avg' AND 'Averages' spellings ----
    def is_dirvsavg(fname):
        fl = fname.lower()
        return ('dirvsavg' in fl) or ('dirvsaverages' in fl)   # both spellings, case-insensitive

    dva_files = [f for f in all_files if is_dirvsavg(f)]        # Direction-vs-Averages -> ALL into training
    sh_files  = [f for f in all_files if not is_dirvsavg(f)]    # SmartHealth -> volunteer-based CV

    def vol_num(f):
        m = re.search(r'Volunteer_(\d+)_', f)
        return int(m.group(1)) if m else None

    print(f"cohorts: {len(sh_files)} SmartHealth + {len(dva_files)} DirVsAvg = {len(all_files)} cases")
    print(f"DirVsAvg volunteers (all -> training): "
          f"{sorted({v for f in dva_files if (v := vol_num(f)) is not None})}")

    # ---- SmartHealth CV folds (by volunteer number) ----
    validation_groups = [
        list(range(6, 15)),    # Fold 0
        list(range(15, 23)),   # Fold 1
        list(range(23, 31)),   # Fold 2
        list(range(31, 39)),   # Fold 3
        list(range(39, 46)),   # Fold 4
    ]
    test_volunteers      = list(range(46, 53))                       # held out (SmartHealth)
    train_val_volunteers = [v for v in range(6, 53) if v not in test_volunteers]

    def sh_with(vols):
        return [f for f in sh_files if any(f"Volunteer_{v:02d}_" in f for v in vols)]

    splits = []
    for fold_idx, val_vols in enumerate(validation_groups):
        train_vols = [v for v in train_val_volunteers if v not in val_vols]
        sh_train, sh_val = sh_with(train_vols), sh_with(val_vols)

        train_ids = sh_train + dva_files      # DirVsAvg in EVERY fold's training
        val_ids   = sh_val                    # validation stays pure SmartHealth (== test distribution)

        # leakage check: no SmartHealth volunteer in both train and val
        tr_v = {vol_num(f) for f in sh_train}
        va_v = {vol_num(f) for f in sh_val}
        overlap = tr_v & va_v
        print(f"Fold {fold_idx}: {len(train_ids)} train ({len(sh_train)} SH + {len(dva_files)} DVA), "
              f"{len(val_ids)} val | " + (f"LEAKAGE {sorted(overlap)}" if overlap else "no leakage"))
        splits.append({"train": train_ids, "val": val_ids})

    # any case never placed in a fold? (SmartHealth volunteer outside 6-45, or odd naming)
    placed = {c for s in splits for c in s["train"] + s["val"]}
    unplaced = [f for f in all_files if f not in placed]
    if unplaced:
        print(f"\n!! {len(unplaced)} cases never assigned (check numbering): {unplaced[:10]}")

    out = os.path.join("/home/sastocke/nnUNet/nnUNet_preprocessed", dataset_name, "splits_final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"\nTest (held out): SmartHealth volunteers {test_volunteers}\nSaved {out}")


if __name__ == "__main__":
    for dataset_name in ['Dataset111_HannumSmartHealthandDirVsAvgCrop']:
        specific_split_json_file(dataset_name)
