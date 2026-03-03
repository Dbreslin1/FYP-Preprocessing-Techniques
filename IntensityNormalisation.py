# ==========================================================
# HU window + foreground z-score + ROI cropping
#
# This script creates a NEW nnU-Net raw dataset where:
#   1) HU windowing is applied [-200, 500]
#   2) Foreground-only z-score normalisation is used
#   3) Image + label are cropped to body ROI
#
# Labels remain multi-class (values unchanged).
# ==========================================================

import json
from pathlib import Path
import numpy as np
import nibabel as nib

# -------------------------
# Configuration
# -------------------------

srcID = 1   # Baseline dataset (Dataset001_ImageTBAD)
dstID = 2   # New dataset (Dataset002_ImageTBAD_HUwin_fgZ_crop)

HU_LO = -200.0
HU_HI = 500.0
CROP_MARGIN = 16

nnUNet_raw = Path("/content/nnUNet_raw")

SRC = nnUNet_raw / "Dataset001_ImageTBAD"
DST = nnUNet_raw / "Dataset002_ImageTBAD_HUwin_fgZ_crop"

(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------

def bbox_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mins, maxs

def crop_with_margin(arr, mins, maxs, margin):
    mins2 = np.maximum(mins - margin, 0)
    maxs2 = np.minimum(maxs + margin, np.array(arr.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mins2, maxs2))
    return arr[sl], sl

def hu_window_and_fg_zscore(img):

    img = img.astype(np.float32)

    # 1) HU window
    img_clipped = np.clip(img, HU_LO, HU_HI)

    # 2) Foreground mask (remove air / padding dominance)
    fg_mask = img_clipped > -150.0
    if fg_mask.sum() < 1000:
        fg_mask = np.ones_like(img_clipped, dtype=bool)

    # 3) Foreground-only z-score
    mu = img_clipped[fg_mask].mean()
    sigma = img_clipped[fg_mask].std() + 1e-8

    out = np.zeros_like(img_clipped, dtype=np.float32)
    out[fg_mask] = (img_clipped[fg_mask] - mu) / sigma
    out[~fg_mask] = 0.0

    return out, fg_mask

# -------------------------
# Process ALL training cases
# -------------------------

img_files = sorted((SRC / "imagesTr").glob("*.nii.gz"))

if len(img_files) == 0:
    raise FileNotFoundError(f"No images found in {SRC / 'imagesTr'}")

written = 0

for img_path in img_files:

    case_id = img_path.name.replace("_0000.nii.gz", "")
    lab_path = SRC / "labelsTr" / f"{case_id}.nii.gz"

    if not lab_path.exists():
        raise FileNotFoundError(f"Missing label for {case_id}")

    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.int16)

    # Apply intensity preprocessing
    img_norm, fg_mask = hu_window_and_fg_zscore(img)

    # ROI crop using foreground mask
    bb = bbox_from_mask(fg_mask)
    if bb is None:
        img_crop = img_norm
        lab_crop = lab
    else:
        mins, maxs = bb
        img_crop, sl = crop_with_margin(img_norm, mins, maxs, CROP_MARGIN)
        lab_crop = lab[sl]

    # Save outputs
    out_img = nib.Nifti1Image(img_crop, affine=img_nii.affine, header=img_nii.header)
    out_img.set_data_dtype(np.float32)

    out_lab = nib.Nifti1Image(lab_crop, affine=lab_nii.affine, header=lab_nii.header)
    out_lab.set_data_dtype(np.int16)

    nib.save(out_img, str(DST / "imagesTr" / img_path.name))
    nib.save(out_lab, str(DST / "labelsTr" / lab_path.name))

    written += 1

print(f"Wrote {written} cases to {DST.name}")

# -------------------------
# Write dataset.json
# -------------------------

src_json = json.load(open(SRC / "dataset.json", "r"))
dst_json = dict(src_json)

dst_json["name"] = f"{src_json.get('name','ImageTBAD')}_HUwin_fgZ_crop"
dst_json["file_ending"] = ".nii.gz"
dst_json["numTraining"] = written
dst_json["numTest"] = 0

with open(DST / "dataset.json", "w") as f:
    json.dump(dst_json, f, indent=2)

print("dataset.json written with", written, "cases")
print("HU window:", (HU_LO, HU_HI))
print("Crop margin:", CROP_MARGIN)