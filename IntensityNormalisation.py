#Percentile clipping and foreground only z score normalisation.

#I made this script to create a new nnuNet raw dataset where the iamges are modified in the following way:
#   (A) HU windowing: clip intensities to a clinically meaningful range [-200, 500]
#   (B) Foreground-only z-score normalization: compute mean only in body region
#   (C) ROI cropping: crop image+label to a body bounding box to reduce air/padding
#

#The labels are left unchanged.

import json
import shutil 
from pathlib import Path
import numpy as np
import nibabel as nib

#Configuration
srcID = 1 #baseline dataset ID
dstID = 2 #new dataset ID

HU_LO = -200.0 #from testing air is around -2000 at the lowest so this is the cutoff 
HU_HI = 500.0

CROP_MARGIN =16 #ROI cropping margin

nnUNet_raw = Path("/content/nnUNet_raw")  
SRC = nnUNet_raw / "Dataset001_ImageTBAD"
DST = nnUNet_raw / "Dataset002_ImageTBAD_HUwin"


(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)

#Copying the labels but not changing the masks
for lab_path in sorted((SRC / "labelsTr").glob("*.nii.gz")):
    shutil.copy2(lab_path, DST / "labelsTr" / lab_path.name)

def bbox_from_mask(mask: np.ndarray):
    """
    Compute bounding box (mins, maxs) for a boolean mask.
    mins inclusive, maxs exclusive.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mins, maxs

def crop_with_margin(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray, margin: int):
    """
    Crop an array to bounding box with margin, safely within array bounds.
    Returns cropped array and slice tuple .
    """
    mins2 = np.maximum(mins - margin, 0)
    maxs2 = np.minimum(maxs + margin, np.array(arr.shape))

    sl = tuple(slice(int(a), int(b)) for a, b in zip(mins2, maxs2))
    return arr[sl], sl
def hu_window_and_fg_zscore(img: np.ndarray) -> np.ndarray:
    

    img = img.astype(np.float32)

    # 1) HU windowing
    img_clipped = np.clip(img, HU_LO, HU_HI)

    # 2) Foreground mask
    # Padding/air values are very low (-2000, -1000).
    # Threshold at -150 HU removes most air and padding.
    foreground_mask = img_clipped > -150.0

    # Safety fallback
    if foreground_mask.sum() < 1000:
        foreground_mask = np.ones_like(img_clipped, dtype=bool)

    # 3) Foreground-only z-score
    mu = img_clipped[foreground_mask].mean()
    sigma = img_clipped[foreground_mask].std() + 1e-8

    out = np.zeros_like(img_clipped, dtype=np.float32)
    out[foreground_mask] = (img_clipped[foreground_mask] - mu) / sigma
    out[~foreground_mask] = 0.0

    return out, foreground_mask

#processing the images
img_files = sorted((SRC / "imagesTr").glob("*.nii.gz"))

if len(img_files) == 0:
    raise FileNotFoundError(f"No .nii.gz images found in {SRC / 'imagesTr'}. Check folder name and path.")

for img_path in img_files:
    # Derive matching label path
    case_id = img_path.name.replace("_0000.nii.gz", "")
    lab_path = SRC / "labelsTr" / f"{case_id}.nii.gz"

    if not lab_path.exists():
        raise FileNotFoundError(f"Missing label for {case_id}: expected {lab_path}")

    # Load image and label
    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.int16)  # keep labels as integers

    # 1) Apply HU window + fg z-score 
    img_norm, fg_mask = hu_window_and_fg_zscore(img)

    # 2) ROI cropping based on body/foreground mask bounding box
    bb = bbox_from_mask(fg_mask)
    if bb is None:
        # No crop possible just keep as is
        img_crop = img_norm
        lab_crop = lab
    else:
        mins, maxs = bb
        img_crop, sl = crop_with_margin(img_norm, mins, maxs, margin=CROP_MARGIN)
        lab_crop = lab[sl]  # apply identical crop to label

    # Save image + label 
    out_img = nib.Nifti1Image(img_crop.astype(np.float32), affine=img_nii.affine, header=img_nii.header)
    out_img.set_data_dtype(np.float32)

    out_lab = nib.Nifti1Image(lab_crop.astype(np.int16), affine=lab_nii.affine, header=lab_nii.header)
    out_lab.set_data_dtype(np.int16)

    nib.save(out_img, str(DST / "imagesTr" / img_path.name))
    nib.save(out_lab, str(DST / "labelsTr" / lab_path.name))


# Write dataset.json for the new dataset

src_json = json.load(open(SRC / "dataset.json", "r"))
dst_json = dict(src_json)

#updating name
dst_json["name"] = f"{src_json.get('name','ImageTBAD')}_HUwin_fgZ_crop"
dst_json["file_ending"] = ".nii.gz"

with open(DST / "dataset.json", "w") as f:
    json.dump(dst_json, f, indent=2)

print(" Created:", DST)
print(" HU window:", (HU_LO, HU_HI))
print(" Added ROI crop with margin:", CROP_MARGIN)
print(" Labels unchanged in class values (multi-class preserved); spatially cropped to match images.")
