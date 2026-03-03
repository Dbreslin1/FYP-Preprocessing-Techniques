#Percentile clipping and foreground only z score normalisation.

#I made this script to create a new nnuNet raw dataset where the iamges are modified in the following way:
# 1. Foreground mask is estimated to reduce air domination in the images.
# 2. Intensities are clipped using percintiles to remove outliers.
# 3. Z score normalisation is applied using the foreground voxels only and not the background.
# 4. Voxels outside the foreground are set to 0.

#The labels are left unchanged.

import json
import shutil 
from pathlib import Path
import numpy as np
import nibabel as nib

#Configuration
srcID = 552 #baseline dataset ID
dstID = 554 #new dataset ID

HU_LO = -200.0 #from testing air is around -2000 at the lowest so this is the cutoff 
HU_HI = 500.0

nnUNet_raw = Path("/content/drive/MyDrive/FYP_nnUNet/nnUNet_raw")  
SRC = nnUNet_raw / f"Dataset{srcID:03d}_ImageTBAD_subX"
DST = nnUNet_raw / f"Dataset{dstID:03d}_ImageTBAD_subX_clipz"


(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)

#Copying the labels but not changing the masks
for lab_path in sorted((SRC / "labelsTr").glob("*.nii.gz")):
    shutil.copy2(lab_path, DST / "labelsTr" / lab_path.name)

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

    return out

#processing the images
for img_path in sorted((SRC / "imagesTr").glob("*.nii.gz")):
    nii = nib.load(str(img_path))

    #load image and change it from float 64 to 32 cause memory is an issue with images
    img = nii.get_fdata().astype(np.float32)

    # Apply normalisation
    img_norm = hu_window_and_fg_zscore(img)

    # Save output image preserving geometry 
    out_nii = nib.Nifti1Image(img_norm, affine=nii.affine, header=nii.header)
    out_nii.set_data_dtype(np.float32)

    nib.save(out_nii, str(DST / "imagesTr" / img_path.name))

# Write dataset.json for the new dataset 
src_json = json.load(open(SRC / "dataset.json", "r"))
dst_json = dict(src_json)

dst_json["name"] = f"{src_json.get('name','ImageTBAD')}_clipz"
dst_json["file_ending"] = ".nii.gz"

with open(DST / "dataset.json", "w") as f:
    json.dump(dst_json, f, indent=2)

print("Created:", DST)
print("✅ HU window:", (HU_LO, HU_HI))
print("Labels unchanged; images normalised with clip + foreground z score.")
