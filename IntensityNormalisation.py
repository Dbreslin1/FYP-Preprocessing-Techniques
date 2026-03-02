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

nnUNet_raw = Path("/content/nnUNet_raw")  
SRC = nnUNet_raw / f"Dataset{srcID:03d}_ImageTBAD_subX"
DST = nnUNet_raw / f"Dataset{dstID:03d}_ImageTBAD_subX_clipz"


(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)

#Copying the labels but not changing the masks
for lab_path in sorted((SRC / "labelsTr").glob("*.nii.gz")):
    shutil.copy2(lab_path, DST / "labelsTr" / lab_path.name)

def clip_and_foreground_zscore(img:np.ndarray,
                               p_lo: float = 0.5,
                               p_hi: float = 99.5,
                               fg_percentile: float = 10.0) -> np.ndarray:
        
    """
    img: 3D float 32 image colume
    p lo and p hi are percintiles for robust clipping
    fg percentile is the lw percentile used to define a foreground ish looking mask
    It returns a float 32 normlised image 
    
    """ 

    # esimating the foreground mask
    # Thoracic CT contains a lot of air which can dominate the intensity distribution and make normalisation less effective.
    # To stop this im gonna use fg_threshold = 10th percentile intentsity
    # I'll also have foreground = voxels greater than this threshold
    # This tend to exclude most air while keeping the body tissues and cells taht I want which are important.

    fg_threshold = np.percentile(img, fg_percentile)
    foreground_mask = img > fg_threshold

    #safety check for if the mask is too small then just treat the whole volume as the foreground
    if foreground_mask.sum() < 1000:
        foreground_mask = np.ones_like(img, dtype=bool)

    #2 clipping intensities
    #computed within foreground voxels only so that the bonds reflect tissue intensities and not air or other outliers
    vals = img[foreground_mask]
    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)
    img_clipped = np.clip(img, lo, hi)

    #z score normalisation using foreground voxels only
    mu = img_clipped[foreground_mask].mean()
    sigma = img_clipped[foreground_mask].std() + 1e-8 # added small value to avoid division by zero

    out = np.zeros_like(img_clipped, dtype=np.float32)
    out[foreground_mask] = (img_clipped[foreground_mask] - mu) / sigma

    #outside foreground set to 0
    out[~foreground_mask] = 0.0

    return out

#processing the images
for img_path in sorted((SRC / "imagesTr").glob("*.nii.gz")):
    nii = nib.load(str(img_path))

    #load image and change it from float 64 to 32 cause memory is an issue with images
    img = nii.get_fdata().astype(np.float32)

    # Apply normalisation
    img_norm = clip_and_foreground_zscore(img, p_lo=0.5, p_hi=99.5, fg_percentile=10.0)

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
print("Labels unchanged; images normalised with clip + foreground z score.")
