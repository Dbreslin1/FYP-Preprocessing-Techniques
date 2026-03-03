

import json
from pathlib import Path
import numpy as np
import nibabel as nib

print("===  Dataset002 (HU window + crop ONLY) ===")

HU_LO = -200.0
HU_HI = 500.0
CROP_MARGIN = 16

nnUNet_raw = Path("/content/drive/MyDrive/FYP_nnUNet/nnUNet_raw")

SRC = nnUNet_raw / "Dataset001_ImageTBAD"
DST = nnUNet_raw / "Dataset002_ImageTBAD_HUwin_crop"

(DST / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST / "labelsTr").mkdir(parents=True, exist_ok=True)

def bbox_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mins, maxs

def crop_with_margin(arr: np.ndarray, mins: np.ndarray, maxs: np.ndarray, margin: int):
    mins2 = np.maximum(mins - margin, 0)
    maxs2 = np.minimum(maxs + margin, np.array(arr.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mins2, maxs2))
    return arr[sl], sl

def hu_window_only(img: np.ndarray):
    return np.clip(img.astype(np.float32), HU_LO, HU_HI)

img_files = sorted((SRC / "imagesTr").glob("*.nii.gz"))

written = 0

for img_path in img_files:

    case_id = img_path.name.replace("_0000.nii.gz", "")
    lab_path = SRC / "labelsTr" / f"{case_id}.nii.gz"

    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    img = img_nii.get_fdata(dtype=np.float32)
    lab = lab_nii.get_fdata(dtype=np.float32).astype(np.int16)

    img_win = hu_window_only(img)

    lbl_mask = lab > 0
    bb = bbox_from_mask(lbl_mask)

    if bb is None:
        img_crop = img_win
        lab_crop = lab
    else:
        mins, maxs = bb
        img_crop, sl = crop_with_margin(img_win, mins, maxs, CROP_MARGIN)
        lab_crop = lab[sl]

    img_out = img_crop.astype(np.int16)

    nib.save(
        nib.Nifti1Image(img_out, img_nii.affine),
        str(DST / "imagesTr" / img_path.name),
    )

    nib.save(
        nib.Nifti1Image(lab_crop.astype(np.int16), lab_nii.affine),
        str(DST / "labelsTr" / lab_path.name),
    )

    written += 1
    if written % 10 == 0:
        print("Processed", written)

# ---- dataset.json ----

src_json = json.load(open(SRC / "dataset.json"))
dst_json = dict(src_json)

dst_json["name"] = "ImageTBAD_HUwin_crop"
dst_json["numTraining"] = written
dst_json["numTest"] = 0
dst_json["file_ending"] = ".nii.gz"

with open(DST / "dataset.json", "w") as f:
    json.dump(dst_json, f, indent=2)

print("Finished. Cases written:", written)