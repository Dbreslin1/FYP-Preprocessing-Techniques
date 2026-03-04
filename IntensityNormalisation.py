import json
from pathlib import Path
import numpy as np
import nibabel as nib

print("=== REBUILDING Dataset002: HU window [-200, 500] + label ROI crop (NO z-score) ===")

# -------------------------
# Configuration
# -------------------------
HU_LO = -200.0
HU_HI = 500.0
CROP_MARGIN = 16

nnUNet_raw = Path("/content/drive/MyDrive/FYP_nnUNet/nnUNet_raw")

SRC = nnUNet_raw / "Dataset001_ImageTBAD"
DST = nnUNet_raw / "Dataset002_ImageTBAD_HUwin_crop"

imagesTr_src = SRC / "imagesTr"
labelsTr_src = SRC / "labelsTr"
imagesTr_dst = DST / "imagesTr"
labelsTr_dst = DST / "labelsTr"

# IMPORTANT: wipe dst folders so you truly overwrite Dataset002
imagesTr_dst.mkdir(parents=True, exist_ok=True)
labelsTr_dst.mkdir(parents=True, exist_ok=True)
for p in imagesTr_dst.glob("*"):
    p.unlink()
for p in labelsTr_dst.glob("*"):
    p.unlink()

# -------------------------
# Helpers
# -------------------------
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

def hu_window_int16(img_f32: np.ndarray) -> np.ndarray:
    """Clip to HU window and store as SIGNED int16 (keeps negatives)."""
    clipped = np.clip(img_f32, HU_LO, HU_HI)
    return clipped.astype(np.int16)

def make_img_header_like(src_nii: nib.Nifti1Image):
    """Create a safe header: int16 + slope/inter set (prevents nan scaling)."""
    hdr = src_nii.header.copy()
    hdr.set_data_dtype(np.int16)
    hdr["scl_slope"] = 1
    hdr["scl_inter"] = 0
    hdr["cal_min"] = HU_LO
    hdr["cal_max"] = HU_HI
    return hdr

def make_lab_header_like(src_nii: nib.Nifti1Image):
    """Labels should be uint8 (small integers)."""
    hdr = src_nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"] = 1
    hdr["scl_inter"] = 0
    hdr["cal_min"] = 0
    hdr["cal_max"] = 255
    return hdr

# -------------------------
# Main loop
# -------------------------
img_files = sorted(imagesTr_src.glob("*_0000.nii.gz"))
if len(img_files) == 0:
    raise FileNotFoundError(f"No images found in {imagesTr_src}")

print("SRC:", SRC)
print("DST:", DST)
print("Total images found:", len(img_files))

written = 0

try:
    for idx, img_path in enumerate(img_files):
        case_id = img_path.name.replace("_0000.nii.gz", "")
        lab_path = labelsTr_src / f"{case_id}.nii.gz"
        if not lab_path.exists():
            raise FileNotFoundError(f"Missing label for {case_id}: {lab_path}")

        # Load (float32 for math)
        img_nii = nib.load(str(img_path))
        lab_nii = nib.load(str(lab_path))

        img = img_nii.get_fdata(dtype=np.float32)
        lab = lab_nii.get_fdata(dtype=np.float32).astype(np.int16)

        # sanity check source HU
        if idx == 0:
            print("SOURCE example", img_path.name, "min/max:", float(img.min()), float(img.max()))
            print("SOURCE frac<0:", float((img < 0).mean()))

        # 1) HU window -> int16 (keeps negatives)
        img_win = hu_window_int16(img)

        if idx == 0:
            print("WIN example stored dtype:", img_win.dtype)
            print("WIN min/max:", int(img_win.min()), int(img_win.max()))
            print("WIN frac<0:", float((img_win < 0).mean()))
            print("WIN percentiles [0,1,50,99,100]:",
                  np.percentile(img_win.astype(np.float32), [0,1,50,99,100]).tolist())
            if float((img_win < 0).mean()) == 0.0:
                raise RuntimeError("Windowed image has no negatives. That means your SOURCE is not HU.")

        # 2) ROI crop from LABEL bbox
        bb = bbox_from_mask(lab > 0)
        if bb is not None:
            mins, maxs = bb
            img_win, sl = crop_with_margin(img_win, mins, maxs, CROP_MARGIN)
            lab = lab[sl]

        # 3) Save outputs (SAFE headers)
        img_hdr = make_img_header_like(img_nii)
        out_img = nib.Nifti1Image(img_win.astype(np.int16), img_nii.affine, header=img_hdr)
        out_img.set_qform(img_nii.get_qform(), code=img_nii.header.get_qform_code())
        out_img.set_sform(img_nii.get_sform(), code=img_nii.header.get_sform_code())

        lab_hdr = make_lab_header_like(lab_nii)
        out_lab = nib.Nifti1Image(lab.astype(np.uint8), lab_nii.affine, header=lab_hdr)
        out_lab.set_qform(lab_nii.get_qform(), code=lab_nii.header.get_qform_code())
        out_lab.set_sform(lab_nii.get_sform(), code=lab_nii.header.get_sform_code())

        nib.save(out_img, str(imagesTr_dst / img_path.name))
        nib.save(out_lab, str(labelsTr_dst / lab_path.name))

        written += 1
        if written % 10 == 0:
            print(f"Processed {written}/{len(img_files)}")

finally:
    # Always write dataset.json
    src_json_path = SRC / "dataset.json"
    dst_json_path = DST / "dataset.json"

    if src_json_path.exists():
        src_json = json.load(open(src_json_path, "r"))
        dst_json = dict(src_json)
    else:
        dst_json = {
            "name": "ImageTBAD_HUwin_crop",
            "tensorImageSize": "3D",
            "modality": {"0": "CT"},
            "labels": {"background": 0, "class1": 1, "class2": 2, "class3": 3},
            "file_ending": ".nii.gz",
        }

    dst_json["name"] = "ImageTBAD_HUwin_crop"
    dst_json["file_ending"] = ".nii.gz"
    dst_json["numTraining"] = len(list(imagesTr_dst.glob("*_0000.nii.gz")))
    dst_json["numTest"] = 0

    with open(dst_json_path, "w") as f:
        json.dump(dst_json, f, indent=2)

    print("dataset.json updated. numTraining =", dst_json["numTraining"])

print(f"DONE. Wrote {written} cases to {DST.name}")