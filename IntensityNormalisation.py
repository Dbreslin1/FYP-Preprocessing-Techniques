import json
from pathlib import Path
import numpy as np
import nibabel as nib

print("=== REBUILDING Dataset002: HU window [-200, 500], NO label crop, store int16 ===")

# -------------------------
# Configuration
# -------------------------
HU_LO = -200.0
HU_HI = 500.0

nnUNet_raw = Path("/content/drive/MyDrive/FYP_nnUNet/nnUNet_raw")

SRC = nnUNet_raw / "Dataset001_ImageTBAD"
DST = nnUNet_raw / "Dataset002_ImageTBAD_HUwin"

imagesTr_src = SRC / "imagesTr"
labelsTr_src = SRC / "labelsTr"
imagesTr_dst = DST / "imagesTr"
labelsTr_dst = DST / "labelsTr"

# -------------------------
# Helpers
# -------------------------
def hu_window_int16(img_f32: np.ndarray) -> np.ndarray:
    """Clip to HU window and store as SIGNED int16 (keeps negatives)."""
    return np.clip(img_f32, HU_LO, HU_HI).astype(np.int16)

def save_nifti_like(ref_nii: nib.Nifti1Image, data: np.ndarray, out_path: Path, dtype):
    """
    Save NIfTI preserving affine + copying header fields safely
    without assigning out.header (some nibabel builds make it read-only).
    """
    out = nib.Nifti1Image(data.astype(dtype), ref_nii.affine)

    # set dtype safely
    out.set_data_dtype(dtype)

    ref_hdr = ref_nii.header

    # Copy qform/sform if available
    try:
        out.set_qform(ref_nii.get_qform(), code=int(ref_hdr["qform_code"]))
    except Exception:
        pass

    try:
        out.set_sform(ref_nii.get_sform(), code=int(ref_hdr["sform_code"]))
    except Exception:
        pass

    # Copy pixdim if present
    try:
        out.header["pixdim"] = ref_hdr["pixdim"]
    except Exception:
        pass

    # Avoid scaling
    out.header["scl_slope"] = np.nan
    out.header["scl_inter"] = np.nan

    nib.save(out, str(out_path))

# -------------------------
# Prepare destination folders
# -------------------------
imagesTr_dst.mkdir(parents=True, exist_ok=True)
labelsTr_dst.mkdir(parents=True, exist_ok=True)

# wipe dst contents so you truly overwrite Dataset002
for p in imagesTr_dst.glob("*"):
    p.unlink()
for p in labelsTr_dst.glob("*"):
    p.unlink()

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

# sanity prints on first case
first_img = img_files[0]
first_case_id = first_img.name.replace("_0000.nii.gz", "")
first_lab = labelsTr_src / f"{first_case_id}.nii.gz"
if not first_lab.exists():
    raise FileNotFoundError(f"Missing label for first case: {first_lab}")

img0 = nib.load(str(first_img)).get_fdata(dtype=np.float32)
print("\n[SOURCE sanity]")
print("file:", first_img.name)
print("source min/max:", float(img0.min()), float(img0.max()))
print("source frac<0:", float((img0 < 0).mean()))
print("source percentiles [0,1,50,99,100]:", np.percentile(img0, [0,1,50,99,100]).tolist())

img0_win = hu_window_int16(img0)
print("\n[WINDOW sanity]")
print("window dtype:", img0_win.dtype, "shape:", img0_win.shape)
print("window min/max:", int(img0_win.min()), int(img0_win.max()))
print("window frac<0:", float((img0_win < 0).mean()))
print("window percentiles [0,1,50,99,100]:", np.percentile(img0_win.astype(np.float32), [0,1,50,99,100]).tolist())

# now process all
for img_path in img_files:
    case_id = img_path.name.replace("_0000.nii.gz", "")
    lab_path = labelsTr_src / f"{case_id}.nii.gz"
    if not lab_path.exists():
        raise FileNotFoundError(f"Missing label for {case_id}: {lab_path}")

    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    img = img_nii.get_fdata(dtype=np.float32)
    lab = lab_nii.get_fdata(dtype=np.float32).astype(np.int16)

    img_win = hu_window_int16(img)

    save_nifti_like(img_nii, img_win, imagesTr_dst / img_path.name, np.int16)
    save_nifti_like(lab_nii, lab, labelsTr_dst / lab_path.name, np.int16)

    written += 1
    if written % 10 == 0:
        print(f"Processed {written}/{len(img_files)}")

# -------------------------
# dataset.json
# -------------------------
src_json_path = SRC / "dataset.json"
dst_json_path = DST / "dataset.json"

if src_json_path.exists():
    src_json = json.load(open(src_json_path, "r"))
    dst_json = dict(src_json)
else:
    dst_json = {
        "name": "ImageTBAD_HUwin",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {"background": 0, "class1": 1, "class2": 2, "class3": 3},
        "file_ending": ".nii.gz",
    }

dst_json["name"] = "ImageTBAD_HUwin"
dst_json["file_ending"] = ".nii.gz"
dst_json["numTraining"] = len(list(imagesTr_dst.glob("*_0000.nii.gz")))
dst_json["numTest"] = 0

with open(dst_json_path, "w") as f:
    json.dump(dst_json, f, indent=2)

print("\n dataset.json updated. numTraining =", dst_json["numTraining"])
print(f"DONE. Wrote {written} cases to {DST.name}")