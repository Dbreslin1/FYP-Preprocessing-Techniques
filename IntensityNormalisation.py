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
# Safety checks
# -------------------------
if not imagesTr_src.exists() or not labelsTr_src.exists():
    raise FileNotFoundError("Dataset001 missing imagesTr/labelsTr")

img_files = sorted(imagesTr_src.glob("*_0000.nii.gz"))
if len(img_files) == 0:
    raise FileNotFoundError(f"No images found in {imagesTr_src}")

print("SRC:", SRC)
print("DST:", DST)
print("Total images found:", len(img_files))

# -------------------------
# Wipe destination (true overwrite)
# -------------------------
if DST.exists():
    if imagesTr_dst.exists():
        for p in imagesTr_dst.glob("*"):
            p.unlink()
    if labelsTr_dst.exists():
        for p in labelsTr_dst.glob("*"):
            p.unlink()

imagesTr_dst.mkdir(parents=True, exist_ok=True)
labelsTr_dst.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def hu_window_to_int16(img_f32: np.ndarray) -> np.ndarray:
    """
    Clip to HU window and store as SIGNED int16.
    Important: negatives must be preserved (-200..500).
    """
    win = np.clip(img_f32, HU_LO, HU_HI)
    # round before cast so -0.4 doesn't become 0 accidentally
    win = np.rint(win).astype(np.int16)
    return win

def save_nifti_like(ref_nii: nib.Nifti1Image, data: np.ndarray, out_path: Path, dtype):
    """
    Save NIfTI preserving affine + header geometry.
    Avoid qform/sform helper methods that differ across nibabel versions.
    """
    out = nib.Nifti1Image(data.astype(dtype), ref_nii.affine)

    # copy header safely
    hdr = ref_nii.header.copy()
    hdr.set_data_dtype(dtype)

    # Ensure scaling is neutral (avoid slope/inter weirdness)
    hdr["scl_slope"] = np.nan
    hdr["scl_inter"] = np.nan

    out.header = hdr
    nib.save(out, str(out_path))

# -------------------------
# Main loop
# -------------------------
written = 0

for idx, img_path in enumerate(img_files):
    case_id = img_path.name.replace("_0000.nii.gz", "")
    lab_path = labelsTr_src / f"{case_id}.nii.gz"
    if not lab_path.exists():
        raise FileNotFoundError(f"Missing label for {case_id}: {lab_path}")

    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    # Use get_fdata float32 to correctly read HU values
    img = img_nii.get_fdata(dtype=np.float32)
    lab = lab_nii.get_fdata(dtype=np.float32)

    # --- Source sanity (first case only) ---
    if idx == 0:
        print("\n[SOURCE sanity]")
        print("file:", img_path.name)
        print("source min/max:", float(img.min()), float(img.max()))
        print("source frac<0:", float((img < 0).mean()))
        print("source percentiles [0,1,50,99,100]:",
              np.percentile(img, [0,1,50,99,100]).tolist())

        if float((img < 0).mean()) < 0.01:
            raise RuntimeError("Dataset001 doesn't look like HU (almost no negatives). Stop.")

    # 1) HU window to signed int16
    img_win = hu_window_to_int16(img)

    # --- Post-window sanity (first case only) ---
    if idx == 0:
        print("\n[WINDOW sanity]")
        print("window dtype:", img_win.dtype, "shape:", img_win.shape)
        print("window min/max:", int(img_win.min()), int(img_win.max()))
        print("window frac<0:", float((img_win < 0).mean()))
        print("window percentiles [0,1,50,99,100]:",
              np.percentile(img_win.astype(np.float32), [0,1,50,99,100]).tolist())

        if float((img_win < 0).mean()) == 0.0:
            raise RuntimeError("After windowing, still no negatives. Something is wrong. Stop.")

    # 2) Labels: store as uint8/int16 (keep as int16)
    lab_int = np.rint(lab).astype(np.int16)

    # 3) Save
    save_nifti_like(img_nii, img_win, imagesTr_dst / img_path.name, np.int16)
    save_nifti_like(lab_nii, lab_int, labelsTr_dst / lab_path.name, np.int16)

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

print("\nDONE.")
print("Wrote images:", len(list(imagesTr_dst.glob('*_0000.nii.gz'))))
print("Wrote labels:", len(list(labelsTr_dst.glob('*.nii.gz'))))
print("dataset.json numTraining:", dst_json["numTraining"])
print("Dataset002 path:", DST)