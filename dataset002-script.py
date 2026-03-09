from pathlib import Path
import shutil
import json
import numpy as np
import nibabel as nib


# CONFIG

SRC = Path("/content/nnUNet_raw/Dataset001_ImageTBAD")
DST = Path("/content/nnUNet_raw/Dataset002_ImageTBAD_All")

HU_MIN = -1000
HU_MAX = 2000

# foreground estimate for thoracic CT
BODY_THRESHOLD = -500


# PATHS

src_images = SRC / "imagesTr"
src_labels = SRC / "labelsTr"

dst_images = DST / "imagesTr"
dst_labels = DST / "labelsTr"

if DST.exists():
    shutil.rmtree(DST)

dst_images.mkdir(parents=True, exist_ok=True)
dst_labels.mkdir(parents=True, exist_ok=True)


# HELPERS

def bbox_from_mask(mask: np.ndarray):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) + 1 for c in coords]
    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))

def load_nifti(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return img, data


# CASE LOOP

image_files = sorted(src_images.glob("*_0000.nii.gz"))
case_ids = [p.name.replace("_0000.nii.gz", "") for p in image_files]

for case in case_ids:
    img_path = src_images / f"{case}_0000.nii.gz"
    lab_path = src_labels / f"{case}.nii.gz"

    img_nii, img = load_nifti(img_path)
    lab_nii = nib.load(str(lab_path))
    lab = lab_nii.get_fdata()

    # 1) HU clip
    img = np.clip(img, HU_MIN, HU_MAX)

    # 2) estimate body foreground for cropping
    body_mask = img > BODY_THRESHOLD

    #  if threshold fails, fall back to nonzero region
    if not np.any(body_mask):
        body_mask = img != 0

    bbox = bbox_from_mask(body_mask)
    if bbox is None:
        bbox = tuple(slice(0, s) for s in img.shape)

    img = img[bbox]
    lab = lab[bbox]

    # recompute foreground on cropped image
    fg = img > BODY_THRESHOLD
    if not np.any(fg):
        fg = img != 0

    # 3) foreground z-score normalize
    if np.any(fg):
        mu = float(img[fg].mean())
        sigma = float(img[fg].std())
        if sigma > 0:
            img = (img - mu) / sigma
        else:
            img = img - mu

    # set outside foreground to 0
    img[~fg] = 0.0

    # save image
    out_img = nib.Nifti1Image(img.astype(np.float32), img_nii.affine, img_nii.header)
    nib.save(out_img, str(dst_images / f"{case}_0000.nii.gz"))

    # save label unchanged except crop
    out_lab = nib.Nifti1Image(lab.astype(np.uint8), lab_nii.affine, lab_nii.header)
    nib.save(out_lab, str(dst_labels / f"{case}.nii.gz"))


# dataset.json

src_json = SRC / "dataset.json"
with open(src_json, "r") as f:
    dj = json.load(f)

dj["channel_names"] = {"0": "noNorm"}
dj["numTraining"] = len(case_ids)

with open(DST / "dataset.json", "w") as f:
    json.dump(dj, f, indent=2)

# copy case list if present
sel = SRC / "selected_cases.txt"
if sel.exists():
    shutil.copy2(sel, DST / "selected_cases.txt")

print("Created Dataset002_ImageTBAD_All")
print("Images:", len(list(dst_images.glob('*_0000.nii.gz'))))
print("Labels:", len(list(dst_labels.glob('*.nii.gz'))))