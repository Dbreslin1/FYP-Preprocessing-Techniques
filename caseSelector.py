
# CLEAN nnU-Net folders + CREATE 50-CASE BASELINE DATASET


from pathlib import Path
import shutil
import random
import json


# CONFIG
#Google Drive nnU-Net root
DRIVE_ROOT = Path("/content/drive/MyDrive")
NNUNET_ROOT = DRIVE_ROOT / "FYP_nnUNet"

# Standard nnU-Net folders
NNUNET_RAW = NNUNET_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_ROOT / "nnUNet_results"


SRC_DATASET_NAME = "Dataset001_ImageTBAD"   

# The new 50-case baseline dataset
DST_DATASET_ID = 1
DST_DATASET_NAME = "Dataset001_ImageTBAD"

# Number of cases to keep
N_CASES = 50

# Reproducibility
SEED = 42

# If True, deletes old Dataset001 from raw/preprocessed/results
CLEAN_OLD_EXPERIMENTS = True

# Which dataset IDs to clean
DATASET_IDS_TO_CLEAN = [1, 2, 3, 4, 5]

# If destination already exists, delete it and recreate it
OVERWRITE_DESTINATION = True


# PATHS


src_ds = NNUNET_RAW / SRC_DATASET_NAME
dst_ds = NNUNET_RAW / DST_DATASET_NAME

src_images = src_ds / "imagesTr"
src_labels = src_ds / "labelsTr"

dst_images = dst_ds / "imagesTr"
dst_labels = dst_ds / "labelsTr"

selected_cases_file = dst_ds / "selected_cases.txt"
copied_dataset_json = dst_ds / "dataset.json"


# HELPER FUNCTIONS


def delete_folder_if_exists(path: Path):
    if path.exists():
        print(f"Deleting: {path}")
        shutil.rmtree(path)

def find_dataset_folder_by_id(base: Path, dataset_id: int):
    prefix = f"Dataset{dataset_id:03d}_"
    matches = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return matches


# SAFETY CHECKS

print("Checking source dataset...")
print("Source dataset path:", src_ds)

if not src_ds.exists():
    raise FileNotFoundError(f"Source dataset folder not found: {src_ds}")

if not src_images.exists():
    raise FileNotFoundError(f"Missing imagesTr: {src_images}")

if not src_labels.exists():
    raise FileNotFoundError(f"Missing labelsTr: {src_labels}")

dataset_json_src = src_ds / "dataset.json"
if not dataset_json_src.exists():
    raise FileNotFoundError(f"Missing dataset.json: {dataset_json_src}")


# OPTIONAL CLEANUP


if CLEAN_OLD_EXPERIMENTS:
    print("\n=== CLEANING OLD EXPERIMENT FOLDERS ===")
    
    # raw
    for ds_id in DATASET_IDS_TO_CLEAN:
        matches = find_dataset_folder_by_id(NNUNET_RAW, ds_id)
        for m in matches:
            # do not delete the source folder if it happens to match
            if m.resolve() == src_ds.resolve():
                print(f"Skipping source dataset: {m}")
                continue
            delete_folder_if_exists(m)

    # preprocessed
    if NNUNET_PREPROCESSED.exists():
        for ds_id in DATASET_IDS_TO_CLEAN:
            matches = find_dataset_folder_by_id(NNUNET_PREPROCESSED, ds_id)
            for m in matches:
                delete_folder_if_exists(m)

    # results
    if NNUNET_RESULTS.exists():
        for ds_id in DATASET_IDS_TO_CLEAN:
            matches = find_dataset_folder_by_id(NNUNET_RESULTS, ds_id)
            for m in matches:
                delete_folder_if_exists(m)


# CREATE DESTINATION


print("\n=== CREATING NEW 50-CASE BASELINE ===")

if dst_ds.exists() and OVERWRITE_DESTINATION:
    print(f"Destination exists, deleting: {dst_ds}")
    shutil.rmtree(dst_ds)

dst_images.mkdir(parents=True, exist_ok=True)
dst_labels.mkdir(parents=True, exist_ok=True)


# GET CASE LIST


image_files = sorted(src_images.glob("*_0000.nii.gz"))
cases = [p.name.replace("_0000.nii.gz", "") for p in image_files]

print(f"Found {len(cases)} image cases in source dataset.")

if len(cases) < N_CASES:
    raise ValueError(f"Source only has {len(cases)} cases, cannot sample {N_CASES}.")

# Make sure every selected image has a matching label
valid_cases = []
missing_labels = []

for case in cases:
    expected_label = src_labels / f"{case}.nii.gz"
    if expected_label.exists():
        valid_cases.append(case)
    else:
        missing_labels.append(case)

if missing_labels:
    print("\nWARNING: These cases are missing labels and will be ignored:")
    for c in missing_labels[:10]:
        print(" -", c)
    if len(missing_labels) > 10:
        print(f" ... and {len(missing_labels)-10} more")

print(f"Valid paired image+label cases: {len(valid_cases)}")

if len(valid_cases) < N_CASES:
    raise ValueError(f"Only {len(valid_cases)} valid paired cases found, cannot sample {N_CASES}.")


# SELECT 50 CASES


random.seed(SEED)
selected_cases = sorted(random.sample(valid_cases, N_CASES))

print(f"\nSelected {len(selected_cases)} cases with seed={SEED}")
print("First 10 selected cases:")
for c in selected_cases[:10]:
    print(" -", c)

# COPY FILES


print("\n=== COPYING FILES ===")

for i, case in enumerate(selected_cases, start=1):
    src_img = src_images / f"{case}_0000.nii.gz"
    src_lab = src_labels / f"{case}.nii.gz"

    dst_img = dst_images / f"{case}_0000.nii.gz"
    dst_lab = dst_labels / f"{case}.nii.gz"

    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lab, dst_lab)

    if i % 10 == 0 or i == len(selected_cases):
        print(f"Copied {i}/{len(selected_cases)}")

# COPY + UPDATE dataset.json


print("\n=== UPDATING dataset.json ===")

with open(dataset_json_src, "r") as f:
    ds_json = json.load(f)

# Update numTraining if present
ds_json["numTraining"] = N_CASES

# Keep Dataset001 as baseline CT
# If channel_names already exists, force it to CT baseline
ds_json["channel_names"] = {"0": "CT"}

with open(copied_dataset_json, "w") as f:
    json.dump(ds_json, f, indent=2)


# SAVE CASE LIST


with open(selected_cases_file, "w") as f:
    for case in selected_cases:
        f.write(case + "\n")

print("\n=== DONE ===")
print("Created:", dst_ds)
print("Images copied:", len(list(dst_images.glob("*_0000.nii.gz"))))
print("Labels copied:", len(list(dst_labels.glob("*.nii.gz"))))
print("dataset.json exists:", copied_dataset_json.exists())
print("selected_cases.txt exists:", selected_cases_file.exists())
print("\nSelected cases file:", selected_cases_file)