from __future__ import annotations
import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


@dataclass
class CropConfig:
    threshold_hu: float = -650.0
    closing_iters: int = 1
    opening_iters: int = 0
    margin_xy: int = 20
    margin_z: int = 8
    crop_mode: str = "xy_only"
    min_component_voxels: int = 10000


@dataclass
class OutputConfig:
    overwrite: bool = True
    write_case_report: bool = True


@dataclass
class PipelineConfig:
    src_dataset: str
    dst_dataset: str
    crop: CropConfig
    output: OutputConfig


def load_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    nii = nib.load(str(path))
    data = np.asanyarray(nii.dataobj).astype(np.float32, copy=False)
    return nii, data


def save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path) -> None:
    out = nib.Nifti1Image(data, affine, header=header)
    nib.save(out, str(path))


def ensure_3d(arr: np.ndarray, path: Path) -> None:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {arr.shape}")


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    coords = np.where(mask)
    if coords[0].size == 0:
        return None
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) + 1 for c in coords]
    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))  # type: ignore[return-value]


def grow_bbox(
    bbox: Tuple[slice, slice, slice],
    shape: Sequence[int],
    margin_xyz: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    out: List[slice] = []
    for sl, dim, margin in zip(bbox, shape, margin_xyz):
        start = max(0, sl.start - margin)
        stop = min(dim, sl.stop + margin)
        out.append(slice(start, stop))
    return tuple(out)  # type: ignore[return-value]


def affine_for_crop(affine: np.ndarray, crop_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    starts = np.array([crop_slices[0].start, crop_slices[1].start, crop_slices[2].start, 1.0], dtype=np.float64)
    new_affine = affine.copy()
    new_affine[:3, 3] = (affine @ starts)[:3]
    return new_affine


def largest_component(mask: np.ndarray, min_size: int) -> np.ndarray:
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    out = labeled == largest
    if counts[largest] < min_size:
        return mask
    return out


def body_mask_from_ct(image_hu: np.ndarray, cfg: CropConfig) -> np.ndarray:
    mask = image_hu > cfg.threshold_hu
    if cfg.closing_iters > 0:
        mask = ndi.binary_closing(mask, iterations=cfg.closing_iters)
    if cfg.opening_iters > 0:
        mask = ndi.binary_opening(mask, iterations=cfg.opening_iters)
    mask = ndi.binary_fill_holes(mask)
    mask = largest_component(mask, min_size=cfg.min_component_voxels)
    return mask.astype(bool)


def choose_crop_bbox(image_hu: np.ndarray, cfg: CropConfig) -> Tuple[slice, slice, slice]:
    body = body_mask_from_ct(image_hu, cfg)
    bbox = bbox_from_mask(body)
    if bbox is None:
        return (slice(0, image_hu.shape[0]), slice(0, image_hu.shape[1]), slice(0, image_hu.shape[2]))

    if cfg.crop_mode == "xy_only":
        bbox = (bbox[0], bbox[1], slice(0, image_hu.shape[2]))
    elif cfg.crop_mode != "xyz":
        raise ValueError(f"Unsupported crop_mode={cfg.crop_mode}. Use 'xy_only' or 'xyz'.")

    margin_z = 0 if cfg.crop_mode == "xy_only" else cfg.margin_z
    return grow_bbox(bbox, image_hu.shape, (cfg.margin_xy, cfg.margin_xy, margin_z))


def case_id_from_image(path: Path) -> str:
    return path.name.replace("_0000.nii.gz", "")


def maybe_copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def collect_training_cases(src_images: Path, src_labels: Path) -> List[Tuple[str, Path, Path]]:
    image_files = sorted(src_images.glob("*_0000.nii.gz"))
    cases: List[Tuple[str, Path, Path]] = []
    for img_path in image_files:
        case = case_id_from_image(img_path)
        lab_path = src_labels / f"{case}.nii.gz"
        if not lab_path.exists():
            raise FileNotFoundError(f"Missing label for case {case}: {lab_path}")
        cases.append((case, img_path, lab_path))
    if not cases:
        raise RuntimeError(f"No training images found in {src_images}")
    return cases


def preprocess_one_case(
    case: str,
    img_path: Path,
    lab_path: Path,
    dst_images: Path,
    dst_labels: Path,
    crop_cfg: CropConfig,
) -> Dict[str, object]:
    img_nii, img_hu = load_nifti(img_path)
    lab_nii, lab = load_nifti(lab_path)

    ensure_3d(img_hu, img_path)
    ensure_3d(lab, lab_path)

    original_shape = tuple(int(x) for x in img_hu.shape)
    bbox = choose_crop_bbox(img_hu, crop_cfg)

    cropped_img = img_hu[bbox].astype(np.float32)
    cropped_lab = np.rint(lab[bbox]).astype(np.uint8)

    new_affine_img = affine_for_crop(img_nii.affine, bbox)
    new_affine_lab = affine_for_crop(lab_nii.affine, bbox)

    out_img_path = dst_images / f"{case}_0000.nii.gz"
    out_lab_path = dst_labels / f"{case}.nii.gz"

    img_header = img_nii.header.copy()
    lab_header = lab_nii.header.copy()
    img_header.set_data_dtype(np.float32)
    lab_header.set_data_dtype(np.uint8)

    save_nifti(cropped_img, new_affine_img, img_header, out_img_path)
    save_nifti(cropped_lab, new_affine_lab, lab_header, out_lab_path)

    report = {
        "case": case,
        "original_shape": list(original_shape),
        "cropped_shape": list(map(int, cropped_img.shape)),
        "crop_bbox": {
            "x": [int(bbox[0].start), int(bbox[0].stop)],
            "y": [int(bbox[1].start), int(bbox[1].stop)],
            "z": [int(bbox[2].start), int(bbox[2].stop)],
        },
        "label_voxels_after_crop": int(np.sum(cropped_lab > 0)),
        "image_stats_after_crop": {
            "min": float(cropped_img.min()),
            "max": float(cropped_img.max()),
            "mean": float(cropped_img.mean()),
            "std": float(cropped_img.std()),
        },
    }
    return report


def build_dataset_json(src_dataset_json: Path, dst_dataset_json: Path, num_training: int) -> None:
    with open(src_dataset_json, "r") as f:
        dj = json.load(f)

    dj["channel_names"] = {"0": "CT"}
    dj["numTraining"] = int(num_training)

    with open(dst_dataset_json, "w") as f:
        json.dump(dj, f, indent=2)


def run_pipeline(cfg: PipelineConfig) -> None:
    src = Path(cfg.src_dataset)
    dst = Path(cfg.dst_dataset)

    src_images = src / "imagesTr"
    src_labels = src / "labelsTr"
    src_json = src / "dataset.json"

    dst_images = dst / "imagesTr"
    dst_labels = dst / "labelsTr"

    if cfg.output.overwrite and dst.exists():
        shutil.rmtree(dst)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    cases = collect_training_cases(src_images, src_labels)
    print(f"[INFO] Found {len(cases)} training cases")
    print("[INFO] Preprocessing cases...")

    reports: List[Dict[str, object]] = []
    total = len(cases)
    for i, (case, img_path, lab_path) in enumerate(cases, 1):
        rep = preprocess_one_case(
            case=case,
            img_path=img_path,
            lab_path=lab_path,
            dst_images=dst_images,
            dst_labels=dst_labels,
            crop_cfg=cfg.crop,
        )
        reports.append(rep)
        print(f"[INFO] [{i}/{total}] {case}: {rep['cropped_shape']}")

    build_dataset_json(src_json, dst / "dataset.json", len(cases))
    maybe_copy_file(src / "selected_cases.txt", dst / "selected_cases.txt")

    metadata = {
        "pipeline_name": "ImageTBADBodyCropCT",
        "config": asdict(cfg),
        "n_cases": len(cases),
        "reports": reports if cfg.output.write_case_report else None,
    }

    with open(dst / "preprocessing_report.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Created dataset at: {dst}")
    print(f"[INFO] Images: {len(list(dst_images.glob('*_0000.nii.gz')))}")
    print(f"[INFO] Labels: {len(list(dst_labels.glob('*.nii.gz')))}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImageTBAD body-crop preprocessing for nnU-Net")
    p.add_argument("--src", required=True, help="Source nnU-Net raw dataset folder")
    p.add_argument("--dst", required=True, help="Destination nnU-Net raw dataset folder")
    p.add_argument("--body-threshold", type=float, default=-650.0)
    p.add_argument("--crop-mode", choices=["xy_only", "xyz"], default="xy_only")
    p.add_argument("--margin-xy", type=int, default=20)
    p.add_argument("--margin-z", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        src_dataset=args.src,
        dst_dataset=args.dst,
        crop=CropConfig(
            threshold_hu=args.body_threshold,
            margin_xy=args.margin_xy,
            margin_z=args.margin_z,
            crop_mode=args.crop_mode,
        ),
        output=OutputConfig(),
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()