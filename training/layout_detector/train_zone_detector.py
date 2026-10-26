#!/usr/bin/env python3
"""
YOLOv8-nano Zone Detector Training — Multi-dataset merge for cheque field detection.

Datasets (merged, ~560 images covering all 9 target classes):
  1. Roboflow "Cheques-Entities-Detection": 233 images, 8 classes (BEST class coverage)
  2. Roboflow "Cheques" (IDPS2): 215 images, 6 classes (MIT licensed)
  3. Roboflow "Cheque Segmentation": 112 images, 8 classes (has Signature)

Class remapping (each dataset uses different names for the same zones):
  Dataset 1: Amount_In_Numbers -> amount_figures, Payee_Name -> payee_name, etc.
  Dataset 2: Amount_In_Numbers -> amount_figures, Sign -> signature, etc.
  Dataset 3: Amt -> amount_figures, ReceiverName -> payee_name, Sign -> signature, etc.

Training produces a YOLOv8-nano model (~3.2M params) that runs in 20-30ms on CPU.

Usage:
    python training/train_zone_detector.py \
        --roboflow_api_key YOUR_KEY \
        --epochs 100 \
        --output models/cheque_zones

    python training/train_zone_detector.py \
        --dataset_path /path/to/merged/dataset \
        --epochs 100
"""

import argparse
import os
import sys
import shutil
import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Unified class mapping ───────────────────────────────────────────────
# Maps source dataset class names to our unified 9-class schema
UNIFIED_CLASSES = {
    0: "amount_figures",
    1: "amount_words",
    2: "payee_name",
    3: "payer_name",
    4: "date",
    5: "micr_line",
    6: "bank_name",
    7: "signature",
    8: "account_number",
}

# Class name remapping per dataset
# Maps dataset-specific class names -> unified class name
DATASET_CLASS_MAPS = {
    "cheques-entities-detection": {
        "Amount_In_Numbers": "amount_figures",
        "Amount_In_Words": "amount_words",
        "Payee_Name": "payee_name",
        "Date": "date",
        "MICR": "micr_line",
        "Bank_Detail": "bank_name",
        "Account_Number": "account_number",
        "IBAN": "account_number",  # merge IBAN into account_number
    },
    "cheques-idps2": {
        "Amount_In_Numbers": "amount_figures",
        "Amount_In_Words": "amount_words",
        "Payee_Name": "payee_name",
        "Date": "date",
        "MICR": "micr_line",
        "Sign": "signature",
    },
    "cheque-segmentation": {
        "Amt": "amount_figures",
        "AmtWords": "amount_words",
        "ReceiverName": "payee_name",
        "DateIss": "date",
        "IssueBank": "bank_name",
        "Sign": "signature",
        "AcNo": "account_number",
        "ChqNo": "micr_line",  # cheque number is part of MICR
    },
}

# Reverse lookup: unified class name -> class ID
UNIFIED_NAME_TO_ID = {v: k for k, v in UNIFIED_CLASSES.items()}


def download_all_datasets(api_key: str, output_dir: str) -> list:
    """
    Download all 3 Roboflow cheque zone datasets.

    Returns list of (dataset_dir, dataset_key) tuples.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow not installed. Install with: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    datasets = []

    # Dataset 1: Cheques-Entities-Detection (233 images, 8 classes)
    print("\n[1/3] Downloading Cheques-Entities-Detection (233 imgs)...")
    try:
        project1 = rf.workspace("chequedetection-xn2ih").project("cheques-entities-detection")
        v1 = project1.version(1)
        d1 = v1.download("yolov8", location=os.path.join(output_dir, "dataset1"))
        datasets.append((d1.location, "cheques-entities-detection"))
        print(f"  Downloaded to: {d1.location}")
    except Exception as e:
        print(f"  WARNING: Failed to download dataset 1: {e}")

    # Dataset 2: Cheques IDPS2 (215 images, 6 classes, MIT)
    print("\n[2/3] Downloading Cheques IDPS2 (215 imgs)...")
    try:
        project2 = rf.workspace("idps2-alcit").project("cheques-c6o30")
        v2 = project2.version(1)
        d2 = v2.download("yolov8", location=os.path.join(output_dir, "dataset2"))
        datasets.append((d2.location, "cheques-idps2"))
        print(f"  Downloaded to: {d2.location}")
    except Exception as e:
        print(f"  WARNING: Failed to download dataset 2: {e}")

    # Dataset 3: Cheque Segmentation (112 images, 8 classes)
    print("\n[3/3] Downloading Cheque Segmentation (112 imgs)...")
    try:
        project3 = rf.workspace("data-annotation-nwd6d").project("cheque-segmentation")
        v3 = project3.version(1)
        d3 = v3.download("yolov8", location=os.path.join(output_dir, "dataset3"))
        datasets.append((d3.location, "cheque-segmentation"))
        print(f"  Downloaded to: {d3.location}")
    except Exception as e:
        print(f"  WARNING: Failed to download dataset 3: {e}")

    return datasets


def read_dataset_classes(dataset_dir: str) -> dict:
    """Read class mapping from data.yaml."""
    import yaml

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        return {}

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    # Convert to {id: name} if it's a list
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


def remap_label_file(
    label_path: str,
    source_classes: dict,
    class_map: dict,
    output_path: str,
) -> int:
    """
    Remap a YOLO label file from source classes to unified classes.

    Args:
        label_path: Path to source .txt label file
        source_classes: {id: name} mapping from source dataset
        class_map: {source_name: unified_name} mapping
        output_path: Path to write remapped label file

    Returns:
        Number of valid annotations written
    """
    valid_lines = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            src_cls_id = int(parts[0])
            src_cls_name = source_classes.get(src_cls_id, "")

            # Look up unified class
            unified_name = class_map.get(src_cls_name)
            if unified_name is None:
                continue  # Skip unmapped classes

            unified_id = UNIFIED_NAME_TO_ID.get(unified_name)
            if unified_id is None:
                continue

            # Replace class ID, keep bbox coordinates
            parts[0] = str(unified_id)
            valid_lines.append(" ".join(parts))

    with open(output_path, "w") as f:
        f.write("\n".join(valid_lines))

    return len(valid_lines)


def merge_datasets(
    datasets: list,
    merged_dir: str,
) -> str:
    """
    Merge multiple YOLO-format datasets with class remapping.

    Args:
        datasets: List of (dataset_dir, dataset_key) tuples
        merged_dir: Output directory for merged dataset

    Returns:
        Path to merged data.yaml
    """
    import yaml

    print(f"\nMerging {len(datasets)} datasets into unified format...")

    # Create merged directory structure
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(merged_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(merged_dir, split, "labels"), exist_ok=True)

    total_images = 0
    total_annotations = 0

    for dataset_dir, dataset_key in datasets:
        source_classes = read_dataset_classes(dataset_dir)
        class_map = DATASET_CLASS_MAPS.get(dataset_key, {})

        if not source_classes:
            print(f"  WARNING: No class mapping found for {dataset_key}")
            continue

        print(f"\n  Processing {dataset_key}:")
        print(f"    Source classes: {source_classes}")
        print(f"    Class map: {class_map}")

        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(dataset_dir, split, "images")
            lbl_dir = os.path.join(dataset_dir, split, "labels")

            if not os.path.isdir(img_dir):
                continue

            for img_file in os.listdir(img_dir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # Source paths
                src_img = os.path.join(img_dir, img_file)
                lbl_name = os.path.splitext(img_file)[0] + ".txt"
                src_lbl = os.path.join(lbl_dir, lbl_name)

                # Unique destination names (prefix with dataset key)
                prefix = dataset_key.replace("-", "_")
                dst_img_name = f"{prefix}_{img_file}"
                dst_lbl_name = f"{prefix}_{lbl_name}"

                dst_img = os.path.join(merged_dir, split, "images", dst_img_name)
                dst_lbl = os.path.join(merged_dir, split, "labels", dst_lbl_name)

                # Copy image
                shutil.copy2(src_img, dst_img)

                # Remap and copy labels
                if os.path.exists(src_lbl):
                    n_annots = remap_label_file(
                        src_lbl, source_classes, class_map, dst_lbl
                    )
                    total_annotations += n_annots

                total_images += 1

        print(f"    Images processed: {total_images}")

    # Write unified data.yaml
    yaml_content = {
        "path": os.path.abspath(merged_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }

    yaml_path = os.path.join(merged_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n  Merge complete:")
    print(f"    Total images: {total_images}")
    print(f"    Total annotations: {total_annotations}")
    print(f"    Classes: {len(UNIFIED_CLASSES)}")
    print(f"    data.yaml: {yaml_path}")

    return yaml_path


def train_yolov8(
    dataset_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    model_size: str = "n",
):
    """Train YOLOv8-nano on the merged cheque zone dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print(f"Training YOLOv8-{model_size} Zone Detector")
    print("=" * 60)

    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print(f"  Dataset: {dataset_path}")
    print(f"  Config: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}")

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        project=output_dir,
        name="cheque_zones",
        patience=20,
        save=True,
        plots=True,
        device="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        workers=4,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # Export to ONNX
    print("\nExporting to ONNX...")
    best_model = os.path.join(output_dir, "cheque_zones", "weights", "best.pt")
    if os.path.exists(best_model):
        model = YOLO(best_model)
        onnx_path = model.export(format="onnx", imgsz=image_size, simplify=True)
        print(f"  ONNX model: {onnx_path}")

        models_dir = os.path.join(ROOT_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        dest = os.path.join(models_dir, "cheque_zones.onnx")
        shutil.copy2(onnx_path, dest)
        print(f"  Copied to: {dest}")
    else:
        print(f"  WARNING: Best model not found at {best_model}")

    print(f"\nTraining complete!")
    print(f"  Best model: {best_model}")
    return best_model


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-nano cheque zone detector (multi-dataset merge)"
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (downloads 3 datasets automatically)",
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        help="Path to pre-merged YOLO dataset (skip download)",
    )
    parser.add_argument("--output", "-o", default=os.path.join(ROOT_DIR, "models"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--model_size", choices=["n", "s"], default="n",
                        help="YOLOv8 size: n=nano (3.2M), s=small (11.2M)")

    args = parser.parse_args()

    dataset_path = args.dataset_path

    if not dataset_path:
        if not args.roboflow_api_key:
            print("ERROR: Provide --roboflow_api_key or --dataset_path")
            print("  Get API key from: https://app.roboflow.com/settings/api")
            sys.exit(1)

        # Download all 3 datasets
        raw_dir = os.path.join(ROOT_DIR, "training", "data", "zone_datasets_raw")
        datasets = download_all_datasets(args.roboflow_api_key, raw_dir)

        if not datasets:
            print("ERROR: No datasets downloaded")
            sys.exit(1)

        # Merge with class remapping
        merged_dir = os.path.join(ROOT_DIR, "training", "data", "zone_dataset_merged")
        merge_datasets(datasets, merged_dir)
        dataset_path = merged_dir

    train_yolov8(
        dataset_path=dataset_path,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
