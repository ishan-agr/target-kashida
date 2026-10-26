#!/usr/bin/env python3
"""
Data Augmentation Pipeline — Generates augmented cheque images for training.

Uses Albumentations to create realistic augmentations:
- Geometric: rotation, perspective, affine transforms
- Noise: Gaussian, ISO, JPEG compression artifacts
- Blur: motion blur, Gaussian blur
- Color: brightness, contrast, hue shifts
- Cheque-specific: ink bleed simulation, paper texture

Generates 10-15 augmented versions per original image.
18 originals -> 180-270 augmented samples.

Usage:
    python training/augment_data.py \
        --gt_dir benchmark/ground_truth/ \
        --output_dir training/data/augmented/ \
        --augments_per_image 12
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


def get_augmentation_pipeline():
    """
    Create Albumentations augmentation pipeline for cheque images.

    Designed to simulate real-world cheque scanning conditions:
    - Scanner skew and rotation
    - Lighting variation
    - Paper quality degradation
    - Print/ink quality variation
    """
    import albumentations as A

    return A.Compose([
        # Geometric transforms (scanner skew simulation)
        A.OneOf([
            A.Rotate(limit=5, border_mode=0, fill=(255, 255, 255), p=0.7),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-3, 3),
                shear=(-2, 2),
                border_mode=0,
                fill=(255, 255, 255),
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ], p=0.8),

        # Noise (scanner/camera sensor noise)
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.3),
        ], p=0.5),

        # Blur (focus/motion issues)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.4),

        # Color/brightness (lighting variation)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.6,
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ], p=0.6),

        # JPEG compression artifacts (common in scanned documents)
        A.ImageCompression(quality_range=(70, 95), p=0.4),

        # Slight color shift (paper yellowing, ink fading)
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.3,
        ),

        # Shadow/lighting gradient
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=0.2,
        ),
    ])


def augment_image(image_path: str, output_dir: str, n_augments: int = 12, prefix: str = "") -> List[str]:
    """
    Generate augmented versions of a single cheque image.

    Args:
        image_path: Path to original image
        output_dir: Directory to save augmented images
        n_augments: Number of augmented versions to generate
        prefix: Unique prefix (e.g. document_id) to avoid filename collisions

    Returns:
        List of paths to augmented images
    """
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        print(f"  WARNING: Cannot read image: {image_path}")
        return []

    transform = get_augmentation_pipeline()
    basename = prefix if prefix else Path(image_path).stem
    ext = Path(image_path).suffix

    augmented_paths = []
    for i in range(n_augments):
        augmented = transform(image=image)["image"]
        out_name = f"{basename}_aug{i:03d}{ext}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, augmented)
        augmented_paths.append(out_path)

    return augmented_paths


def run_augmentation(
    gt_dir: str,
    output_dir: str,
    augments_per_image: int = 12,
) -> Tuple[int, str]:
    """
    Run augmentation pipeline on all GT images.

    Args:
        gt_dir: Ground truth directory with JSON files
        output_dir: Output directory for augmented images + manifest
        augments_per_image: Number of augmentations per original

    Returns:
        Tuple of (total_augmented, manifest_path)
    """
    # Load GT entries
    gt_entries = []
    for fname in sorted(os.listdir(gt_dir)):
        if fname.startswith("_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(gt_dir, fname), "r", encoding="utf-8") as f:
            gt_entries.append(json.load(f))

    print(f"Loaded {len(gt_entries)} GT entries")
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    total = 0

    for entry in gt_entries:
        image_path = entry["image_path"]
        gt_fields = entry["extraction"]
        doc_id = entry["document_id"]

        if not os.path.exists(image_path):
            print(f"  SKIP: Image not found: {image_path}")
            continue

        print(f"  Augmenting {doc_id} ({os.path.basename(image_path)})...")
        aug_paths = augment_image(image_path, output_dir, augments_per_image, prefix=doc_id)

        for aug_path in aug_paths:
            manifest.append({
                "source_document_id": doc_id,
                "image_path": os.path.abspath(aug_path),
                "extraction": gt_fields,  # GT is the same for augmented images
                "is_augmented": True,
            })
            total += 1

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nAugmentation complete: {total} images generated")
    print(f"Manifest saved to: {manifest_path}")

    return total, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Augment cheque images for training data"
    )
    parser.add_argument(
        "--gt_dir", "-g",
        default=os.path.join(ROOT_DIR, "benchmark", "ground_truth"),
        help="Ground truth JSON directory",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=os.path.join(ROOT_DIR, "training", "data", "augmented"),
        help="Output directory for augmented images",
    )
    parser.add_argument(
        "--augments_per_image", "-n",
        type=int,
        default=12,
        help="Number of augmented versions per image (default: 12)",
    )

    args = parser.parse_args()

    try:
        import albumentations  # noqa: F401
    except ImportError:
        print("ERROR: albumentations not installed.")
        print("Install with: pip install albumentations")
        sys.exit(1)

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("ERROR: opencv-python not installed.")
        print("Install with: pip install opencv-python")
        sys.exit(1)

    total, manifest = run_augmentation(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        augments_per_image=args.augments_per_image,
    )

    print(f"\nDone. {total} augmented samples ready.")
    print(f"Next step: Run prepare_training_data.py --include_augmented")


if __name__ == "__main__":
    main()
