#!/usr/bin/env python3
"""
Sherif v3 Fine-Tuning Data Preparation — Task-Specific Payee Field Isolation.

Transforms sherif v3 from "read all text" to "read ONLY handwritten payee name"
by creating training data from three sources at validated ratios:

  ~35% Cheque payee crops (83 cheques, 31 Arabic + 52 English)
  ~40% KHATT Arabic handwriting lines (diverse writers, 1000+ hands)
  ~20% English cheque payees (prevent regression)
  ~5%  Negative augmentation (template-only crops → "NOT_FOUND")

Key differences from VLM training data (cheque_train_v15_server.jsonl):
  - VLM data: full cheque image → all 8 fields (multi-field structured extraction)
  - Sherif data: payee CROP image → single field (payee name only)
  - Sherif prompt: "Read ONLY the handwritten name" (field isolation task)
  - KHATT data: handwriting line image → text transcription (Arabic reinforcement)

Output format: Unsloth-compatible JSONL (same as existing training data).

Kaggle-compatible: downloads KHATT via huggingface_hub, generates crops from
local cheque images, no external API calls.

Usage:
    # Prepare all data (downloads KHATT ~500MB)
    python training/prepare_sherif_ft_data.py \
        --gt_json cheque-poc/test-data/ground_truth_v14.json \
        --image_base cheque-poc/test-data \
        --output training/data/sherif_ft_train.jsonl \
        --khatt_samples 250

    # Dry run (cheque data only, no KHATT download)
    python training/prepare_sherif_ft_data.py \
        --gt_json cheque-poc/test-data/ground_truth_v14.json \
        --image_base cheque-poc/test-data \
        --output training/data/sherif_ft_train.jsonl \
        --skip_khatt
"""

import argparse
import json
import os
import random
import sys
import zipfile
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Prompts — task-specific field isolation (NOT generic OCR)
# ---------------------------------------------------------------------------

PROMPTS_PAYEE = [
    # Primary prompt (matches baseline_script.py sherif invocation)
    "Read the handwritten name in this image. Return ONLY the name, nothing else.",
    # Arabic-emphasis variant
    "اقرأ الاسم المكتوب بخط اليد في هذه الصورة. أعد الاسم فقط، لا شيء آخر.",
    # Explicit field-isolation variant
    (
        "This is a cropped region from a bank cheque. "
        "Read ONLY the handwritten text. Ignore any printed text, "
        "template words, or bank logos. Return just the handwritten name."
    ),
    # Minimal variant
    "What is the handwritten name in this image?",
]

PROMPT_KHATT = [
    # KHATT uses pure Arabic handwriting transcription prompts
    "Read the handwritten Arabic text in this image. Return the text exactly as written.",
    "اقرأ النص العربي المكتوب بخط اليد. أعد النص كما هو مكتوب بالضبط.",
    "Extract the handwritten Arabic text from this image.",
]


def _make_sample(image_uri, text, prompt, source, meta=None):
    """Create Unsloth-compatible training sample."""
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ],
        "_source": source,
    }
    if meta:
        sample["_meta"] = meta
    return sample


def _image_to_uri(path, server_base=None):
    """Convert image path to file:// URI."""
    if server_base:
        fname = Path(path).name
        return f"file://{server_base.rstrip('/')}/images/{fname}"
    abs_path = os.path.abspath(path).replace("\\", "/")
    return f"file:///{abs_path}"


# ---------------------------------------------------------------------------
# Step 1: Cheque payee crop samples
# ---------------------------------------------------------------------------

def prepare_cheque_payee_samples(gt_json, image_base, server_base=None):
    """Generate payee-only training samples from cheque GT.

    For each cheque:
    - Crop the payee region from the full cheque image
    - Create samples with multiple prompt variants
    - Arabic cheques get 5x augmentation, English get 2x

    Returns list of (sample_dict, is_arabic) tuples.
    """
    import cv2
    from PIL import Image

    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # ROI for payee crop (same as baseline_script.py V7)
    ROI_PAYEE = (0.0, 0.28, 1.0, 0.63)

    samples = []
    crop_dir = os.path.join(os.path.dirname(gt_json), "payee_ft_crops")
    os.makedirs(crop_dir, exist_ok=True)

    for cheque in gt["cheques"]:
        fields = cheque.get("fields", {})
        payee_name = fields.get("payee_name", "")
        if not payee_name:
            continue

        image_rel = cheque.get("image_path", "")
        image_path = os.path.join(image_base, image_rel)
        if not os.path.exists(image_path):
            print(f"  WARNING: Image not found: {image_path}")
            continue

        # Detect Arabic
        is_arabic = any(0x0600 <= ord(ch) <= 0x06FF for ch in payee_name)

        # Crop payee region
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Upscale small images (same as baseline_script.py)
        if w < 1400:
            scale = 1400 / w
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        x1 = int(ROI_PAYEE[0] * w)
        y1 = int(ROI_PAYEE[1] * h)
        x2 = int(ROI_PAYEE[2] * w)
        y2 = int(ROI_PAYEE[3] * h)
        crop = img[y1:y2, x1:x2]

        # Ensure minimum height
        ch, cw = crop.shape[:2]
        if ch < 200:
            scale = 200 / ch
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        # Save crop
        crop_name = image_rel.replace("/", "_").replace("\\", "_")
        crop_name = os.path.splitext(crop_name)[0] + "_payee.png"
        crop_path = os.path.join(crop_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        # Generate training samples with prompt variants
        # Arabic: more variants (5x) because they're the bottleneck
        # English: fewer variants (2x) just to prevent regression
        n_variants = 5 if is_arabic else 2
        prompts_to_use = PROMPTS_PAYEE[:n_variants]

        for prompt in prompts_to_use:
            uri = _image_to_uri(crop_path, server_base)
            sample = _make_sample(
                uri, payee_name, prompt,
                source="cheque_payee",
                meta={
                    "cheque_id": cheque.get("id", image_rel),
                    "category": cheque.get("category", ""),
                    "is_arabic": is_arabic,
                },
            )
            samples.append((sample, is_arabic))

        # Arabic: also generate augmented crops (slight ROI perturbation)
        if is_arabic:
            for aug_idx, (dy1, dy2, dx2) in enumerate([
                (-0.02, 0.02, 0.0),   # taller crop
                (0.02, -0.02, -0.05),  # tighter crop
                (-0.01, 0.01, -0.10),  # narrower (exclude amount area)
            ]):
                ay1 = max(0, int((ROI_PAYEE[1] + dy1) * h))
                ay2 = min(h, int((ROI_PAYEE[3] + dy2) * h))
                ax2 = max(x1 + 100, int((ROI_PAYEE[2] + dx2) * w))
                aug_crop = img[ay1:ay2, x1:ax2]
                ach, acw = aug_crop.shape[:2]
                if ach < 100 or acw < 100:
                    continue
                if ach < 200:
                    s = 200 / ach
                    aug_crop = cv2.resize(aug_crop, None, fx=s, fy=s,
                                          interpolation=cv2.INTER_CUBIC)

                aug_name = f"{os.path.splitext(crop_name)[0]}_aug{aug_idx}.png"
                aug_path = os.path.join(crop_dir, aug_name)
                cv2.imwrite(aug_path, aug_crop)

                prompt = random.choice(PROMPTS_PAYEE)
                uri = _image_to_uri(aug_path, server_base)
                sample = _make_sample(
                    uri, payee_name, prompt,
                    source="cheque_payee_aug",
                    meta={
                        "cheque_id": cheque.get("id", image_rel),
                        "augmentation": f"roi_perturb_{aug_idx}",
                        "is_arabic": True,
                    },
                )
                samples.append((sample, True))

    ar_count = sum(1 for _, is_ar in samples if is_ar)
    en_count = len(samples) - ar_count
    print(f"  Cheque payee samples: {len(samples)} (Arabic: {ar_count}, English: {en_count})")
    return samples


# ---------------------------------------------------------------------------
# Step 2: KHATT Arabic handwriting samples
# ---------------------------------------------------------------------------

def download_and_prepare_khatt(output_dir, max_samples=250, server_base=None):
    """Download KHATT handwritten Arabic line images and create training samples.

    Downloads only the required quantity from HuggingFace Hub.
    KHATT contains ~9000 line-level Arabic handwriting images from 1000 writers.

    Returns list of (sample_dict, True) tuples (all Arabic).
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image

    khatt_dir = os.path.join(output_dir, "khatt_images")
    os.makedirs(khatt_dir, exist_ok=True)

    print(f"  Downloading KHATT (max {max_samples} samples)...")

    samples = []
    count = 0

    # Download both splits
    for split_file in ["data/train.zip", "data/validation.zip"]:
        if count >= max_samples:
            break
        try:
            zip_path = hf_hub_download("benhachem/KHATT", split_file, repo_type="dataset")
        except Exception as e:
            print(f"    WARNING: Failed to download {split_file}: {e}")
            continue

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = sorted(zf.namelist())
            tif_files = [n for n in names if n.endswith(".tif")]

            for tif_name in tif_files:
                if count >= max_samples:
                    break

                txt_name = tif_name.rsplit(".", 1)[0] + ".txt"
                if txt_name not in names:
                    continue

                try:
                    text = zf.read(txt_name).decode("utf-8").strip()
                    if not text or len(text) < 2:
                        continue

                    img_path = os.path.join(khatt_dir, f"khatt_{count:06d}.png")
                    with zf.open(tif_name) as img_file:
                        img = Image.open(img_file)
                        img.save(img_path, "PNG")
                except Exception:
                    continue

                prompt = random.choice(PROMPT_KHATT)
                uri = _image_to_uri(img_path, server_base)
                sample = _make_sample(
                    uri, text, prompt,
                    source="khatt_handwritten",
                    meta={"khatt_id": tif_name},
                )
                samples.append((sample, True))
                count += 1

    print(f"  KHATT samples: {count}")
    return samples


# ---------------------------------------------------------------------------
# Step 3: Negative samples (template-only → NOT_FOUND)
# ---------------------------------------------------------------------------

def prepare_negative_samples(gt_json, image_base, server_base=None):
    """Create negative samples: bank/MICR region crops where payee should be NOT_FOUND.

    Teaches the model that NOT all crops contain a handwritten name.
    Uses bank logo area and MICR strip — regions that contain printed text but no payee.
    """
    import cv2

    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    ROI_BANK = (0.0, 0.0, 0.65, 0.3)
    ROI_MICR = (0.0, 0.75, 1.0, 1.0)

    samples = []
    crop_dir = os.path.join(os.path.dirname(gt_json), "payee_ft_crops")
    os.makedirs(crop_dir, exist_ok=True)

    # Take only a subset of cheques for negatives
    cheques = gt["cheques"][:20]

    for cheque in cheques:
        image_rel = cheque.get("image_path", "")
        image_path = os.path.join(image_base, image_rel)
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w < 1400:
            scale = 1400 / w
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        # Bank region (no payee here)
        bx1, by1 = int(ROI_BANK[0] * w), int(ROI_BANK[1] * h)
        bx2, by2 = int(ROI_BANK[2] * w), int(ROI_BANK[3] * h)
        bank_crop = img[by1:by2, bx1:bx2]

        base_name = image_rel.replace("/", "_").replace("\\", "_")
        bank_path = os.path.join(crop_dir, f"{os.path.splitext(base_name)[0]}_neg_bank.png")
        cv2.imwrite(bank_path, bank_crop)

        prompt = random.choice(PROMPTS_PAYEE)
        uri = _image_to_uri(bank_path, server_base)
        sample = _make_sample(
            uri, "NOT_FOUND", prompt,
            source="negative",
            meta={"region": "bank", "cheque_id": cheque.get("id", image_rel)},
        )
        samples.append((sample, False))

    print(f"  Negative samples: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Step 4: Merge and shuffle at target ratio
# ---------------------------------------------------------------------------

def merge_and_shuffle(cheque_samples, khatt_samples, negative_samples, target_total=630):
    """Merge at approximately the designed ratio, then shuffle.

    Target ratio:
      ~35% cheque payee (Arabic + augmented)
      ~40% KHATT Arabic handwriting
      ~20% cheque payee (English)
      ~5%  negative samples
    """
    ar_cheque = [s for s, is_ar in cheque_samples if is_ar]
    en_cheque = [s for s, is_ar in cheque_samples if not is_ar]
    khatt = [s for s, _ in khatt_samples]
    negatives = [s for s, _ in negative_samples]

    print(f"\n  Available: ar_cheque={len(ar_cheque)}, en_cheque={len(en_cheque)}, "
          f"khatt={len(khatt)}, negatives={len(negatives)}")

    # Compute targets
    n_ar_cheque = min(len(ar_cheque), int(target_total * 0.35))
    n_khatt = min(len(khatt), int(target_total * 0.40))
    n_en_cheque = min(len(en_cheque), int(target_total * 0.20))
    n_negative = min(len(negatives), int(target_total * 0.05))

    # Sample (or take all if fewer available)
    random.seed(42)
    final = []
    final.extend(random.sample(ar_cheque, n_ar_cheque) if len(ar_cheque) > n_ar_cheque else ar_cheque)
    final.extend(random.sample(khatt, n_khatt) if len(khatt) > n_khatt else khatt)
    final.extend(random.sample(en_cheque, n_en_cheque) if len(en_cheque) > n_en_cheque else en_cheque)
    final.extend(random.sample(negatives, n_negative) if len(negatives) > n_negative else negatives)

    random.shuffle(final)

    # Report actual composition
    sources = {}
    for s in final:
        src = s.get("_source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"  Final dataset: {len(final)} samples")
    print(f"  Composition:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = count / len(final) * 100
        print(f"    {src:<25} {count:>4} ({pct:.1f}%)")

    return final


# ---------------------------------------------------------------------------
# Step 5: Also reuse existing V15 JSONL cheque data (for compatibility check)
# ---------------------------------------------------------------------------

def check_v15_compatibility(v15_path):
    """Check if existing V15 JSONL data can be partially reused.

    V15 data is multi-field cheque extraction (full image → 8 fields).
    We CANNOT directly reuse it for sherif (single-field payee-only).
    But we can extract payee GT values for validation.
    """
    if not os.path.exists(v15_path):
        print(f"  V15 JSONL not found at {v15_path}")
        return

    count = 0
    payee_count = 0
    with open(v15_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            obj = json.loads(line)
            # V15 assistant content has "Payee_Name: ..." in multi-field format
            msgs = obj.get("messages", [])
            for m in msgs:
                if m.get("role") == "assistant":
                    content = m.get("content", [])
                    if isinstance(content, list) and content:
                        text = content[0].get("text", "")
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""
                    if "Payee_Name:" in text:
                        payee_count += 1

    print(f"\n  V15 JSONL compatibility check:")
    print(f"    Total samples: {count}")
    print(f"    Samples with Payee_Name: {payee_count}")
    print(f"    Format: multi-field (full cheque -> 8 fields)")
    print(f"    Compatible for sherif FT: NO (need single-field payee-only)")
    print(f"    Can extract payee GT for validation: YES")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for sherif v3 payee field-isolation fine-tuning"
    )
    parser.add_argument("--gt_json", required=True,
                        help="Path to ground_truth_v14.json")
    parser.add_argument("--image_base", required=True,
                        help="Base directory for cheque images")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--khatt_samples", type=int, default=250,
                        help="Number of KHATT samples to include (default: 250)")
    parser.add_argument("--target_total", type=int, default=630,
                        help="Target total training samples (default: 630)")
    parser.add_argument("--skip_khatt", action="store_true",
                        help="Skip KHATT download (cheque-only for testing)")
    parser.add_argument("--server_base", type=str, default=None,
                        help="Server base dir for file:// URIs (e.g. /kaggle/working)")
    parser.add_argument("--check_v15", type=str, default=None,
                        help="Path to V15 JSONL for compatibility check")
    args = parser.parse_args()

    random.seed(42)
    print("=" * 60)
    print("SHERIF V3 FINE-TUNING DATA PREPARATION")
    print("Task: Payee field isolation (handwritten name only)")
    print("=" * 60)

    # Check V15 compatibility if requested
    if args.check_v15:
        check_v15_compatibility(args.check_v15)

    # Step 1: Cheque payee crops
    print("\n[1/4] Preparing cheque payee crop samples...")
    cheque_samples = prepare_cheque_payee_samples(
        args.gt_json, args.image_base, args.server_base)

    # Step 2: KHATT Arabic handwriting
    khatt_samples = []
    if not args.skip_khatt:
        print(f"\n[2/4] Downloading and preparing KHATT ({args.khatt_samples} samples)...")
        khatt_dir = os.path.dirname(args.output)
        khatt_samples = download_and_prepare_khatt(
            khatt_dir, max_samples=args.khatt_samples, server_base=args.server_base)
    else:
        print("\n[2/4] KHATT: SKIPPED (--skip_khatt)")

    # Step 3: Negative samples
    print("\n[3/4] Preparing negative samples...")
    negative_samples = prepare_negative_samples(
        args.gt_json, args.image_base, args.server_base)

    # Step 4: Merge at target ratio
    print(f"\n[4/4] Merging at target ratio (total ~{args.target_total})...")
    final_samples = merge_and_shuffle(
        cheque_samples, khatt_samples, negative_samples,
        target_total=args.target_total)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n  Output: {args.output}")
    print(f"  Total samples: {len(final_samples)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
