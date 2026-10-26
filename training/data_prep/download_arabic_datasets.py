#!/usr/bin/env python3
"""
Arabic OCR/VLM Training Dataset Downloader — Downloads production-proven datasets.

Datasets (prioritized by proven benchmark results):
  1. QARI-OCR v0.3: 37K samples, CER 0.061 (Unsloth QLoRA on Qwen2-VL)
  2. AtlasOCR-data: 30K samples (defines our training recipe: r=128, vision unfrozen)
  3. SARD: 743K samples, 5 Arabic fonts (Apache 2.0)
  4. KHATT: 9K handwritten samples (critical for cheque handwriting)
  5. arabic-img2md: 15K samples, image-to-markdown

Combined training strategy (validated by AtlasOCR research):
  Phase 1: General Arabic OCR (QARI-OCR + AtlasOCR + SARD subset + KHATT)
  Phase 2: Cheque-specific fine-tuning (our GT + augmented)

Usage:
    # Download all datasets (recommended)
    python training/download_arabic_datasets.py --all --output training/data/arabic_datasets

    # Download specific dataset
    python training/download_arabic_datasets.py --dataset qari_ocr --max_samples 5000

    # Download and prepare merged training set
    python training/download_arabic_datasets.py --all --prepare_merged --max_total 50000
"""

import argparse
import json
import os
import sys
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# OCR prompt templates for converting datasets to training format
PROMPTS = {
    "ocr_general": (
        "Extract all text from this document image. "
        "Preserve the original language (Arabic or English). "
        "Return the extracted text exactly as written."
    ),
    "ocr_handwritten": (
        "Extract the handwritten Arabic text from this image. "
        "Return the text exactly as written, preserving all characters."
    ),
    "ocr_markdown": (
        "Extract all text from this document image and format as Markdown. "
        "Preserve Arabic text exactly. Maintain headers, lists, and structure."
    ),
}


def _count_jsonl(path: str) -> int:
    """Count lines in a JSONL file (for resume checking)."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _save_image(image, img_path):
    """Save image handling both PIL Image objects and URL strings."""
    from PIL import Image as PILImage
    if isinstance(image, PILImage.Image):
        image.save(img_path)
    elif isinstance(image, str):
        # URL string — download it
        import urllib.request
        urllib.request.urlretrieve(image, img_path)
    elif isinstance(image, bytes):
        with open(img_path, "wb") as f:
            f.write(image)
    else:
        raise ValueError(f"Unknown image type: {type(image)}")


def _make_sample(img_path, text, prompt_key, source):
    """Create a training sample with consistent format."""
    uri = f"file:///{os.path.abspath(img_path).replace(chr(92), '/')}"
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": uri},
                    {"type": "text", "text": PROMPTS[prompt_key]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ],
        "source": source,
    }


def download_qari_ocr(output_dir: str, max_samples: int = 37000) -> str:
    """
    Download QARI-OCR v0.3 (37K Arabic OCR samples).
    Source: NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset
    Proven: CER 0.061 with Unsloth QLoRA on Qwen2-VL-2B
    """
    from datasets import load_dataset

    out_path = os.path.join(output_dir, "qari_ocr.jsonl")
    existing = _count_jsonl(out_path)
    if existing >= max_samples:
        print(f"\n[QARI-OCR] Already have {existing} samples, skipping.")
        return out_path

    print(f"\n[QARI-OCR] Downloading (max {max_samples}, have {existing})...")
    ds = load_dataset(
        "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset",
        split="train",
        streaming=True,
    )

    img_dir = os.path.join(output_dir, "qari_images")
    os.makedirs(img_dir, exist_ok=True)

    count = existing
    mode = "a" if existing > 0 else "w"
    skip = existing  # skip already-downloaded samples
    with open(out_path, mode, encoding="utf-8") as f:
        for sample in ds:
            if count >= max_samples:
                break
            if skip > 0:
                skip -= 1
                continue

            text = sample.get("text", "")
            image = sample.get("image")
            if not text or not image:
                continue

            img_path = os.path.join(img_dir, f"qari_{count:06d}.png")
            try:
                _save_image(image, img_path)
            except Exception as e:
                print(f"  WARNING: Failed to save image {count}: {e}")
                continue

            training_sample = _make_sample(img_path, text, "ocr_general", "qari_ocr")
            f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"  {count}/{max_samples}...")

    print(f"  QARI-OCR: {count} samples -> {out_path}")
    return out_path


def download_atlas_ocr(output_dir: str, max_samples: int = 30000) -> str:
    """Download AtlasOCR-data (30K Arabic OCR samples)."""
    from datasets import load_dataset

    out_path = os.path.join(output_dir, "atlas_ocr.jsonl")
    existing = _count_jsonl(out_path)
    if existing >= max_samples:
        print(f"\n[AtlasOCR] Already have {existing} samples, skipping.")
        return out_path

    print(f"\n[AtlasOCR] Downloading (max {max_samples}, have {existing})...")
    ds = load_dataset("atlasia/atlasOCR-data", split="train", streaming=True)

    img_dir = os.path.join(output_dir, "atlas_images")
    os.makedirs(img_dir, exist_ok=True)

    count = existing
    skip = existing
    with open(out_path, "a" if existing > 0 else "w", encoding="utf-8") as f:
        for sample in ds:
            if count >= max_samples:
                break
            if skip > 0:
                skip -= 1
                continue
            text = sample.get("text", "")
            image = sample.get("image")
            if not text or not image:
                continue
            img_path = os.path.join(img_dir, f"atlas_{count:06d}.png")
            try:
                _save_image(image, img_path)
            except Exception:
                continue
            f.write(json.dumps(_make_sample(img_path, text, "ocr_general", "atlas_ocr"), ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  {count}/{max_samples}...")

    print(f"  AtlasOCR: {count} samples -> {out_path}")
    return out_path


def download_sard(output_dir: str, max_samples: int = 25000) -> str:
    """Download SARD (743K Arabic book-style text, subsample)."""
    from datasets import load_dataset
    import base64
    from io import BytesIO
    from PIL import Image

    out_path = os.path.join(output_dir, "sard.jsonl")
    existing = _count_jsonl(out_path)
    if existing >= max_samples:
        print(f"\n[SARD] Already have {existing} samples, skipping.")
        return out_path

    print(f"\n[SARD] Downloading (max {max_samples}, have {existing})...")
    img_dir = os.path.join(output_dir, "sard_images")
    os.makedirs(img_dir, exist_ok=True)

    try:
        ds = load_dataset("riotu-lab/SARD", split="Amiri", streaming=True)
    except Exception:
        ds = load_dataset("riotu-lab/SARD", streaming=True)
        for split_name in ds:
            ds = ds[split_name]
            break

    count = existing
    skip = existing
    with open(out_path, "a" if existing > 0 else "w", encoding="utf-8") as f:
        for sample in ds:
            if count >= max_samples:
                break
            if skip > 0:
                skip -= 1
                continue
            text = sample.get("chunk", "")
            img_b64 = sample.get("image_base64", "")
            if not text or not img_b64:
                continue
            try:
                img_bytes = base64.b64decode(img_b64)
                image = Image.open(BytesIO(img_bytes))
                img_path = os.path.join(img_dir, f"sard_{count:06d}.png")
                image.save(img_path)
            except Exception:
                continue
            f.write(json.dumps(_make_sample(img_path, text, "ocr_general", "sard"), ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  {count}/{max_samples}...")

    print(f"  SARD: {count} samples -> {out_path}")
    return out_path


def download_khatt(output_dir: str, max_samples: int = 9000) -> str:
    """Download KHATT handwritten Arabic text (9K line images). Critical for cheques.

    Downloads zip files directly from HF and extracts paired .tif/.txt files,
    since the dataset uses a custom loading script no longer supported by `datasets`.
    """
    import zipfile
    from huggingface_hub import hf_hub_download
    from PIL import Image

    out_path = os.path.join(output_dir, "khatt.jsonl")
    existing = _count_jsonl(out_path)
    if existing >= max_samples:
        print(f"\n[KHATT] Already have {existing} samples, skipping.")
        return out_path

    print(f"\n[KHATT] Downloading (max {max_samples}, have {existing})...")

    img_dir = os.path.join(output_dir, "khatt_images")
    os.makedirs(img_dir, exist_ok=True)

    # Download both splits
    zip_files = []
    for split_file in ["data/train.zip", "data/validation.zip"]:
        zip_path = hf_hub_download("benhachem/KHATT", split_file, repo_type="dataset")
        zip_files.append(zip_path)

    count = existing
    skip = existing
    with open(out_path, "a" if existing > 0 else "w", encoding="utf-8") as f:
        for zip_path in zip_files:
            if count >= max_samples:
                break
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Pair .tif images with their .txt ground truth
                names = sorted(zf.namelist())
                tif_files = [n for n in names if n.endswith(".tif")]
                for tif_name in tif_files:
                    if count >= max_samples:
                        break
                    txt_name = tif_name.rsplit(".", 1)[0] + ".txt"
                    if txt_name not in names:
                        continue
                    if skip > 0:
                        skip -= 1
                        continue
                    try:
                        text = zf.read(txt_name).decode("utf-8").strip()
                        if not text:
                            continue
                        img_path = os.path.join(img_dir, f"khatt_{count:06d}.png")
                        with zf.open(tif_name) as img_file:
                            img = Image.open(img_file)
                            img.save(img_path, "PNG")
                    except Exception:
                        continue
                    f.write(json.dumps(_make_sample(img_path, text, "ocr_handwritten", "khatt_handwritten"), ensure_ascii=False) + "\n")
                    count += 1
                    if count % 500 == 0:
                        print(f"  {count}/{max_samples}...")

    print(f"  KHATT: {count} samples -> {out_path}")
    return out_path


def download_arabic_img2md(output_dir: str, max_samples: int = 15000) -> str:
    """Download arabic-img2md (15K page images with Markdown GT)."""
    from datasets import load_dataset

    out_path = os.path.join(output_dir, "arabic_img2md.jsonl")
    existing = _count_jsonl(out_path)
    if existing >= max_samples:
        print(f"\n[arabic-img2md] Already have {existing} samples, skipping.")
        return out_path

    print(f"\n[arabic-img2md] Downloading (max {max_samples}, have {existing})...")
    ds = load_dataset("MohamedRashad/arabic-img2md", split="train", streaming=True)

    img_dir = os.path.join(output_dir, "img2md_images")
    os.makedirs(img_dir, exist_ok=True)

    count = existing
    skip = existing
    with open(out_path, "a" if existing > 0 else "w", encoding="utf-8") as f:
        for sample in ds:
            if count >= max_samples:
                break
            if skip > 0:
                skip -= 1
                continue
            text = sample.get("markdown", sample.get("text", ""))
            image = sample.get("image")
            if not text or not image:
                continue
            img_path = os.path.join(img_dir, f"img2md_{count:06d}.png")
            try:
                _save_image(image, img_path)
            except Exception:
                continue
            f.write(json.dumps(_make_sample(img_path, text, "ocr_markdown", "arabic_img2md"), ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  {count}/{max_samples}...")

    print(f"  arabic-img2md: {count} samples -> {out_path}")
    return out_path


def download_presightai_doc2md(output_dir: str, max_samples: int = 40000) -> str:
    """Download presightai/arabic_doc_to_markdown (40K Arabic govt/banking docs)."""
    from datasets import load_dataset

    out_path = os.path.join(output_dir, "presightai_doc2md.jsonl")
    existing = _count_jsonl(out_path)
    if existing >= max_samples:
        print(f"\n[presightai-doc2md] Already have {existing} samples, skipping.")
        return out_path

    print(f"\n[presightai-doc2md] Downloading (max {max_samples}, have {existing})...")
    ds = load_dataset("presightai/arabic_doc_to_markdown", split="train", streaming=True)

    img_dir = os.path.join(output_dir, "presightai_images")
    os.makedirs(img_dir, exist_ok=True)

    count = existing
    skip = existing
    with open(out_path, "a" if existing > 0 else "w", encoding="utf-8") as f:
        for sample in ds:
            if count >= max_samples:
                break
            if skip > 0:
                skip -= 1
                continue
            text = sample.get("markdown", "")
            image = sample.get("image")
            if not text or not image:
                continue
            img_path = os.path.join(img_dir, f"presightai_{count:06d}.png")
            try:
                _save_image(image, img_path)
            except Exception:
                continue
            f.write(json.dumps(_make_sample(img_path, text, "ocr_markdown", "presightai_doc2md"), ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  {count}/{max_samples}...")

    print(f"  presightai-doc2md: {count} samples -> {out_path}")
    return out_path


def prepare_merged_training_set(
    output_dir: str,
    merged_path: str,
    max_total: int = 50000,
):
    """Merge all downloaded datasets into a single shuffled training file."""
    import random

    random.seed(42)

    print(f"\nPreparing merged training set (max {max_total})...")

    # Composition ratios (validated by AtlasOCR research + presightai addition)
    # Priority: handwritten (KHATT) > domain OCR (QARI, Atlas) > structured (presightai, img2md) > general (SARD)
    ratios = {
        "qari_ocr.jsonl": 0.25,            # 25% - proven Arabic OCR
        "atlas_ocr.jsonl": 0.20,            # 20% - defines training recipe
        "khatt.jsonl": 0.18,                # 18% - handwritten (critical)
        "presightai_doc2md.jsonl": 0.15,    # 15% - Arabic govt/banking docs (NEW)
        "sard.jsonl": 0.12,                 # 12% - font diversity
        "arabic_img2md.jsonl": 0.10,        # 10% - structured output
    }

    all_samples = {}
    for filename in ratios:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            samples = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(line.strip())
            all_samples[filename] = samples
            print(f"  {filename}: {len(samples)} samples")

    # Sample according to ratios
    merged = []
    for filename, ratio in ratios.items():
        if filename not in all_samples:
            continue
        n_target = int(max_total * ratio)
        samples = all_samples[filename]
        if len(samples) <= n_target:
            merged.extend(samples)
        else:
            merged.extend(random.sample(samples, n_target))

    random.shuffle(merged)

    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merged) + "\n")

    print(f"\n  Merged: {len(merged)} samples -> {merged_path}")
    print(f"  Composition:")
    for filename in ratios:
        count = sum(1 for s in merged if f'"source": "{filename.replace(".jsonl", "")}"' in s
                    or f'"source": "{filename.split(".")[0]}"' in s)
        print(f"    {filename}: ~{count}")

    return merged_path


def main():
    parser = argparse.ArgumentParser(
        description="Download production-proven Arabic OCR training datasets"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "training", "data", "arabic_datasets"),
    )
    parser.add_argument("--dataset", choices=[
        "qari_ocr", "atlas_ocr", "sard", "khatt", "arabic_img2md", "presightai_doc2md"
    ], help="Download specific dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per dataset (0=dataset default)")
    parser.add_argument("--prepare_merged", action="store_true",
                        help="After downloading, merge into single training file")
    parser.add_argument("--max_total", type=int, default=50000,
                        help="Max total samples in merged file")

    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Install with: pip install datasets pillow")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    downloaded = []

    defaults = {
        "qari_ocr": 37000,
        "atlas_ocr": 30000,
        "sard": 25000,
        "khatt": 9000,
        "arabic_img2md": 15000,
        # "presightai_doc2md": 40000,
    }

    download_fns = {
        "qari_ocr": download_qari_ocr,
        "atlas_ocr": download_atlas_ocr,
        "sard": download_sard,
        "khatt": download_khatt,
        "arabic_img2md": download_arabic_img2md,
        # "presightai_doc2md": download_presightai_doc2md,
    }

    if args.all:
        for name, fn in download_fns.items():
            max_s = args.max_samples if args.max_samples > 0 else defaults[name]
            try:
                path = fn(args.output, max_s)
                downloaded.append(path)
            except Exception as e:
                print(f"  WARNING: Failed to download {name}: {e}")
    elif args.dataset:
        fn = download_fns[args.dataset]
        max_s = args.max_samples if args.max_samples > 0 else defaults[args.dataset]
        path = fn(args.output, max_s)
        downloaded.append(path)
    else:
        parser.print_help()
        return

    if args.prepare_merged and downloaded:
        merged_path = os.path.join(
            ROOT_DIR, "training", "data", "arabic_merged_train.jsonl"
        )
        prepare_merged_training_set(args.output, merged_path, args.max_total)

    print(f"\nDone. Downloaded {len(downloaded)} dataset(s).")


if __name__ == "__main__":
    main()