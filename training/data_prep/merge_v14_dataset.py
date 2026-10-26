#!/usr/bin/env python3
"""
Merge v14 cheque training data + Arabic OCR datasets → 4K-5K total.

All 83 cheques included (249 samples with 3 prompt variants).
Arabic OCR padded to reach 4K-5K total, KHATT highest weight.

Run on server after uploading cheque_train_v14.jsonl:
    python merge_v14_dataset.py

Output: merged_v14_train.jsonl (ready for train_qlora.py)
"""

import json
import os
import random

random.seed(42)

TARGET_TOTAL = 4500  # Aim for 4K-5K balanced dataset

# ── Load cheque data ──────────────────────────────────────────────────────────
cheque_path = "cheque_train_v14.jsonl"
if not os.path.exists(cheque_path):
    # Fallback to old name
    cheque_path = "cheque_train.jsonl"

with open(cheque_path, "r", encoding="utf-8") as f:
    cheque = [l.strip() for l in f if l.strip()]

print(f"Cheque samples: {len(cheque)}")

# ── Load Arabic datasets ─────────────────────────────────────────────────────
n_arabic_needed = TARGET_TOTAL - len(cheque)
print(f"Arabic samples needed: {n_arabic_needed}")

# Weights — KHATT highest (handwritten Arabic, closest to cheque handwriting)
arabic_dir = "./arabic_datasets"
arabic_sources = [
    ("khatt",          40),   # 40% — handwritten Arabic, HIGHEST weight
    ("qari_ocr",       20),   # 20% — printed Arabic OCR
    ("atlas_ocr",      15),   # 15% — general Arabic
    ("sard",           15),   # 15% — font diversity
    ("arabic_img2md",  10),   # 10% — structured output
]
total_weight = sum(w for _, w in arabic_sources)

arabic_samples = []

for source, weight in arabic_sources:
    # Try multiple naming patterns
    candidates = [
        os.path.join(arabic_dir, f"{source}.jsonl"),
        os.path.join(arabic_dir, f"{source}_train.jsonl"),
        os.path.join(arabic_dir, f"{source}_merged.jsonl"),
    ]
    src_path = None
    for c in candidates:
        if os.path.exists(c):
            src_path = c
            break

    if not src_path:
        print(f"  WARNING: {source} not found, skipping")
        continue

    with open(src_path, "r", encoding="utf-8") as f:
        src_data = [l.strip() for l in f if l.strip()]

    n = int(n_arabic_needed * weight / total_weight)
    sampled = random.sample(src_data, min(len(src_data), n))
    arabic_samples.extend(sampled)
    print(f"  {source}: {len(sampled)}/{len(src_data)} (weight {weight}%)")

# If we didn't get enough from individual files, try merged fallback
if len(arabic_samples) < n_arabic_needed * 0.8:
    merged_path = os.path.join(arabic_dir, "merged_arabic_train.jsonl")
    if not os.path.exists(merged_path):
        merged_path = "arabic_merged_train.jsonl"  # old location
    if os.path.exists(merged_path):
        with open(merged_path, "r", encoding="utf-8") as f:
            all_arabic = [l.strip() for l in f if l.strip()]
        shortfall = n_arabic_needed - len(arabic_samples)
        # Don't re-sample ones we already have
        existing = set(arabic_samples)
        available = [s for s in all_arabic if s not in existing]
        extra = random.sample(available, min(len(available), shortfall))
        arabic_samples.extend(extra)
        print(f"  Merged fallback: +{len(extra)} samples")

# ── Merge and shuffle ─────────────────────────────────────────────────────────
merged = cheque + arabic_samples
random.shuffle(merged)

output_path = "merged_v14_train.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(merged) + "\n")

total = len(merged)
cheque_pct = len(cheque) / total * 100 if total else 0
arabic_pct = len(arabic_samples) / total * 100 if total else 0

print(f"\n{'='*50}")
print(f"  Total: {total}")
print(f"  Cheque: {len(cheque)} ({cheque_pct:.1f}%)")
print(f"  Arabic: {len(arabic_samples)} ({arabic_pct:.1f}%)")
print(f"  Saved: {output_path}")
print(f"{'='*50}")
print(f"\nNext: python train_qlora.py --data {output_path} --warm_start U4RASD/ar-ms-baseline --output ./cheque-adapter-v14 --epochs 3")
