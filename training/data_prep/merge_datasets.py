#!/usr/bin/env python3
"""
Dataset Merger — Combines cheque training data + QARI-OCR into a single training set.

Training data composition strategy (validated by AtlasOCR recipe):
- 80% domain-specific (cheque GT + augmented cheques)
- 20% Arabic OCR generalization (QARI-OCR subset)

This ratio prevents catastrophic forgetting of general Arabic OCR while
focusing the model on cheque-specific extraction patterns.

Usage:
    python training/merge_datasets.py \
        --cheque_data training/data/cheque_train.jsonl \
        --qari_data training/data/qari_ocr_subset.jsonl \
        --output training/data/merged_train.jsonl \
        --cheque_ratio 0.8
"""

import argparse
import json
import os
import random
import sys
from typing import List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def merge_datasets(
    cheque_path: str,
    qari_path: str,
    output_path: str,
    cheque_ratio: float = 0.8,
    max_total: int = 0,
    seed: int = 42,
) -> int:
    """
    Merge cheque and QARI-OCR datasets.

    Args:
        cheque_path: Path to cheque training JSONL
        qari_path: Path to QARI-OCR JSONL
        output_path: Output merged JSONL
        cheque_ratio: Ratio of cheque samples in final dataset
        max_total: Maximum total samples (0 = unlimited)
        seed: Random seed for shuffling

    Returns:
        Number of samples in merged dataset
    """
    random.seed(seed)

    cheque_samples = load_jsonl(cheque_path) if os.path.exists(cheque_path) else []
    qari_samples = load_jsonl(qari_path) if os.path.exists(qari_path) else []

    print(f"Cheque samples: {len(cheque_samples)}")
    print(f"QARI-OCR samples: {len(qari_samples)}")

    if not cheque_samples:
        print("WARNING: No cheque samples found. Using only QARI-OCR data.")
        merged = qari_samples
    elif not qari_samples:
        print("WARNING: No QARI-OCR samples found. Using only cheque data.")
        merged = cheque_samples
    else:
        # Calculate target counts based on ratio
        if max_total > 0:
            n_cheque = min(int(max_total * cheque_ratio), len(cheque_samples))
            n_qari = min(max_total - n_cheque, len(qari_samples))
        else:
            # Use all cheque data, sample QARI to match ratio
            n_cheque = len(cheque_samples)
            n_qari = min(
                int(n_cheque * (1 - cheque_ratio) / cheque_ratio),
                len(qari_samples),
            )

        print(f"Target: {n_cheque} cheque + {n_qari} QARI-OCR = {n_cheque + n_qari} total")

        # Sample
        selected_cheque = cheque_samples[:n_cheque]
        selected_qari = random.sample(qari_samples, min(n_qari, len(qari_samples)))

        merged = selected_cheque + selected_qari

    # Shuffle
    random.shuffle(merged)

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in merged:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Merged dataset: {len(merged)} samples -> {output_path}")

    # Print composition
    cheque_count = sum(
        1 for s in merged
        if any(
            "cheque" in str(m.get("content", "")).lower()
            or "Payee_Name" in str(m.get("content", ""))
            for m in s.get("messages", [])
        )
    )
    print(f"Composition: ~{cheque_count} cheque ({cheque_count/len(merged)*100:.0f}%), "
          f"~{len(merged)-cheque_count} QARI-OCR ({(len(merged)-cheque_count)/len(merged)*100:.0f}%)")

    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Merge training datasets")
    parser.add_argument(
        "--cheque_data",
        default=os.path.join(ROOT_DIR, "training", "data", "cheque_train.jsonl"),
    )
    parser.add_argument(
        "--qari_data",
        default=os.path.join(ROOT_DIR, "training", "data", "qari_ocr_subset.jsonl"),
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "training", "data", "merged_train.jsonl"),
    )
    parser.add_argument(
        "--cheque_ratio",
        type=float,
        default=0.8,
        help="Ratio of cheque samples (default: 0.8)",
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=0,
        help="Max total samples, 0=unlimited (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    merge_datasets(
        cheque_path=args.cheque_data,
        qari_path=args.qari_data,
        output_path=args.output,
        cheque_ratio=args.cheque_ratio,
        max_total=args.max_total,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
