#!/usr/bin/env python3
"""
Fix GT payee_name for training:
- Inward cheques: payee_name already null (no change needed)
- Outward cheques with "/": split into Arabic (primary) and English (secondary)
- Outward cheques without "/": keep as-is

Also updates the augmented manifest to match.

Usage:
    python training/fix_gt_payee.py --gt_dir benchmark/ground_truth/
    python training/fix_gt_payee.py --gt_dir benchmark/ground_truth/ --dry_run
"""

import argparse
import io
import json
import os
import re
import sys

# Force UTF-8 output on Windows (cp1252 can't encode Arabic)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def has_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))


def split_payee_name(payee_name: str) -> dict:
    """
    Split bilingual payee_name into Arabic and English parts.

    Returns dict with:
        payee_name: Arabic part (primary for extraction)
        payee_name_english: English part (secondary/transliteration)
    """
    if not payee_name or "/" not in payee_name:
        return {"payee_name": payee_name, "payee_name_english": None}

    parts = [p.strip() for p in payee_name.split("/", 1)]

    if len(parts) != 2:
        return {"payee_name": payee_name, "payee_name_english": None}

    part_a, part_b = parts

    # Detect which part is Arabic
    a_arabic = has_arabic(part_a)
    b_arabic = has_arabic(part_b)

    if a_arabic and not b_arabic:
        # Standard: "Arabic / English"
        return {"payee_name": part_a, "payee_name_english": part_b}
    elif b_arabic and not a_arabic:
        # Reversed: "English / Arabic" (e.g., outward_8)
        return {"payee_name": part_b, "payee_name_english": part_a}
    elif a_arabic and b_arabic:
        # Both Arabic — keep first as primary
        return {"payee_name": part_a, "payee_name_english": None}
    else:
        # Both English — keep as-is
        return {"payee_name": payee_name, "payee_name_english": None}


def fix_gt_file(filepath: str, dry_run: bool = False) -> dict:
    """Fix a single GT JSON file. Returns change info."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("document_id", os.path.basename(filepath))
    category = data.get("category", "")
    extraction = data.get("extraction", {})
    original_payee = extraction.get("payee_name")

    changes = {"document_id": doc_id, "category": category, "changed": False}

    if category == "inward":
        # Inward: payee_name should be null (no payee info available)
        if original_payee is not None:
            changes["changed"] = True
            changes["action"] = f"Set payee_name to null (was: {original_payee})"
            if not dry_run:
                extraction["payee_name"] = None
        else:
            changes["action"] = "Already null"
    elif category == "outward" and original_payee and "/" in original_payee:
        # Outward with bilingual name: split
        result = split_payee_name(original_payee)
        changes["changed"] = True
        changes["action"] = (
            f"Split: '{original_payee}' -> "
            f"Arabic='{result['payee_name']}', "
            f"English='{result['payee_name_english']}'"
        )
        if not dry_run:
            extraction["payee_name"] = result["payee_name"]
            extraction["payee_name_english"] = result["payee_name_english"]
    else:
        changes["action"] = f"No change (payee={original_payee})"

    if not dry_run and changes["changed"]:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return changes


def fix_augmented_manifest(manifest_path: str, dry_run: bool = False) -> int:
    """Fix payee_name in augmented manifest.json."""
    if not os.path.exists(manifest_path):
        print(f"  Augmented manifest not found: {manifest_path}")
        return 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    fixed = 0
    for entry in entries:
        extraction = entry.get("extraction", {})
        # Manifest entries may have "category" or we infer from "source_document_id"
        category = entry.get("category", "")
        if not category:
            src_id = entry.get("source_document_id", "")
            if src_id.startswith("inward"):
                category = "inward"
            elif src_id.startswith("outward"):
                category = "outward"
        original_payee = extraction.get("payee_name")

        if category == "inward" and original_payee is not None:
            extraction["payee_name"] = None
            fixed += 1
        elif category == "outward" and original_payee and "/" in original_payee:
            result = split_payee_name(original_payee)
            extraction["payee_name"] = result["payee_name"]
            extraction["payee_name_english"] = result["payee_name_english"]
            fixed += 1

    if not dry_run and fixed > 0:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Fix GT payee_name for training")
    parser.add_argument("--gt_dir", "-g", default="benchmark/ground_truth/")
    parser.add_argument("--augmented_manifest", "-a",
                        default="training/data/augmented/manifest.json")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print changes without modifying files")

    args = parser.parse_args()

    print("=" * 60)
    print(f"  FIX GT PAYEE_NAME {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    # Fix GT files
    gt_files = sorted([
        f for f in os.listdir(args.gt_dir)
        if f.endswith(".json") and not f.startswith("_")
    ])

    for fname in gt_files:
        filepath = os.path.join(args.gt_dir, fname)
        result = fix_gt_file(filepath, dry_run=args.dry_run)
        marker = "*" if result["changed"] else " "
        print(f"  [{marker}] {result['document_id']:15s} ({result['category']:7s}): "
              f"{result['action']}")

    # Fix augmented manifest
    print(f"\n  Augmented manifest: {args.augmented_manifest}")
    n_fixed = fix_augmented_manifest(args.augmented_manifest, dry_run=args.dry_run)
    print(f"  Fixed {n_fixed} entries in augmented manifest")

    if args.dry_run:
        print(f"\n  DRY RUN — no files modified. Remove --dry_run to apply.")
    else:
        print(f"\n  Done. Re-run prepare_training_data.py to regenerate training JSONL.")


if __name__ == "__main__":
    main()
