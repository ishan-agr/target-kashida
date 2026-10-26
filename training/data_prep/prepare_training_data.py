#!/usr/bin/env python3
"""
Training Data Preparation — Converts cheque GT + images into Unsloth/SWIFT training format.

Creates JSONL training data for Qwen3-VL-8B QLoRA fine-tuning.
Each sample is a conversation with an image and expected extraction output.

Supports two GT formats:
  A) v14 JSON (single file with 'cheques' array — 83 cheques)
  B) Legacy per-file GT directory (benchmark/ground_truth/*.json)

Output format (Unsloth/Qwen3-VL compatible):
{
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": "file:///path/to/image.png"},
            {"type": "text", "text": "Extract cheque fields..."}
        ]},
        {"role": "assistant", "content": "Payee_Name: ...\\nAmount_Numeric: ..."}
    ]
}

Usage:
    # v14 GT (83 cheques) — recommended
    python training/prepare_training_data.py \
        --gt_json cheque-poc/test-data/ground_truth_v14.json \
        --image_base cheque-poc/test-data \
        --output training/data/cheque_train_v14.jsonl \
        --augment_prompts

    # Legacy per-file GT
    python training/prepare_training_data.py \
        --gt_dir benchmark/ground_truth/ \
        --output training/data/cheque_train.jsonl
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ── Extraction prompts ────────────────────────────────────────────────────────
# V7 production prompt — must match vllm_extractor.py EXTRACTION_PROMPT exactly
EXTRACTION_PROMPT_V7 = (
    "You are a bank cheque OCR engine.\n\n"
    "Your task is to extract cheque fields EXACTLY as written in the image.\n"
    "Rules:\n"
    "- Replace ONLY the values after ':'\n"
    "- If a value is missing OR ambiguous, write NOT_FOUND\n"
    "- Do NOT add or remove lines\n"
    "- Do NOT repeat lines\n"
    "- Preserve numeric formatting and spacing exactly\n"
    "- Stop generation immediately after the last line\n\n"
    "- Signature_Present: YES if ANY pen stroke, scribble, or signature mark is visible; otherwise NO\n\n"
    "FIELD DEFINITIONS (FOR CLARITY ONLY — DO NOT OUTPUT):\n"
    "- Payee_Name: Name written ONLY on the line that starts with or immediately follows the keyword 'Pay'.\n"
    "- Drawer_Name: Account holder / cheque issuer. A printed personal or business name located near the Account_Number.\n"
    "- Bank_Name: Printed bank name in the cheque header/logo area.\n\n"
    "TEMPLATE (DO NOT MODIFY KEYS):\n"
    "Payee_Name: <value>\n"
    "Amount_Numeric: <value>\n"
    "Amount_Words_Arabic: <value>\n"
    "Amount_Words_English: <value>\n"
    "Cheque_Date: <value>\n"
    "Drawer_Name: <value>\n"
    "Bank_Name: <value>\n"
    "Signature_Present: <YES/NO>\n\n"
    "STRICT EXTRACTION RULES:\n"
    "- Do NOT summarize or paraphrase\n"
    "- Do NOT infer missing values\n"
    "- Preserve numeric spacing exactly\n"
    "- Extract Arabic text as it is and also provide its english translated version\n"
    "- Ignore logos, stamps, watermarks, and background patterns\n\n"
    "CRITICAL NAME EXTRACTION RULES (MANDATORY):\n"
    "- Payee_Name MUST be taken ONLY from the line that starts with or immediately follows the keyword 'Pay'\n"
    "- Text that does NOT appear on the 'Pay' line MUST NOT be used as Payee_Name\n"
    "- Drawer_Name MUST be a printed personal or business name near the Account_Number\n"
    "- NEVER use Bank_Name text for Payee_Name or Drawer_Name\n"
    "- If a name EXACTLY matches or is a substring of Bank_Name, it MUST NOT be used for Payee_Name or Drawer_Name\n"
    "- If the ONLY visible name on the cheque is the bank name, output NOT_FOUND for Payee_Name and Drawer_Name\n"
    "- If multiple names exist and the correct role is unclear, output NOT_FOUND\n\n"
    "END OF TASK. STOP AFTER Signature_Present"
)

# Concise variant — teaches the same extraction pattern with less prompt overhead
EXTRACTION_PROMPT_CONCISE = (
    "Extract all fields from this bank cheque image exactly as written.\n"
    "Output NOT_FOUND for missing or ambiguous values.\n\n"
    "Payee_Name: <value>\n"
    "Amount_Numeric: <value>\n"
    "Amount_Words_Arabic: <value>\n"
    "Amount_Words_English: <value>\n"
    "Cheque_Date: <value>\n"
    "Drawer_Name: <value>\n"
    "Bank_Name: <value>\n"
    "Signature_Present: <YES/NO>"
)

# Arabic-emphasis variant — reinforces Arabic text extraction
EXTRACTION_PROMPT_ARABIC = (
    "You are an expert Arabic/English bilingual cheque OCR system.\n\n"
    "Extract cheque fields exactly as written. Preserve Arabic text precisely — "
    "do not transliterate, summarize, or paraphrase.\n"
    "For missing values, write NOT_FOUND.\n\n"
    "Payee_Name: <value>\n"
    "Amount_Numeric: <value>\n"
    "Amount_Words_Arabic: <value>\n"
    "Amount_Words_English: <value>\n"
    "Cheque_Date: <value>\n"
    "Drawer_Name: <value>\n"
    "Bank_Name: <value>\n"
    "Signature_Present: <YES/NO>\n\n"
    "STOP after Signature_Present."
)

# Map of prompt variant names to prompts
PROMPT_VARIANTS = {
    "v7": EXTRACTION_PROMPT_V7,
    "concise": EXTRACTION_PROMPT_CONCISE,
    "arabic": EXTRACTION_PROMPT_ARABIC,
}


def gt_v14_to_expected_output(fields: dict) -> str:
    """
    Convert v14 GT fields dict to the expected model output format.
    v14 field names: payee_name, amount_figures, amount_words, date,
                     drawer_name, bank_name, signature, account_number, micr, iban
    """
    payee = fields.get("payee_name") or "NOT_FOUND"
    amount_fig = fields.get("amount_figures")
    if amount_fig is not None:
        # Format: keep as number, strip trailing .0 for integers
        if isinstance(amount_fig, float) and amount_fig == int(amount_fig):
            amount_fig = str(int(amount_fig))
        else:
            amount_fig = str(amount_fig)
    else:
        amount_fig = "NOT_FOUND"
    amount_ar = fields.get("amount_words") or "NOT_FOUND"
    amount_en = "NOT_FOUND"  # v14 GT has Arabic amount words only
    date = fields.get("date") or "NOT_FOUND"
    drawer = fields.get("drawer_name") or "NOT_FOUND"
    bank = fields.get("bank_name") or "NOT_FOUND"
    sig = "YES" if fields.get("signature") else "NO"

    lines = [
        f"Payee_Name: {payee}",
        f"Amount_Numeric: {amount_fig}",
        f"Amount_Words_Arabic: {amount_ar}",
        f"Amount_Words_English: {amount_en}",
        f"Cheque_Date: {date}",
        f"Drawer_Name: {drawer}",
        f"Bank_Name: {bank}",
        f"Signature_Present: {sig}",
    ]
    return "\n".join(lines)


def gt_legacy_to_expected_output(gt_fields: dict) -> str:
    """
    Convert legacy GT fields dict to expected model output format.
    Legacy field names: payee_name, amount_figures, amount_words_arabic,
                        date, payer_name, bank_name
    """
    payee = gt_fields.get("payee_name") or "NOT_FOUND"
    amount_fig = gt_fields.get("amount_figures") or "NOT_FOUND"
    amount_ar = gt_fields.get("amount_words_arabic") or "NOT_FOUND"
    amount_en = "NOT_FOUND"
    date = gt_fields.get("date") or "NOT_FOUND"
    drawer = gt_fields.get("payer_name") or "NOT_FOUND"
    bank = gt_fields.get("bank_name") or "NOT_FOUND"
    sig = "YES"

    lines = [
        f"Payee_Name: {payee}",
        f"Amount_Numeric: {amount_fig}",
        f"Amount_Words_Arabic: {amount_ar}",
        f"Amount_Words_English: {amount_en}",
        f"Cheque_Date: {date}",
        f"Drawer_Name: {drawer}",
        f"Bank_Name: {bank}",
        f"Signature_Present: {sig}",
    ]
    return "\n".join(lines)


def make_image_uri(image_path: str, server_base_dir: str = "", image_base_dir: str = "") -> str:
    """Build the file:// URI for an image.

    When server_base_dir is set, flattens the relative path to avoid collisions:
        Outward_images/image1.png → Outward_images_image1.png
        Inward_Cheques_images/image1.png → Inward_Cheques_images_image1.png
    """
    if server_base_dir:
        base = server_base_dir.rstrip("/")
        if image_base_dir:
            # Compute relative path from image_base_dir, then flatten
            rel = os.path.relpath(image_path, image_base_dir).replace("\\", "/")
            fname = rel.replace("/", "_")
        else:
            fname = os.path.basename(image_path)
        return f"file://{base}/images/{fname}"
    else:
        abs_path = os.path.abspath(image_path)
        return "file:///" + abs_path.replace("\\", "/")


def create_sample(
    image_path: str,
    expected_output: str,
    prompt: str,
    server_base_dir: str = "",
    image_base_dir: str = "",
    fmt: str = "unsloth",
) -> dict:
    """Create a single training sample."""
    image_uri = make_image_uri(image_path, server_base_dir, image_base_dir)

    if fmt == "swift":
        if server_base_dir:
            abs_path = f"{server_base_dir.rstrip('/')}/images/{os.path.basename(image_path)}"
        else:
            abs_path = os.path.abspath(image_path).replace("\\", "/")
        return {
            "messages": [
                {"role": "user", "content": f"<image>{abs_path}</image>\n{prompt}"},
                {"role": "assistant", "content": expected_output},
            ]
        }
    elif fmt == "base64":
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": expected_output},
                ]},
            ]
        }
    else:  # unsloth
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": expected_output},
                ]},
            ]
        }


# ── v14 GT loading ────────────────────────────────────────────────────────────

def load_gt_v14(gt_json_path: str, image_base_dir: str) -> List[dict]:
    """
    Load v14 GT format (single JSON with 'cheques' array).

    Returns list of dicts with keys: image_path (absolute), fields, category, id
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    entries = []
    for cheque in gt["cheques"]:
        # Resolve image path: image_path is relative like "Outward_images/image1.png"
        rel_path = cheque["image_path"]
        abs_path = os.path.normpath(os.path.join(image_base_dir, rel_path))

        entries.append({
            "image_path": abs_path,
            "fields": cheque["fields"],
            "category": cheque.get("category", "unknown"),
            "id": cheque.get("id", rel_path),
        })

    return entries


def load_gt_legacy(gt_dir: str) -> List[dict]:
    """Load legacy per-file GT format."""
    entries = []
    for fname in sorted(os.listdir(gt_dir)):
        if fname.startswith("_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(gt_dir, fname), "r", encoding="utf-8") as f:
            entry = json.load(f)
        entries.append({
            "image_path": entry["image_path"],
            "fields": entry["extraction"],
            "category": "unknown",
            "id": fname,
        })
    return entries


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_v14_dataset(
    gt_json_path: str,
    image_base_dir: str,
    output_path: str,
    fmt: str = "unsloth",
    server_base_dir: str = "",
    augment_prompts: bool = False,
) -> int:
    """
    Prepare training dataset from v14 GT.

    Args:
        gt_json_path: Path to ground_truth_v14.json
        image_base_dir: Base directory for resolving image paths
        output_path: Output JSONL file path
        fmt: Format - "unsloth", "swift", or "base64"
        server_base_dir: If set, rewrite image paths for server deployment
        augment_prompts: If True, generate 3 prompt variants per cheque (~3x samples)

    Returns:
        Number of samples written
    """
    entries = load_gt_v14(gt_json_path, image_base_dir)
    print(f"Loaded {len(entries)} v14 GT entries")

    # Determine which prompts to use
    if augment_prompts:
        prompts = [
            ("v7", EXTRACTION_PROMPT_V7),
            ("concise", EXTRACTION_PROMPT_CONCISE),
            ("arabic", EXTRACTION_PROMPT_ARABIC),
        ]
        print(f"Prompt augmentation: ON (3 variants per cheque)")
    else:
        prompts = [("v7", EXTRACTION_PROMPT_V7)]
        print(f"Prompt augmentation: OFF (V7 production prompt only)")

    samples = []
    skipped = 0

    for entry in entries:
        image_path = entry["image_path"]

        if not os.path.exists(image_path):
            print(f"  WARNING: Image not found, skipping: {image_path}")
            skipped += 1
            continue

        expected_output = gt_v14_to_expected_output(entry["fields"])

        for variant_name, prompt in prompts:
            sample = create_sample(
                image_path=image_path,
                expected_output=expected_output,
                prompt=prompt,
                server_base_dir=server_base_dir,
                image_base_dir=image_base_dir,
                fmt=fmt,
            )
            # Add metadata for debugging (not used by training)
            sample["_meta"] = {
                "cheque_id": entry["id"],
                "category": entry["category"],
                "prompt_variant": variant_name,
            }
            samples.append(sample)

    # Write JSONL
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Written {len(samples)} training samples to {output_path}")
    if skipped:
        print(f"  Skipped {skipped} entries (missing images)")

    # Summary by category
    cats = {}
    for entry in entries:
        cat = entry["category"]
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  Categories: {cats}")

    return len(samples)


def prepare_legacy_dataset(
    gt_dir: str,
    output_path: str,
    fmt: str = "unsloth",
    server_base_dir: str = "",
    include_augmented: bool = False,
    augmented_dir: Optional[str] = None,
) -> int:
    """Prepare training dataset from legacy per-file GT directory."""
    entries = load_gt_legacy(gt_dir)
    print(f"Loaded {len(entries)} legacy GT entries")

    samples = []

    for entry in entries:
        image_path = entry["image_path"]
        if not os.path.exists(image_path):
            print(f"  WARNING: Image not found, skipping: {image_path}")
            continue

        expected_output = gt_legacy_to_expected_output(entry["fields"])
        sample = create_sample(
            image_path=image_path,
            expected_output=expected_output,
            prompt=EXTRACTION_PROMPT_V7,
            server_base_dir=server_base_dir,
            fmt=fmt,
        )
        samples.append(sample)

    # Augmented samples (if available)
    if include_augmented and augmented_dir and os.path.isdir(augmented_dir):
        aug_manifest = os.path.join(augmented_dir, "manifest.json")
        if os.path.exists(aug_manifest):
            with open(aug_manifest, "r", encoding="utf-8") as f:
                aug_entries = json.load(f)
            for aug_entry in aug_entries:
                aug_image = aug_entry["image_path"]
                if os.path.exists(aug_image):
                    expected_output = gt_legacy_to_expected_output(aug_entry["extraction"])
                    sample = create_sample(
                        image_path=aug_image,
                        expected_output=expected_output,
                        prompt=EXTRACTION_PROMPT_V7,
                        server_base_dir=server_base_dir,
                        fmt=fmt,
                    )
                    samples.append(sample)
            print(f"Added {len(aug_entries)} augmented samples")

    # Write JSONL
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Written {len(samples)} training samples to {output_path}")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Qwen3-VL QLoRA fine-tuning"
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--gt_json",
        help="v14 GT JSON file (single file with cheques array)",
    )
    input_group.add_argument(
        "--gt_dir", "-g",
        help="Legacy GT JSON directory (one file per cheque)",
    )

    parser.add_argument(
        "--image_base",
        default="",
        help="Base directory for resolving v14 image paths "
             "(e.g., cheque-poc/test-data). Required with --gt_json.",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "training", "data", "cheque_train_v14.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["unsloth", "swift", "base64"],
        default="unsloth",
        help="Output format (default: unsloth)",
    )
    parser.add_argument(
        "--server_base_dir",
        default="",
        help="Rewrite image paths for server deployment "
             "(e.g., /workspace/cheque-training). Images go under /images/",
    )
    parser.add_argument(
        "--augment_prompts",
        action="store_true",
        help="Generate 3 prompt variants per cheque (~3x samples). "
             "Recommended for v14 to reach ~250 samples from 83 cheques.",
    )

    # Legacy-only options
    parser.add_argument(
        "--augmented_dir",
        default=os.path.join(ROOT_DIR, "training", "data", "augmented"),
        help="Augmented images directory (legacy mode only)",
    )
    parser.add_argument(
        "--include_augmented",
        action="store_true",
        help="Include augmented samples (legacy mode only)",
    )

    args = parser.parse_args()

    if args.gt_json:
        # v14 mode
        if not args.image_base:
            # Auto-detect: assume images are relative to the GT file's parent
            args.image_base = os.path.dirname(os.path.abspath(args.gt_json))
            print(f"Auto-detected image base: {args.image_base}")

        n = prepare_v14_dataset(
            gt_json_path=args.gt_json,
            image_base_dir=args.image_base,
            output_path=args.output,
            fmt=args.format,
            server_base_dir=args.server_base_dir,
            augment_prompts=args.augment_prompts,
        )
    else:
        # Legacy mode
        n = prepare_legacy_dataset(
            gt_dir=args.gt_dir,
            output_path=args.output,
            fmt=args.format,
            server_base_dir=args.server_base_dir,
            include_augmented=args.include_augmented,
            augmented_dir=args.augmented_dir,
        )

    print(f"\nDone. {n} samples ready for training.")


if __name__ == "__main__":
    main()
