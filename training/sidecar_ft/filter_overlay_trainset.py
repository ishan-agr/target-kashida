# -*- coding: utf-8 -*-
"""Filter the 3,690-image overlaid set down to a leak-free ~225-image train split.

Rule (decided 2026-08-03):
  * 45 unique Arabic words x 5 backgrounds each = 225 images.
  * A background is *eligible* only if it is NOT the source image of an Arabic-payee
    validation cheque (ground_truth_v16.json) -> prevents train/val background leak.
    24 of 82 overlaid templates are Arabic-val backgrounds; 58 remain eligible.
  * Each word's 5 backgrounds are spread across the sorted eligible list (category
    diversity: docx-inward / docx-outward / inward-scan / outward-scan) and offset by
    word index, deterministically (no RNG -> reproducible).

Outputs:
  * copies the selected PNGs into DST_DIR (flat, original filenames)
  * writes a filtered GT manifest (same schema as the overlaid GT `cheques` records)

Usage:
  python training/filter_overlay_trainset.py [--per_word 5] [--dry_run]
"""
import argparse
import json
import os
import shutil
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVER_GT = r"<HOME>\Downloads\ground_truth_overlaid.json"
V16_GT = os.path.join(ROOT, "cheque-poc", "test-data", "ground_truth_v16.json")
SRC_DIR = r"<LOCAL_PATH>/output_cheques"
DST_DIR = r"<LOCAL_PATH>/output_cheques_train200"
MANIFEST = os.path.join(ROOT, "training", "data", "overlay_train225_gt.json")

def is_arabic(s):
    return any('؀' <= c <= 'ۿ' or 'ݐ' <= c <= 'ݿ'
               or 'ﭐ' <= c <= '﷿' or 'ﹰ' <= c <= '﻿'
               for c in (s or ""))

def template_of(img_id):          # img_001_docx_inward_01.png -> docx_inward_01.png
    p = img_id.split("_", 2)
    return p[2] if len(p) == 3 else img_id

def word_idx(img_id):             # img_001_... -> "001"
    return img_id.split("_")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_word", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    over = json.load(open(OVER_GT, encoding="utf-8"))
    recs = {c["id"]: c for c in over["cheques"]}
    templates = sorted({template_of(i) for i in recs})
    words = sorted({word_idx(i) for i in recs}, key=int)

    # --- leak set: templates whose source == an Arabic val cheque ---
    # Match on the FULL relative path (overlaid source_cheque and v16 image_path share
    # the same format), NOT basename — scan names collide across in/outward (image1.png).
    norm = lambda p: str(p).replace("\\", "/").lower()
    v16 = json.load(open(V16_GT, encoding="utf-8"))
    ar_val_paths = {norm(c.get("image_path"))
                    for c in v16["cheques"] if is_arabic(c["fields"].get("payee_name"))}
    tsrc = {template_of(c["id"]): c.get("source_cheque") for c in over["cheques"]}
    leak = {t for t in templates if norm(tsrc[t]) in ar_val_paths}
    eligible = [t for t in templates if t not in leak]
    print(f"templates total={len(templates)} leak(arabic-val)={len(leak)} "
          f"eligible={len(eligible)} words={len(words)}")
    if args.per_word > len(eligible):
        raise SystemExit("per_word exceeds eligible template count")

    # --- deterministic, category-spread, load-balanced selection ---
    N = len(eligible)
    stride = max(1, N // args.per_word)
    selected, usage = [], collections.Counter()
    for wi, w in enumerate(words):
        picks, seen, j = [], set(), 0
        while len(picks) < args.per_word and j <= N:
            t = eligible[(wi + j * stride) % N]
            if t not in seen:
                seen.add(t); picks.append(t)
            j += 1
        # tie-break toward least-used templates for balance
        for t in picks:
            usage[t] += 1
        img_id = None
        for t in picks:
            img_id = f"img_{w}_{t}"
            if img_id in recs:
                selected.append(recs[img_id])
            else:
                print(f"  WARN missing record for {img_id}")

    print(f"selected={len(selected)} (expected {len(words)*args.per_word})")
    cat = collections.Counter(c["category"] for c in selected)
    kind = collections.Counter(("scan" if "images" in template_of(c["id"]) else "docx")
                               for c in selected)
    print("category spread:", dict(cat), "| kind spread:", dict(kind))
    print("per-template usage: min={} max={} distinct={}".format(
        min(usage.values()), max(usage.values()), len(usage)))
    # leak assertion
    used_templates = {template_of(c["id"]) for c in selected}
    assert not (used_templates & leak), "LEAK: a validation background was selected"
    print("leak check: PASS (0 validation backgrounds in train split)")

    if args.dry_run:
        print("dry run — nothing written")
        return

    os.makedirs(DST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    copied = 0
    for c in selected:
        src = os.path.join(SRC_DIR, c["id"])
        dst = os.path.join(DST_DIR, c["id"])
        if os.path.exists(src):
            shutil.copy2(src, dst); copied += 1
        else:
            print(f"  WARN source image missing: {src}")
    manifest = {"version": "overlay-train225", "n": len(selected),
                "per_word": args.per_word, "num_words": len(words),
                "eligible_templates": len(eligible), "leak_excluded": sorted(leak),
                "src_dir": SRC_DIR, "dst_dir": DST_DIR, "cheques": selected}
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"copied {copied} images -> {DST_DIR}")
    print(f"wrote manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
