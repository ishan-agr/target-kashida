"""Prepare HATFormer FT data: copy the 30 crops_gt, apply the validated UPSCALE
strategy (min-height 200px cubic — the only non-destructive preprocessing for
HATFormer, ties raw at 17.2% fuzzy), and write a labels CSV from v15-m GT.

Output:
  training/data/hatformer_crops_upscaled/<crop_name>.png   (30 upscaled crops)
  training/data/hatformer_crops_upscaled/labels.csv         (img_path,text,is_arabic,category)

Run:
  python training/prepare_hatformer_crops.py
"""
import os, io, csv, json, re
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
CROPS_GT = ROOT / "benchmark" / "sherif_v3_arabic_outward" / "crops_gt"
GT_JSON = ROOT / "cheque-poc" / "test-data" / "ground_truth_v15-m.json"
OUT_DIR = ROOT / "training" / "data" / "hatformer_crops_upscaled"
MIN_HEIGHT = 200
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def upscale_crop(img, min_height=MIN_HEIGHT):
    """Stage-1 upscale (payee_cleanup.upscale_crop): min-height cubic, AR-preserving."""
    h, w = img.shape[:2]
    if h < min_height:
        s = min_height / h
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return img


def norm_crop_name(image_path):
    return image_path.replace("/", "_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = json.load(io.open(GT_JSON, encoding="utf-8"))
    # crop_name -> (payee, category)
    label_map = {}
    for c in gt["cheques"]:
        label_map[norm_crop_name(c["image_path"])] = (
            c["fields"].get("payee_name", "") or "", c.get("category", ""))

    rows = []
    crops = sorted(CROPS_GT.glob("*.png"))
    print(f"crops_gt: {len(crops)} | out: {OUT_DIR}")
    for cp in crops:
        name = cp.name
        payee, cat = label_map.get(name, ("", ""))
        if not payee:
            print(f"  WARN no GT for {name}, skipping")
            continue
        img = cv2.imread(str(cp))
        if img is None:
            print(f"  WARN unreadable {name}, skipping")
            continue
        h0 = img.shape[0]
        up = upscale_crop(img)
        out_path = OUT_DIR / name
        cv2.imwrite(str(out_path), up)
        is_ar = bool(ARABIC_RE.search(payee))
        rows.append({"img_path": str(out_path).replace("\\", "/"),
                     "text": payee, "is_arabic": int(is_ar), "category": cat})
        print(f"  {name}: {h0}px -> {up.shape[0]}px | {'AR' if is_ar else 'EN'} | {payee[:40]}")

    csv_path = OUT_DIR / "labels.csv"
    with io.open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["img_path", "text", "is_arabic", "category"])
        w.writeheader()
        w.writerows(rows)
    n_ar = sum(r["is_arabic"] for r in rows)
    print(f"\nwrote {len(rows)} crops ({n_ar} Arabic, {len(rows)-n_ar} English) -> {csv_path}")


if __name__ == "__main__":
    main()
