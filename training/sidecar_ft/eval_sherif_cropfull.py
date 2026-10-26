#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apples-to-apples: sherif v3 standalone, FULL image vs PAYEE-ROI CROP, on the SAME 31
Arabic val cheques with the SAME fuzzy scorer and the SAME prompt (only the image varies).

Closes the metric-mismatch gap: the older crop numbers (41-48%) were manual crops vs v14 GT;
this measures full-vs-ROI-crop within one run on v16 GT. Vendors has_arabic + fuzzy_name_match
VERBATIM from src/v7a/text_utils.py so the % equals the pipeline's payee metric.

  python eval_sherif_cropfull.py --model sherif1313/Arabic-English-handwritten-OCR-v3 \
      --gt /kaggle/input/<DS>/val/val_arabic_gt.json --img_base /kaggle/input/<DS>/val \
      --out /kaggle/working/eval_cropfull.json
"""
import argparse, json, os, re

ROI_PAYEE = (0.0, 0.28, 1.0, 0.63)   # same ROI the pipeline/prep uses for the payee band
PROMPT = ("Read the payee name — the handwritten name on the cheque's pay-to line. "
          "Return only that name, nothing else.")   # shared across both modes

# ---- vendored VERBATIM from src/v7a/text_utils.py (frozen scorer) ----
def has_arabic(t):
    return bool(t) and bool(re.search(r"[؀-ۿݐ-ݿࢠ-ࣿ]", t))

def fuzzy_name_match(extracted, ground_truth):
    if not extracted or not ground_truth:
        return extracted is None and ground_truth is None
    ext = extracted.lower().strip(); gt = ground_truth.lower().strip()
    if ext == gt or ext in gt or gt in ext:
        return True
    for gt_part in re.split(r"\s*/\s*", gt):
        gt_part = gt_part.strip()
        if not gt_part:
            continue
        if gt_part in ext or ext in gt_part:
            return True
        for ext_part in re.split(r"\s*/\s*", ext):
            ext_part = ext_part.strip()
            if ext_part and (gt_part in ext_part or ext_part in gt_part):
                return True
        gt_words = set(re.findall(r"\w+", gt_part)); ext_words = set(re.findall(r"\w+", ext))
        if gt_words and len(gt_words & ext_words) / len(gt_words) >= 0.5:
            return True
    return False

_DIAC = "".join(chr(c) for c in list(range(0x064B, 0x0653)) + [0x0670, 0x0640])
def norm(s):
    s = "".join(ch for ch in (s or "") if ch not in _DIAC)
    for a in "أإآٱ": s = s.replace(a, "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", s).strip().lower()

def lev(a, b):
    if a == b: return 0
    if not a or not b: return len(a or b)
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]

def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    return lev(ref, hyp) / max(len(ref), 1)


def payee_crop(img):
    """PIL full cheque -> payee-ROI crop (mirrors prepare_sherif_ft_data.py upscale+ROI+min-h)."""
    w, h = img.size
    if w < 1400:
        s = 1400 / w; img = img.resize((int(w * s), int(h * s))); w, h = img.size
    box = (int(ROI_PAYEE[0] * w), int(ROI_PAYEE[1] * h), int(ROI_PAYEE[2] * w), int(ROI_PAYEE[3] * h))
    crop = img.crop(box)
    cw, ch = crop.size
    if ch < 200:
        s = 200 / ch; crop = crop.resize((int(cw * s), int(ch * s)))
    return crop


def load_model(model_id):
    from unsloth import FastVisionModel
    model, tok = FastVisionModel.from_pretrained(model_id, load_in_4bit=True)
    ie, oe = model.get_input_embeddings(), model.get_output_embeddings()
    if ie is not None and oe is not None and ie.weight.data_ptr() != oe.weight.data_ptr():
        oe.weight = ie.weight; print("  [fix] re-tied lm_head")
    FastVisionModel.for_inference(model)
    return model, tok


def infer(model, tok, pil_img):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True)
    inp = tok(pil_img, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=48, do_sample=False)
    return tok.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--img_base", required=True)
    ap.add_argument("--out", default="eval_cropfull.json")
    args = ap.parse_args()

    val = [c for c in json.load(open(args.gt, encoding="utf-8"))["cheques"]
           if has_arabic(c["fields"].get("payee_name"))]
    print(f"sherif crop-vs-full | {len(val)} Arabic cheques | model={args.model}")
    model, tok = load_model(args.model)

    agg = {"full": {"fuzzy": 0, "exact": 0, "cer": []}, "crop": {"fuzzy": 0, "exact": 0, "cer": []}}
    rows = []
    for i, c in enumerate(val):
        gt = c["fields"]["payee_name"]
        img = Image.open(os.path.join(args.img_base, c["image_path"])).convert("RGB")
        variants = {"full": img, "crop": payee_crop(img)}
        row = {"id": c["id"], "gt": gt}
        for mode, im in variants.items():
            pred = infer(model, tok, im)
            fz = bool(fuzzy_name_match(pred, gt)); ex = int(norm(pred) == norm(gt)); cv = cer(gt, pred)
            agg[mode]["fuzzy"] += int(fz); agg[mode]["exact"] += ex; agg[mode]["cer"].append(cv)
            row[mode] = {"pred": pred, "fuzzy": fz, "exact": ex, "cer": round(cv, 3)}
        rows.append(row)
        print(f"  [{i+1:>2}/{len(val)}] full:fz={int(row['full']['fuzzy'])} cer={row['full']['cer']:.2f} | "
              f"crop:fz={int(row['crop']['fuzzy'])} cer={row['crop']['cer']:.2f}")

    n = len(val)
    print("=" * 56)
    for mode in ("full", "crop"):
        d = agg[mode]
        print(f"{mode.upper():4}  fuzzy={d['fuzzy']}/{n}={100*d['fuzzy']/n:5.1f}%  "
              f"exact={d['exact']}/{n}={100*d['exact']/n:5.1f}%  meanCER={sum(d['cer'])/n:.3f}")
    print(f"DELTA fuzzy (crop-full): {100*(agg['crop']['fuzzy']-agg['full']['fuzzy'])/n:+.1f} pp")

    json.dump({"model": args.model, "n": n, "prompt": PROMPT,
               "full": {"fuzzy": agg["full"]["fuzzy"], "exact": agg["full"]["exact"],
                        "mean_cer": round(sum(agg["full"]["cer"])/n, 3)},
               "crop": {"fuzzy": agg["crop"]["fuzzy"], "exact": agg["crop"]["exact"],
                        "mean_cer": round(sum(agg["crop"]["cer"])/n, 3)},
               "rows": rows}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
