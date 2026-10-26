#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a sherif-v3 checkpoint on the 31 Arabic-payee validation cheques.

Full cheque image -> payee name. Reports CER / WER / normalized exact-match.
Run once on the BASE model and once on the fine-tuned merged dir to get the delta:

  python training/eval_sherif_overlay.py --model sherif1313/Arabic-English-handwritten-OCR-v3 --tag base
  python training/eval_sherif_overlay.py --model ./sherif-overlay-payee/merged_bf16 --tag ft

Prints ASCII-safe metrics; writes per-cheque Arabic predictions to a UTF-8 JSON report.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V16 = os.path.join(ROOT, "cheque-poc", "test-data", "ground_truth_v16.json")
IMG_BASE = os.path.join(ROOT, "cheque-poc", "test-data")

PROMPT = ("This is a bank cheque image. Read ONLY the payee name — the handwritten/printed "
          "name on the 'Pay to the order of' line (NOT the account holder/drawer at the "
          "bottom, NOT the amount). Return just that name, nothing else.")

_DIAC = "".join(chr(c) for c in list(range(0x064B, 0x0653)) + [0x0670, 0x0640])


def is_arabic(s):
    return any('؀' <= c <= 'ۿ' or 'ﭐ' <= c <= '﷿' for c in (s or ""))


def norm_ar(s):
    """Normalise Arabic for scoring: drop diacritics/tatweel, unify alef/ya/ta-marbuta,
    strip punctuation, collapse whitespace, lowercase latin."""
    import re
    if not s:
        return ""
    s = "".join(ch for ch in str(s) if ch not in _DIAC)
    for a in "أإآٱ":
        s = s.replace(a, "ا")
    s = s.replace("ة", "ه").replace("ى", "ي").replace("ﻻ", "لا").lower()
    s = re.sub(r"[^0-9a-z؀-ۿ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lev(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref, hyp = norm_ar(ref), norm_ar(hyp)
    return lev(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    r, h = norm_ar(ref).split(), norm_ar(hyp).split()
    return lev(r, h) / max(len(r), 1)


def ensure_tied_head(model):
    """Transformers 5.x + Unsloth can leave lm_head randomly initialised for tied-embedding
    checkpoints (sherif ships no lm_head.weight -> Qwen2.5-VL ties it to input embeddings).
    Symptom: model emits random-vocabulary noise. Re-tie if the head is not shared."""
    try:
        ie, oe = model.get_input_embeddings(), model.get_output_embeddings()
        if ie is not None and oe is not None and ie.weight.data_ptr() != oe.weight.data_ptr():
            oe.weight = ie.weight
            print("  [fix] re-tied lm_head to input embeddings (was untied/random)")
    except Exception as e:
        print("  [fix] tie check skipped:", e)


def load_model(model_id):
    from unsloth import FastVisionModel
    model, tok = FastVisionModel.from_pretrained(model_id, load_in_4bit=True)
    ensure_tied_head(model)
    FastVisionModel.for_inference(model)
    return model, tok


def infer(model, tok, image_path):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = tok(img, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=48, use_cache=True,
                         do_sample=False, temperature=0.0)
    dec = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)[0]
    return dec.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or merged dir")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--gt", default=V16, help="validation GT json (default: repo v16)")
    ap.add_argument("--img_base", default=IMG_BASE, help="dir images are relative to")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    v16 = json.load(open(args.gt, encoding="utf-8"))
    val = [c for c in v16["cheques"] if is_arabic(c["fields"].get("payee_name"))]
    print(f"[{args.tag}] eval on {len(val)} Arabic-payee cheques | model={args.model}")

    model, tok = load_model(args.model)
    rows, cers, wers, exact, contains = [], [], [], 0, 0
    for i, c in enumerate(val):
        gt = c["fields"]["payee_name"]
        path = os.path.join(args.img_base, c["image_path"])
        try:
            pred = infer(model, tok, path)
        except Exception as e:
            pred = f"<ERR:{e}>"
        cv, wv = cer(gt, pred), wer(gt, pred)
        em = int(norm_ar(gt) == norm_ar(pred))
        ct = int(bool(norm_ar(gt)) and norm_ar(gt) in norm_ar(pred))  # payee present anywhere
        cers.append(cv); wers.append(wv); exact += em; contains += ct
        rows.append({"id": c["id"], "gt": gt, "pred": pred, "contains": ct,
                     "cer": round(cv, 3), "wer": round(wv, 3), "exact": em,
                     "pred_len": len(pred)})
        print(f"  [{i+1:>2}/{len(val)}] cer={cv:5.2f} em={em} sub={ct} | pred[:70]={pred[:70]!r}")

    mcer = sum(cers) / len(cers)
    mwer = sum(wers) / len(wers)
    print("=" * 50)
    print(f"[{args.tag}]  mean CER={mcer:.3f}  mean WER={mwer:.3f}  "
          f"exact={exact}/{len(val)}={100*exact/len(val):.1f}%  "
          f"payee-substring-present={contains}/{len(val)}={100*contains/len(val):.1f}%")
    print("  (sub=1 means the payee text appears somewhere in the output -> read OK but not "
          "isolated; sub=0 with high CER -> genuine misread)")

    out = args.out or os.path.join(os.getcwd(), f"eval_overlay_{args.tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({"tag": args.tag, "model": args.model, "n": len(val),
               "mean_cer": round(mcer, 3), "mean_wer": round(mwer, 3),
               "exact": exact, "contains": contains, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
