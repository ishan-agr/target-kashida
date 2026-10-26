# -*- coding: utf-8 -*-
"""Build the sherif-v3 fine-tune JSONL from the 225-image leak-free overlay split.

Task shape (decided 2026-08-03): FULL cheque image -> PAYEE name only (no crop).
The prompt must point unambiguously at the payee ('Pay to' / 'ادفعوا لأمر') line so the
model does not return the (English) drawer name or the amount text.

Image reference stored as a bare filename; the training script resolves it against
--image_dir (portable to Kaggle /kaggle/input/...). Unsloth conversation schema.

Usage:
  python training/prepare_sherif_overlay_data.py [--variants 2] \
      --manifest training/data/overlay_train225_gt.json \
      --out training/data/sherif_overlay_train.jsonl
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full-image payee-isolation prompts. [0]=EN-pointing, [1]=AR-pointing, [2]=minimal.
PROMPTS = [
    ("This is a bank cheque image. Read ONLY the payee name — the handwritten/printed "
     "name on the 'Pay to the order of' line (NOT the account holder/drawer at the "
     "bottom, NOT the amount). Return just that name, nothing else."),
    ("هذه صورة شيك بنكي. اقرأ اسم المستفيد فقط، وهو الاسم المكتوب على سطر «ادفعوا لأمر» "
     "(وليس اسم صاحب الحساب في الأسفل، وليس المبلغ). أعد الاسم فقط لا غير."),
    "What is the payee name written on this cheque? Return only the name.",
]


def make_sample(fname, payee, prompt, meta):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": fname},
                {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": payee}]},
        ],
        "_source": "overlay_payee",
        "_meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "training", "data", "overlay_train225_gt.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "data", "sherif_overlay_train.jsonl"))
    ap.add_argument("--variants", type=int, default=2, help="prompt variants per image (<=3)")
    args = ap.parse_args()
    variants = max(1, min(args.variants, len(PROMPTS)))

    man = json.load(open(args.manifest, encoding="utf-8"))
    recs = man["cheques"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, c in enumerate(recs):
            fname = c["id"]
            payee = c["fields"].get("payee_name", "").strip()
            if not payee:
                continue
            for v in range(variants):
                prompt = PROMPTS[(i + v) % len(PROMPTS)]  # rotate for prompt-invariance
                meta = {"cheque_id": fname, "category": c.get("category"),
                        "template": c.get("source_cheque"), "variant": v}
                f.write(json.dumps(make_sample(fname, payee, prompt, meta),
                                   ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {n} samples ({len(recs)} images x {variants} variants) -> {args.out}")
    # quick integrity: unique payees, sample
    payees = {c['fields']['payee_name'] for c in recs}
    print(f"unique payees: {len(payees)} | first image: {recs[0]['id']}")


if __name__ == "__main__":
    main()
