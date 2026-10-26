# -*- coding: utf-8 -*-
"""Frontier baseline: Claude (vision) on the 83-cheque payee task, scored with the SAME
pipeline matchers so it sits directly next to Qwen3.5 (AR-fuzzy 41.9% / EN 96.2% / all 75.9%).

SECURITY: key is read ONLY from os.environ['ANTHROPIC_API_KEY']; never an arg, never logged.
Runs LOCALLY (key stays on this machine); uses the real src.v7a matchers for apples-to-apples.
External-API use here is a deliberate, user-authorized override of the 'no external API' rule,
scoped to the POC test set (cheque-poc/test-data), not live customer data.

  python benchmark/claude_payee_benchmark.py [--model claude-opus-4-8] [--limit N]
"""
import argparse, base64, json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console defaults to cp1252 -> Arabic crashes
except Exception:
    pass
from src.v7a.text_utils import fuzzy_name_match, has_arabic, bank_name_match  # noqa: E402
from src.v7a.deterministic import normalize_amount_cmp                        # noqa: E402
from src.v7a.amount_words import words_to_number                              # noqa: E402

GT_PATH = os.path.join(ROOT, "cheque-poc", "test-data", "ground_truth_v16.json")
IMG_BASE = os.path.join(ROOT, "cheque-poc", "test-data")

PROMPT = (
    "You are extracting fields from a UAE bank cheque image. Return ONLY a JSON object with keys:\n"
    '  "payee_name": name on the "Pay to the order of" line (may be Arabic handwriting),\n'
    '  "drawer_name": account holder (usually printed at the bottom),\n'
    '  "date": ISO YYYY-MM-DD,\n'
    '  "amount_figures": numeric only, no currency,\n'
    '  "amount_words": the written-out amount,\n'
    '  "bank_name": issuing bank.\n'
    "Keep the exact script shown (Arabic stays Arabic). Use null if a field is absent. "
    "Output JSON only, no commentary, no markdown fences."
)

_DIAC = "".join(chr(c) for c in list(range(0x064B, 0x0653)) + [0x0670, 0x0640])
def norm(s):
    s = "".join(ch for ch in (s or "") if ch not in _DIAC)
    for a in "أإآٱ": s = s.replace(a, "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.I | re.S)
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except Exception:
            pass
    return {}


def call_claude(client, model, img_bytes, media_type, retries=4):
    b64 = base64.standard_b64encode(img_bytes).decode()
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=512,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": media_type, "data": b64}},
                    {"type": "text", "text": PROMPT}]}])
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} in {wait}s ({type(e).__name__}: {str(e)[:80]})")
            time.sleep(wait)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "reports", "claude_payee_benchmark.json"))
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not in environment — aborting (never pass it as an arg).")
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    gt = json.load(open(GT_PATH, encoding="utf-8"))["cheques"]
    if args.limit:
        gt = gt[:args.limit]
    print(f"Claude payee benchmark | model={args.model} | n={len(gt)} | key_len={len(key)}")

    B = {"ar": {"n": 0, "exact": 0, "fuzzy": 0}, "en": {"n": 0, "exact": 0, "fuzzy": 0}}
    fld = {f: {"ok": 0, "tot": 0} for f in ("drawer", "date", "amount", "amount_words", "bank")}
    rows = []
    for i, c in enumerate(gt):
        f = c["fields"]
        path = os.path.join(IMG_BASE, c["image_path"])
        mt = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        raw = call_claude(client, args.model, open(path, "rb").read(), mt)
        d = parse_json(raw)
        pred_payee = d.get("payee_name")
        gtp = f.get("payee_name")
        b = "ar" if has_arabic(gtp) else "en"
        B[b]["n"] += 1
        ex = int(norm(pred_payee) == norm(gtp))
        fz = int(bool(fuzzy_name_match(pred_payee, gtp)))
        B[b]["exact"] += ex; B[b]["fuzzy"] += fz
        # bonus fields
        if f.get("drawer_name"):
            fld["drawer"]["tot"] += 1
            fld["drawer"]["ok"] += int(bool(fuzzy_name_match(d.get("drawer_name"), f["drawer_name"])))
        if f.get("date"):
            fld["date"]["tot"] += 1
            fld["date"]["ok"] += int(str(d.get("date")) == str(f["date"]))
        if f.get("amount_figures") not in (None, "", "null"):
            fld["amount"]["tot"] += 1
            ea = normalize_amount_cmp(d.get("amount_figures")); ga = float(f["amount_figures"])
            fld["amount"]["ok"] += int(ea is not None and abs(ea - ga) < 0.01)
        if f.get("amount_words"):
            fld["amount_words"]["tot"] += 1
            wn, gn = words_to_number(d.get("amount_words")), words_to_number(f["amount_words"])
            fld["amount_words"]["ok"] += int(wn is not None and gn is not None and abs(wn - gn) < 0.01)
        if f.get("bank_name"):
            fld["bank"]["tot"] += 1
            fld["bank"]["ok"] += int(bool(bank_name_match(d.get("bank_name"), f["bank_name"])))
        rows.append({"id": c["id"], "is_arabic": b == "ar", "gt_payee": gtp,
                     "pred_payee": pred_payee, "exact": ex, "fuzzy": fz, "raw_fields": d})
        print(f"  [{i+1:>2}/{len(gt)}] {b} exact={ex} fuzzy={fz} | pred={str(pred_payee)[:40]}")

    print("\n" + "=" * 60)
    print(f"CLAUDE PAYEE | {args.model} | n={len(gt)}")
    for b in ("ar", "en"):
        d = B[b]; n = max(d["n"], 1)
        print(f"{b.upper()}: n={d['n']:>2}  exact={d['exact']:>2}/{d['n']:<2}={100*d['exact']/n:5.1f}%  "
              f"fuzzy={d['fuzzy']:>2}/{d['n']:<2}={100*d['fuzzy']/n:5.1f}%")
    tn = B["ar"]["n"] + B["en"]["n"]; te = B["ar"]["exact"] + B["en"]["exact"]; tf = B["ar"]["fuzzy"] + B["en"]["fuzzy"]
    print(f"ALL: n={tn}  exact={te}/{tn}={100*te/tn:.1f}%  fuzzy={tf}/{tn}={100*tf/tn:.1f}%")
    print("-- bonus fields --")
    for k, v in fld.items():
        if v["tot"]:
            print(f"  {k:12s} {v['ok']:>2}/{v['tot']:<2} = {100*v['ok']/v['tot']:5.1f}%")

    json.dump({"model": args.model, "n": len(gt),
               "payee": {b: B[b] for b in ("ar", "en")},
               "fields": fld, "rows": rows},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
