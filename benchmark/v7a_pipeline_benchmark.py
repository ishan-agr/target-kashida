"""End-to-end v7a pipeline benchmark on all 83 cheques vs v16 GT.

Runs the REAL ChequePipelineV7A.process() per cheque (ingest -> passes -> fuse -> validate ->
detect -> confidence -> exceptions) against a live self-hosted vLLM endpoint, then scores fields
with the same matchers used elsewhere. Produces the standard scorecard + MICR emitted precision +
exception/routing stats + latency, and saves a JSON report.

Usage:
  python benchmark/v7a_pipeline_benchmark.py --endpoint https://HOST --key KEY \
         --model cyankiwi/Qwen3.5-9B-AWQ-4bit [--limit N] [--enable-micr-iban]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.v7a.amount_words import (_is_arabic as aw_is_arabic, normalize_words,  # noqa: E402
                                   parse as parse_amount_words, parse_aligned)
from src.v7a.config import get_config                                   # noqa: E402
from src.v7a.deterministic import normalize_amount_cmp, normalize_micr_cmp  # noqa: E402
from src.v7a.model import ProcessRequest                               # noqa: E402
from src.v7a.pipeline import ChequePipelineV7A                         # noqa: E402
from src.v7a.text_utils import bank_name_match, fuzzy_name_match, has_arabic  # noqa: E402

GT_PATH = os.path.join(ROOT, "cheque-poc", "test-data", "ground_truth_v16.json")
IMG_BASE = os.path.join(ROOT, "cheque-poc", "test-data")
# amount_words numeric-value accuracy is folded into the mean (8-field);
# amount_words exact-string accuracy is reported separately (auxiliary, not in the mean).
FIELDS = ["amount", "amount_words", "date", "bank", "micr", "payee", "drawer", "iban"]


def micr_ok(emitted, gt_raw):
    e, g = normalize_micr_cmp(emitted), normalize_micr_cmp(gt_raw)
    return bool(e and g and (g in e or e in g))


def score(rec, gt):
    """Return per-field pass/None and the MICR emitted/correct booleans."""
    s = {}
    # amount
    ea, ga = normalize_amount_cmp(rec.amount_figures.value), gt.get("amount_figures")
    ga = float(ga) if ga not in (None, "", "null") else None
    s["amount"] = (ea == ga) if (ea is not None and ga is not None) else None
    # amount_words (numeric value match; words-vs-words isolates the words field from figures)
    gp = parse_amount_words(gt.get("amount_words"))
    if gp["value"] is None or gp["ambiguous"]:
        s["amount_words"] = None                 # GT words missing / unparseable / OCR-corrupt -> skip
    else:
        mp = parse_aligned(rec.amount_words.value, aw_is_arabic(gt.get("amount_words")))
        s["amount_words"] = bool(mp["value"] is not None and abs(mp["value"] - gp["value"]) < 0.01)
    # date (ISO already)
    s["date"] = (rec.date.value == gt.get("date")) if (rec.date.value and gt.get("date")) else None
    # bank
    s["bank"] = bool(bank_name_match(rec.bank_name.value, gt.get("bank_name"))) if gt.get("bank_name") else None
    # micr (raw read accuracy over emitted; MANUAL -> None=not emitted)
    if gt.get("micr"):
        s["micr"] = micr_ok(rec.micr_line.value, gt.get("micr")) if rec.micr_line.value else None
    else:
        s["micr"] = None
    # payee / drawer
    s["payee"] = bool(fuzzy_name_match(rec.payee_name.value, gt.get("payee_name"))) if gt.get("payee_name") else None
    s["drawer"] = bool(fuzzy_name_match(rec.payer_name.value, gt.get("drawer_name"))) if gt.get("drawer_name") else None
    # iban
    gi = gt.get("iban")
    if gi:
        s["iban"] = (rec.iban.value or "").replace(" ", "").upper() == gi.replace(" ", "").upper()
    else:
        s["iban"] = None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="self-hosted vLLM base URL (no /v1)")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="cyankiwi/Qwen3.5-9B-AWQ-4bit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--enable-micr-iban", action="store_true")
    ap.add_argument("--enable-anchor-layout", action="store_true",
                    help="V8.3 anchor-relative crops for payee/amount/date (needs PaddleOCR)")
    ap.add_argument("--logprobs", action="store_true", help="request token logprobs (Qwen confidence)")
    args = ap.parse_args()

    cfg = get_config(reload=True)
    cfg.vlm.endpoint = args.endpoint
    cfg.vlm.model = args.model
    cfg.vlm.api_key = args.key
    cfg.runtime.enable_micr_iban = args.enable_micr_iban
    cfg.runtime.enable_anchor_layout = args.enable_anchor_layout
    cfg.vlm.request_logprobs = args.logprobs

    gt = json.load(open(GT_PATH, encoding="utf-8"))["cheques"]
    if args.limit:
        gt = gt[:args.limit]
    pipe = ChequePipelineV7A(cfg)
    print(f"v7a benchmark | model={args.model} | endpoint={args.endpoint} | n={len(gt)} | "
          f"micr_iban={'on' if args.enable_micr_iban else 'off'} | grounding={'on' if cfg.runtime.enable_grounding else 'off'}")
    if not pipe.vlm.health():
        print("WARNING: endpoint /v1/models health check failed — continuing anyway")

    P = {f: 0 for f in FIELDS}
    T = {f: 0 for f in FIELDS}
    micr_emit = micr_emit_ok = exc_total = routed = 0
    aw_exact_ok = aw_exact_tot = 0
    latencies = []
    rows = []
    for i, c in enumerate(gt):
        rel = c["image_path"]
        path = os.path.join(IMG_BASE, rel)
        req = ProcessRequest(iccs_txn_id=f"BENCH-{i:03d}", batch_id="BENCH",
                             image_paths=[path], channel="benchmark")
        t0 = time.time()
        rec = pipe.process(req)
        latencies.append(time.time() - t0)
        s = score(rec, c["fields"])
        for f in FIELDS:
            if s[f] is not None:
                T[f] += 1
                P[f] += int(s[f])
        # micr emitted precision
        if c["fields"].get("micr"):
            if rec.micr_line.value:
                micr_emit += 1
                micr_emit_ok += int(micr_ok(rec.micr_line.value, c["fields"]["micr"]))
        exc_total += len(rec.exceptions)
        routed += int(rec.routing_priority != "NORMAL")
        # amount_words exact-string accuracy (auxiliary; not folded into the mean)
        gw = c["fields"].get("amount_words")
        mw = rec.amount_words.value
        aw_exact = None
        if gw and str(gw).strip():
            aw_exact = (normalize_words(mw) == normalize_words(gw)) if mw else False
            aw_exact_tot += 1
            aw_exact_ok += int(bool(aw_exact))
        gt_payee = c["fields"].get("payee_name")
        rows.append({"image": rel, "status": str(rec.status), "priority": rec.routing_priority,
                     "n_exc": len(rec.exceptions), "scores": s,
                     "payee": rec.payee_name.value, "gt_payee": gt_payee,
                     "payee_correct": s["payee"], "is_arabic": has_arabic(gt_payee),
                     "payee_conf": rec.payee_name.confidence,
                     "amount_words": mw, "gt_amount_words": gw,
                     "amount_words_num": s["amount_words"], "amount_words_exact": aw_exact,
                     "logprob_full": rec.model_logprobs.get("full"),
                     "logprob_payee": rec.model_logprobs.get("payee"),
                     "micr": rec.micr_line.value})
        print(f"  [{i+1:>2}/{len(gt)}] {rel:34s} {rec.status:16s} pri={rec.routing_priority:<6} "
              f"exc={len(rec.exceptions)} {latencies[-1]:.1f}s")
    pipe.close()

    print("\n" + "=" * 56)
    print(f"V7A LIVE BENCHMARK | {args.model} | GT v16 | n={len(gt)}")
    print("=" * 56)
    means = []
    for f in FIELDS:
        pct = 100 * P[f] / T[f] if T[f] else 0
        means.append(pct)
        print(f"{f:8s} {P[f]:>3}/{T[f]:<3} = {pct:>5.1f}%")
    print("-" * 30)
    print(f"{'MEAN':8s} {sum(means)/len(means):>13.1f}%   (8-field, incl. amount_words numeric)")
    print(f"\namount_words exact-string: {aw_exact_ok}/{aw_exact_tot} = "
          f"{100*aw_exact_ok/max(aw_exact_tot,1):.1f}%  (numeric-value acc is the mean row above)")
    print(f"MICR emitted precision: {micr_emit_ok}/{micr_emit} = "
          f"{100*micr_emit_ok/max(micr_emit,1):.1f}%  (coverage {micr_emit}/{len(gt)})")
    print(f"Exceptions: {exc_total} total | routed-to-checker (priority!=NORMAL): {routed}/{len(gt)}")
    print(f"Latency: mean {sum(latencies)/len(latencies):.1f}s/cheque")

    out = os.path.join(ROOT, "benchmark", "reports", "v7a_live_benchmark.json")
    json.dump({"model": args.model, "n": len(gt),
               "fields": {f: {"correct": P[f], "total": T[f]} for f in FIELDS},
               "mean": round(sum(means)/len(means), 1),
               "amount_words_exact": {"ok": aw_exact_ok, "tot": aw_exact_tot},
               "micr_emitted": {"ok": micr_emit_ok, "emit": micr_emit},
               "exceptions_total": exc_total, "routed": routed,
               "latency_s_mean": round(sum(latencies)/len(latencies), 1),
               "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
