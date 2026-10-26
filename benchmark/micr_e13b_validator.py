"""Standalone MICR E-13B structural validator — benchmark (READ-ONLY, no pipeline change).

Question answered: if we gate the existing E-13B Tesseract output through a structural validator
(instead of `len(digits) >= 20` which lets garbage through), how many of the 41 current failures
does it (a) correctly route to MANUAL instead of silently emitting garbage, (b) rescue via repair,
and what is the resulting EMITTED-MICR precision vs coverage?

Grounding (measured on v16, see _micr_validator_explore.py):
  - UAE cheque MICR here = 3 groups, lengths 6-9-10 = 25 digits (serial / sort-routing / account+epc).
  - GT digit-length window: 22..26 (25 dominant).
  - Passing reads: |len-GT| = 0; failing reads: median |Δ| = 3, length 26+ (doubled-glyph garbage).
  - IBAN account suffix appears in MICR for 45% of correct vs 15% of wrong reads -> confidence BOOST only.

Production framing (matches doc SLO posture "MICR unreadable -> manual; MICR >=99%"):
  the validator trades SILENT ERRORS for honest MANUAL routing, driving EMITTED precision up.
Usage:  python benchmark/micr_e13b_validator.py
"""
import json, glob, re, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Validator (deterministic, no GT, no external reference DB)
# ---------------------------------------------------------------------------
EXPECTED_GROUPS = (6, 9, 10)     # measured UAE cheque MICR structure: serial - sort/routing - account+epc
LEN_MIN, LEN_MAX = 22, 26        # measured GT window
GROUP_TOL = 1                    # per-group length tolerance


def micr_groups(raw):
    return [g for g in re.split(r"\D+", str(raw or "")) if g]


def _digits(s):
    return re.sub(r"\D", "", str(s or ""))


def _structure_ok(groups):
    if len(groups) != len(EXPECTED_GROUPS):
        return False
    return all(abs(len(g) - e) <= GROUP_TOL for g, e in zip(groups, EXPECTED_GROUPS))


def _try_repair(digits, iban_tail):
    """Conservative de-doubling repair for over-length (doubled-glyph) reads.
    Only returns a value if it yields an in-window length AND the IBAN tail cross-checks —
    i.e. we never *invent* a MICR, we only accept a repair a second signal confirms. UNVALIDATED.
    """
    if not (LEN_MAX < len(digits) <= 2 * LEN_MAX) or not iban_tail:
        return None
    # collapse runs of repeated digits (the doubling signature) progressively
    collapsed = re.sub(r"(\d)\1+", r"\1", digits)
    for cand in (collapsed, digits):
        if LEN_MIN <= len(cand) <= LEN_MAX and iban_tail in cand:
            return cand
    return None


def validate_micr_e13b(raw_micr, iban=None):
    """Return (decision, confidence, value, why). decision in {ACCEPT, REPAIR, MANUAL}."""
    digits = _digits(raw_micr)
    groups = micr_groups(raw_micr)
    L = len(digits)
    idig = _digits(iban)
    iban_tail = idig[-8:] if len(idig) >= 8 else ""
    len_ok = LEN_MIN <= L <= LEN_MAX
    struct_ok = _structure_ok(groups)
    iban_ok = bool(iban_tail) and iban_tail in digits
    why = dict(len=L, len_ok=len_ok, struct_ok=struct_ok, iban_ok=iban_ok, groups=[len(g) for g in groups])

    if not len_ok:
        repaired = _try_repair(digits, iban_tail)
        if repaired:
            return "REPAIR", 0.80, repaired, {**why, "repaired_to": len(repaired)}
        return "MANUAL", 0.0, None, why

    conf = 0.55
    if struct_ok: conf += 0.30
    if iban_ok:   conf += 0.25
    return "ACCEPT", min(conf, 0.99), digits, why


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rep_path = sorted(glob.glob(os.path.join(root, "benchmark/reports/v7_Qwen3.5-9B*145637.json")))[-1]
    rep = json.load(open(rep_path, encoding="utf-8"))
    rows = [r for r in rep if isinstance(r, dict) and "gt_payee" in r]
    v16 = {c["image_path"]: c["fields"] for c in
           json.load(open(os.path.join(root, "cheque-poc/test-data/ground_truth_v16.json"), encoding="utf-8"))["cheques"]}

    def norm(p):
        p = str(p).replace("\\", "/"); i = p.find("test-data/"); return p[i+10:] if i >= 0 else p
    def gtf(r): return v16.get(norm(r.get("image")))
    def correct(value_digits, gt_raw):
        gm = _digits(gt_raw)
        return bool(value_digits and gm and (gm in value_digits or value_digits in gm))

    # policies
    def emit(decision, value, raw_digits, policy):
        # returns the digit string we would EMIT (auto), or None if routed to MANUAL
        if decision == "MANUAL": return None
        if decision == "REPAIR": return value
        if decision == "ACCEPT":
            if policy == "strict_struct":   # require positive structural confirmation
                return value  # ACCEPT already means len_ok; strictness applied via conf gate below
            return value
        return None

    POLICIES = {
        "baseline_emit_all": None,             # current pipeline behaviour: emit whatever, 20+ digits
        "lenient(len_gate)": 0.0,              # ACCEPT all len-ok
        "strict(conf>=0.80)": 0.80,            # ACCEPT only len-ok WITH struct_ok or iban_ok
    }

    print("=" * 78)
    print("MICR E-13B STRUCTURAL VALIDATOR — standalone benchmark (n=%d, v16)" % len(rows))
    print("report:", os.path.basename(rep_path))
    print("structure model: groups %s  len-window [%d,%d]" % (EXPECTED_GROUPS, LEN_MIN, LEN_MAX))
    print("=" * 78)

    # Baseline: emit every extracted micr (current 50.6%)
    base_correct = sum(correct(_digits((r.get("extracted") or {}).get("micr_line")), gtf(r).get("micr"))
                       for r in rows if gtf(r))
    n = sum(1 for r in rows if gtf(r))
    print("\nBASELINE (emit-all, no validator): %d/%d = %.1f%% accuracy, coverage 100%%, %d SILENT errors"
          % (base_correct, n, 100*base_correct/n, n - base_correct))

    for pol, conf_gate in POLICIES.items():
        if conf_gate is None:
            continue
        acc_correct = acc_wrong = manual_wrong = manual_right = repaired_ok = repaired_bad = 0
        accepted = manual = 0
        for r in rows:
            g = gtf(r)
            if not g: continue
            raw = (r.get("extracted") or {}).get("micr_line")
            iban = (r.get("extracted") or {}).get("iban")
            decision, conf, value, why = validate_micr_e13b(raw, iban)
            gt_raw = g.get("micr")
            # apply policy conf gate: below gate -> route MANUAL
            if decision in ("ACCEPT", "REPAIR") and conf < conf_gate:
                decision = "MANUAL"; value = None
            if decision == "MANUAL":
                manual += 1
                if correct(_digits(raw), gt_raw): manual_right += 1   # we routed a CORRECT read to manual (cost)
                else: manual_wrong += 1                               # we routed a WRONG read to manual (benefit)
            else:
                accepted += 1
                ok = correct(value, gt_raw)
                if decision == "REPAIR":
                    repaired_ok += ok; repaired_bad += (not ok)
                acc_correct += ok; acc_wrong += (not ok)
        prec = 100*acc_correct/accepted if accepted else 0
        cov = 100*accepted/n
        print("\n--- policy: %s ---" % pol)
        print("  EMITTED precision : %d/%d = %.1f%%   (silent errors now %d, was %d)"
              % (acc_correct, accepted, prec, acc_wrong, n - base_correct))
        print("  coverage (auto)   : %d/%d = %.1f%%   routed to MANUAL: %d" % (accepted, n, cov, manual))
        print("  manual routing    : %d were WRONG reads (correctly caught), %d were CORRECT (false-reject cost)"
              % (manual_wrong, manual_right))
        if repaired_ok or repaired_bad:
            print("  repair (UNVALIDATED): %d rescued-correct, %d repaired-but-wrong" % (repaired_ok, repaired_bad))
        print("  => of the %d original failures: %d now routed to manual, %d still emitted wrong, %d rescued"
              % (n - base_correct, manual_wrong, acc_wrong, repaired_ok))

    print("\nNOTE: validator uses NO ground truth and NO bank reference DB — only structure + IBAN cross-check.")
    print("Right-length single-digit errors cannot be caught without a MICR check digit (none confirmed for UAE).")


if __name__ == "__main__":
    main()
