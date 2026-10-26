"""Standalone date-normalizer hardening — benchmark (READ-ONLY, no pipeline change).

Two v16 date 'failures' are CORRECT dates the current parser doesn't normalize:
  outward_07  '31/Dec/2025' -> GT 2025-12-31   (textual month not handled)
  inward_01   '05/02/26'    -> GT 2026-02-05   (2-digit year not handled)
This proves the +2 (65->67 / 83 = 78.3% -> 80.7%) with ZERO regressions before we commit it
into the v7a blueprint.  Usage: python benchmark/date_normalizer_v7a.py
"""
import json, glob, re, os, sys, importlib.util
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("bs", os.path.join(root, "baseline_script.py"))
bs = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(bs)
except Exception: pass
norm_date_current = bs.normalize_date_cmp

MONTHS = {m: i for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], start=1)}
MONTHS_FULL = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
               "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}


def _year4(y):
    y = int(y)
    return y if y >= 100 else (2000 + y if y <= 79 else 1900 + y)   # 2-digit year window


def normalize_date_hardened(date_str):
    """Superset of normalize_date_cmp: also handles textual months and 2-digit years.
    Falls back to the current normalizer so nothing that worked stops working."""
    if not date_str:
        return None
    s = str(date_str).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # textual month: 31/Dec/2025, 31-Dec-25, 31 Dec 2025, 31 December 2025
    m = re.match(r"^(\d{1,2})[\s/\-.]+([A-Za-z]{3,9})[\s/\-.]+(\d{2,4})$", s)
    if m:
        mon = m.group(2).lower()
        mi = MONTHS_FULL.get(mon) or MONTHS.get(mon[:3])
        if mi:
            return f"{_year4(m.group(3)):04d}-{mi:02d}-{int(m.group(1)):02d}"
    # numeric DD/MM/YY or DD/MM/YYYY (2- or 4-digit year), with day<->month swap guard
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), _year4(m.group(3))
        if mo > 12 and d <= 12:
            d, mo = mo, d
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # last resort: current behaviour
    return norm_date_current(s)


def main():
    rep = json.load(open(sorted(glob.glob(os.path.join(root, "benchmark/reports/v7_Qwen3.5-9B*145637.json")))[-1], encoding="utf-8"))
    rows = [r for r in rep if isinstance(r, dict) and "gt_payee" in r]
    v16 = {c["image_path"]: c["fields"] for c in
           json.load(open(os.path.join(root, "cheque-poc/test-data/ground_truth_v16.json"), encoding="utf-8"))["cheques"]}
    def norm(p):
        p = str(p).replace("\\", "/"); i = p.find("test-data/"); return p[i+10:] if i >= 0 else p

    cur_pass = hard_pass = total = 0
    changed = []
    for r in rows:
        g = v16.get(norm(r.get("image")))
        if not g or not g.get("date"): continue
        raw = (r.get("extracted") or {}).get("cheque_date")
        if not raw: continue
        total += 1
        gd = g.get("date")
        c = norm_date_current(raw) if not re.match(r"^\d{4}-\d{2}-\d{2}$", raw or "") else raw
        h = normalize_date_hardened(raw)
        cp, hp = (c == gd), (h == gd)
        cur_pass += cp; hard_pass += hp
        if cp != hp:
            changed.append((norm(r["image"]), raw, c, h, gd, "FIXED" if hp else "REGRESSED"))

    print("=" * 70)
    print("DATE NORMALIZER HARDENING — benchmark (n=%d, v16)" % total)
    print("=" * 70)
    print("current  : %d/%d = %.1f%%" % (cur_pass, total, 100*cur_pass/total))
    print("hardened : %d/%d = %.1f%%   (Δ %+d)" % (hard_pass, total, 100*hard_pass/total, hard_pass-cur_pass))
    print("\nchanged rows (img | raw | current-> | hardened-> | GT | verdict):")
    for img, raw, c, h, gd, v in changed:
        print(f"  {img[:24]:24s} {str(raw)[:12]:12s} {str(c)[:11]:11s} -> {str(h)[:11]:11s}  GT={gd}  [{v}]")
    regress = [x for x in changed if x[-1] == "REGRESSED"]
    print("\nREGRESSIONS: %d   %s" % (len(regress), "(none — safe to commit)" if not regress else "(BLOCK)"))


if __name__ == "__main__":
    main()
