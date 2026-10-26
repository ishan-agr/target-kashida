"""Freeze an immutable active-learning EVAL set (ADR-0001, Phase 0).

The eval set is the small, hand-labeled Arabic-payee crop set used to benchmark ANY sidecar
(Sherif v3 / HATFormer). It is content-hash-locked here so the OFFLINE training harness can assert
that NO eval crop leaks into training — the training-data-leak guard from project memory.

Writes <repo>/eval_frozen/manifest.json:
    {
      "frozen_ts": "<UTC ISO, from --ts or now>",
      "source_dir": "<abs path hashed>",
      "n": <count>,
      "set_hash": "<sha256 of the sorted per-item sha list — tamper-evident>",
      "items": [ {"sha256": "<crop hash>", "file": "<name>", "label": "<gt or null>"} ]
    }

The offline trainer MUST load this manifest and drop any pool sample whose sample_id (== crop sha256)
appears in `items[].sha256`. That is the only thing standing between "held-out eval" and a leak.

Usage:
  python training/al_freeze_eval.py --crops <dir-of-eval-pngs> \
        --repo data/active_learning/v7a \
        [--gt ground_truth.json] [--gt-key payee_name] [--ts 2026-07-11T00:00:00Z]

--gt (optional): a JSON mapping {filename: label} OR the project's ground_truth_*.json
  (list under "cheques" with image_path + fields[gt-key]); labels are matched by basename.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_labels(gt_path: str, gt_key: str) -> dict:
    """Best-effort {basename: label}. Supports a flat map or the project's cheques-list GT."""
    if not gt_path:
        return {}
    data = json.load(open(gt_path, encoding="utf-8"))
    labels = {}
    if isinstance(data, dict) and "cheques" in data:
        for c in data["cheques"]:
            base = os.path.basename(c.get("image_path", ""))
            val = (c.get("fields") or {}).get(gt_key)
            if base and val is not None:
                labels[base] = val
    elif isinstance(data, dict):
        for k, v in data.items():
            labels[os.path.basename(k)] = v
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="directory of eval crop PNGs (immutable set)")
    ap.add_argument("--repo", default="data/active_learning/v7a", help="AL store root")
    ap.add_argument("--gt", default="", help="optional GT json (flat map or cheques-list)")
    ap.add_argument("--gt-key", default="payee_name")
    ap.add_argument("--ts", default="", help="freeze timestamp (UTC ISO); default = now")
    args = ap.parse_args()

    ts = args.ts or datetime.now(timezone.utc).isoformat()
    labels = _load_labels(args.gt, args.gt_key)

    items = []
    for name in sorted(os.listdir(args.crops)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        sha = _sha256_file(os.path.join(args.crops, name))
        items.append({"sha256": sha, "file": name, "label": labels.get(name)})

    set_hash = hashlib.sha256("".join(sorted(it["sha256"] for it in items)).encode()).hexdigest()
    manifest = {
        "frozen_ts": ts,
        "source_dir": os.path.abspath(args.crops),
        "n": len(items),
        "set_hash": set_hash,
        "items": items,
    }

    out_dir = os.path.join(args.repo, "eval_frozen")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(out_path):
        prev = json.load(open(out_path, encoding="utf-8"))
        if prev.get("set_hash") != set_hash:
            print(f"WARNING: existing manifest set_hash differs ({prev.get('set_hash')[:12]}... -> "
                  f"{set_hash[:12]}...). The eval set is meant to be IMMUTABLE — overwriting.")
    json.dump(manifest, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    labeled = sum(1 for it in items if it["label"] is not None)
    print(f"froze {len(items)} eval crops ({labeled} labeled) -> {out_path}")
    print(f"set_hash={set_hash}")


if __name__ == "__main__":
    main()
