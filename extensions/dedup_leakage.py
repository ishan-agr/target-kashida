"""Near-duplicate + leakage audit for small document-image fine-tuning sets.

STATUS: reference implementation for the write-up. In the project, leakage control
was done with (a) full-relative-path template exclusion in
training/sidecar_ft/filter_overlay_trainset.py, (b) group-K-fold by cheque_id in
training/sidecar_ft/kaggle_sherif_ft.py and (c) a sha256-frozen eval manifest in
training/active_learning/al_freeze_eval.py. Exact hashes miss the real danger in
cheque data: the SAME printed template (same bank, same background) re-scanned or
augmented. This script adds perceptual + text-level near-duplicate detection.

  image side : pHash (64-bit DCT hash) -> Hamming distance clusters
  text side  : MinHash over character 3-grams of the label (Arabic-normalised)
  split      : union-find over both graphs -> GroupKFold on connected components

Usage:
  python dedup_leakage.py --train train.jsonl --eval eval.jsonl --img-root data/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imagehash
from datasketch import MinHash, MinHashLSH
from PIL import Image

_TASHKEEL = re.compile(r"[ً-ٰٟـ]")


def norm_ar(s: str) -> str:
    s = _TASHKEEL.sub("", s)
    s = re.sub("[أإآٱ]", "ا", s).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", s).strip().lower()


def minhash(text: str, k: int = 3, perms: int = 128) -> MinHash:
    m = MinHash(num_perm=perms)
    t = f" {norm_ar(text)} "
    for i in range(max(1, len(t) - k + 1)):
        m.update(t[i:i + k].encode("utf-8"))
    return m


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def load(path: str, split: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        user = r["messages"][0]["content"]
        img = next(c["image"] for c in user if c["type"] == "image")
        label = r["messages"][1]["content"][0]["text"]
        rows.append({"split": split, "image": img.removeprefix("file://"), "label": label})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--img-root", default=".")
    ap.add_argument("--phash-thresh", type=int, default=8, help="max Hamming distance (of 64)")
    ap.add_argument("--jaccard", type=float, default=0.8)
    args = ap.parse_args()

    rows = load(args.train, "train") + load(args.eval, "eval")
    hashes = [imagehash.phash(Image.open(Path(args.img_root) / r["image"]).convert("L")) for r in rows]
    dsu = DSU(len(rows))

    # O(n^2) is fine below ~20k images; switch to a BK-tree beyond that.
    img_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if hashes[i] - hashes[j] <= args.phash_thresh:
                dsu.union(i, j)
                img_pairs += 1

    lsh = MinHashLSH(threshold=args.jaccard, num_perm=128)
    mhs = [minhash(r["label"]) for r in rows]
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            dsu.union(i, int(j))
        lsh.insert(str(i), m)

    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(dsu.find(i), []).append(i)

    leaks = [g for g in groups.values() if {rows[i]["split"] for i in g} == {"train", "eval"}]
    print(f"rows={len(rows)} groups={len(groups)} near-dup image pairs={img_pairs}")
    print(f"LEAKING groups (span train+eval): {len(leaks)}")
    for g in leaks[:20]:
        print("  ", [(rows[i]["split"], rows[i]["image"]) for i in g])

    # Emit a group id per row so any trainer can do GroupKFold on it.
    out = Path("group_ids.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"image": r["image"], "split": r["split"], "group": dsu.find(i)}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
