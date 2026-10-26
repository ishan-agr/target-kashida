"""Tokenizer fertility + vision-token budget probe.

STATUS: reference implementation written for the write-up. It was NOT part of the
original experiment log -- the research only *inferred* that GLM-OCR byte-falls-back
on Arabic. This script turns that inference into a measurement.

Two questions it answers:
  1. Fertility: how many tokens does each tokenizer spend per Arabic word?
     (byte-fallback => 3-4x more tokens => longer decode, more room to drift)
  2. Vision budget: how many LLM tokens does one cheque crop cost at a given
     resolution for a Qwen-VL style ViT (patch P, spatial merge M)?

Usage:
  python tokenizer_fertility.py --models Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3.5-2B zai-org/GLM-OCR
"""
from __future__ import annotations

import argparse
import math
import unicodedata

from transformers import AutoTokenizer

# Synthetic, non-PII Arabic name-like strings + cheque boilerplate.
SAMPLES_AR = [
    "محمد عبد الرحمن السويدي",
    "شركة النور للتجارة العامة ذ.م.م",
    "خمسة آلاف وثلاثمائة درهم فقط لا غير",
    "ادفعوا بموجب هذا الشيك لأمر",
]
SAMPLES_EN = [
    "Mohammed Abdulrahman Al Suwaidi",
    "Al Noor General Trading LLC",
    "Five thousand three hundred dirhams only",
    "Pay against this cheque to the order of",
]


def fertility(tok, texts: list[str]) -> tuple[float, float, int]:
    """Return (tokens/word, tokens/char, n_byte_fallback_tokens)."""
    n_tok = n_word = n_char = n_bytes = 0
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        n_tok += len(ids)
        n_word += len(t.split())
        n_char += len(t.replace(" ", ""))
        # SentencePiece byte fallback shows as <0xD8>; byte-level BPE shows
        # as mojibake pieces with no Arabic codepoint at all.
        n_bytes += sum(
            1 for p in pieces
            if p.startswith("<0x") or not any("ARABIC" in unicodedata.name(c, "") for c in p)
        )
    return n_tok / n_word, n_tok / n_char, n_bytes


def vision_tokens(h: int, w: int, patch: int = 16, merge: int = 2,
                  min_pixels: int = 256 * 28 * 28, max_pixels: int = 1280 * 28 * 28) -> int:
    """Qwen-VL smart_resize approximation -> LLM-side visual tokens.

    The ViT sees (H/patch)*(W/patch) patches; the merger fuses merge x merge
    neighbours, so the decoder sees that count / merge^2 tokens.
    """
    factor = patch * merge
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return (hb // patch) * (wb // patch) // (merge * merge)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    print(f"{'tokenizer':45s} {'AR tok/word':>11s} {'EN tok/word':>11s} {'AR/EN':>6s} {'AR byte-ish':>11s}")
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        ar_w, _, ar_b = fertility(tok, SAMPLES_AR)
        en_w, _, _ = fertility(tok, SAMPLES_EN)
        print(f"{name:45s} {ar_w:11.2f} {en_w:11.2f} {ar_w / en_w:6.2f} {ar_b:11d}")

    print("\nVision-token cost (Qwen3-VL: patch 16, merge 2)")
    for h, w in [(2000, 900), (1400, 630), (300, 1200), (200, 900)]:
        print(f"  {w}x{h} px -> {vision_tokens(h, w):5d} LLM tokens")


if __name__ == "__main__":
    main()
