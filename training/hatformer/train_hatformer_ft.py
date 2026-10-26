"""HATFormer FULL (unfrozen) fine-tune on the 30 upscaled crops_gt.

Replicates HATFormer's real training mechanics (src/lightning.py + data.py):
  - base = hatformer-muharaf, FULL fine-tune (encoder ViT + TrOCR decoder both trainable)
  - custom Arabic tokenizer (arabic_tokenizer_clean), special tokens, token-id wiring
  - loss = model(pixel_values, labels, interpolate_pos_encoding=True).loss  (seq2seq CE)
  - preprocessing = UPSCALE (baked into prepared crops) -> height-64 -> RTL flip ->
    384x384 segment-stack -> TrOCR normalize   (data.py new_processor=True)
Small-data adaptations (30 crops): online augmentation, differential LR
(encoder<decoder to preserve Arabic features), warmup+cosine, fp16, early-stop on
held-out generate->CER.

BASE SELECTION RATIONALE (see header comment in code).

Run (local RTX 4060 or Kaggle T4/P100):
  python training/train_hatformer_ft.py --epochs 60 --batch 4 --val_frac 0.2
  python training/train_hatformer_ft.py --epochs 80 --val_frac 0.0   # pure overtrain on all 30
"""
import os, io, csv, re, time, math, argparse, random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (VisionEncoderDecoderModel, PreTrainedTokenizerFast,
                          TrOCRProcessor, get_cosine_schedule_with_warmup)

# ---------------------------------------------------------------------------
# BASE SELECTION RATIONALE
# ---------------------------------------------------------------------------
# Base = hatformer-muharaf (NOT trocr-base-stage1, NOT hatformer-synthetic):
#   1. It already owns the custom Arabic BBPE tokenizer + a decoder that emits
#      Arabic. trocr-base-stage1 has an English RoBERTa vocab -> 30 crops cannot
#      relearn an Arabic decoder/tokenizer from scratch.
#   2. Its ViT encoder + decoder LM are already fine-tuned on Muharaf real Arabic
#      handwriting, so we only need DOMAIN TRANSFER (manuscript -> cheque), not
#      capability acquisition -- the only regime where 30 samples can work.
#   3. Empirically the best HATFormer variant on cheques (16.7% fuzzy vs
#      synthetic's 3.3% / CER 8.861). Synthetic is degenerate -> rejected.
# UNFROZEN rationale: the manuscript->cheque gap is largely VISUAL (scan, printed
# template, ink). The encoder must adapt to the cheque pixel distribution;
# freezing it caps adaptation. Mitigate 334M-params-on-30-samples overfit with
# heavy online augmentation + low differential LR (encoder << decoder) + early stop.
# ---------------------------------------------------------------------------

HAT = r"<LOCAL_PATH>/HATFormer/HATFormer"
DEFAULT_CKPT = HAT + "/hatformer-muharaf"
DEFAULT_TOK = HAT + "/arabic_tokenizer_clean/tokenizer.json"
DEFAULT_PROC = "microsoft/trocr-base-stage1"   # official HATFormer image processor (README L57)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training" / "data" / "hatformer_crops_upscaled"


def fuzzy_name_match(extracted, ground_truth):
    if not extracted or not ground_truth:
        return extracted is None and ground_truth is None
    ext = extracted.lower().strip(); gt = ground_truth.lower().strip()
    if ext == gt: return True
    if ext in gt or gt in ext: return True
    gt_parts = re.split(r"\s*/\s*", gt); ext_parts = re.split(r"\s*/\s*", ext)
    for gp in gt_parts:
        gp = gp.strip()
        if not gp: continue
        if gp in ext or ext in gp: return True
        for ep in ext_parts:
            ep = ep.strip()
            if ep and (gp in ep or ep in gp): return True
        gw = set(re.findall(r"\w+", gp)); ew = set(re.findall(r"\w+", ext))
        if gw and len(gw & ew) / len(gw) >= 0.5: return True
    return False


def cer(pred, ref):
    if not ref: return 0.0 if not pred else 1.0
    a, b = pred, ref; dp = list(range(len(b)+1))
    for i in range(1, len(a)+1):
        prev = dp[0]; dp[0] = i
        for j in range(1, len(b)+1):
            cur = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (a[i-1]!=b[j-1])); prev = cur
    return dp[len(b)]/len(ref)


# ---- augmentation (mild, HTR-safe; applied to the upscaled RGB crop) ----
def augment(img_bgr, rng):
    h, w = img_bgr.shape[:2]
    out = img_bgr
    # small rotation
    if rng.random() < 0.7:
        ang = rng.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=(255, 255, 255))
    # slight shear/affine
    if rng.random() < 0.4:
        sh = rng.uniform(-0.04, 0.04)
        M = np.float32([[1, sh, 0], [0, 1, 0]])
        out = cv2.warpAffine(out, M, (w, h), borderValue=(255, 255, 255))
    # horizontal stretch
    if rng.random() < 0.4:
        sx = rng.uniform(0.92, 1.08)
        out = cv2.resize(out, None, fx=sx, fy=1.0, interpolation=cv2.INTER_LINEAR)
    # brightness / contrast
    if rng.random() < 0.6:
        alpha = rng.uniform(0.85, 1.15); beta = rng.uniform(-12, 12)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # blur
    if rng.random() < 0.25:
        k = rng.choice([3, 3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    # gaussian noise
    if rng.random() < 0.25:
        noise = rng.normal(0, rng.uniform(3, 10), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# ---- HATFormer preprocessing (data.py new_processor=True) on a PIL image ----
def hatformer_pixels(processor, pil_img):
    img = pil_img.convert("RGB")
    ow, oh = img.size
    new_h = 64; new_w = max(1, int(new_h * (ow / oh)))
    r = img.resize((new_w, new_h)).transpose(Image.FLIP_LEFT_RIGHT)
    canvas = Image.new("RGB", (384, 384), (0, 0, 0))
    if r.width <= 384:
        canvas.paste(r, (0, 0))
    else:
        for i in range((r.width + 383) // 384):
            l = i * 384
            canvas.paste(r.crop((l, 0, min(l + 384, r.width), new_h)), (0, i * new_h))
    return processor(canvas, return_tensors="pt").pixel_values[0]


class CropDataset(Dataset):
    def __init__(self, rows, processor, tokenizer, max_len, train, seed=42):
        self.rows = rows; self.processor = processor; self.tok = tokenizer
        self.max_len = max_len; self.train = train
        self.rng = random.Random(seed); self.nprng = np.random.RandomState(seed)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        bgr = cv2.imread(r["img_path"])
        if self.train:
            # numpy RandomState for noise + python Random for choices, both seeded per-call
            class _R:
                def __init__(s, p, n): s.p = p; s.n = n
                def random(s): return s.p.random()
                def uniform(s, a, b): return s.p.uniform(a, b)
                def choice(s, x): return s.p.choice(x)
                def normal(s, m, sd, shape): return s.n.normal(m, sd, shape)
            bgr = augment(bgr, _R(self.rng, self.nprng))
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pv = hatformer_pixels(self.processor, pil)
        ids = self.tok(r["text"], padding="max_length", truncation=True,
                       max_length=self.max_len).input_ids
        labels = [t if t != self.tok.pad_token_id else -100 for t in ids]
        return {"pixel_values": pv, "labels": torch.tensor(labels), "text": r["text"]}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def load_rows(csv_path, arabic_only):
    rows = []
    with io.open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if arabic_only and not int(r["is_arabic"]):
                continue
            rows.append(r)
    return rows


def build_model_tok(ckpt, tok_file, device):
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tok.add_special_tokens({'pad_token': '<pad>', 'eos_token': '</s>',
                            'cls_token': '<s>', 'bos_token': '<s>'})
    # Load in fp32 for training (checkpoint is fp16; GradScaler needs fp32 master
    # weights + autocast does the mixed-precision compute).
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, torch_dtype=torch.float32)
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(device)
    return model, tok


@torch.no_grad()
def evaluate(model, tok, processor, rows, device, beam, max_new_tokens):
    model.eval()
    exact = fuzzy = 0; cers = []
    details = []
    for r in rows:
        pil = Image.open(r["img_path"])
        pv = hatformer_pixels(processor, pil).unsqueeze(0).to(device)
        out = model.generate(pv, num_beams=beam, length_penalty=0.0,
                             max_new_tokens=max_new_tokens)
        pred = tok.batch_decode(out.tolist(), skip_special_tokens=True)[0].strip()
        g = r["text"]
        exact += int(pred == g); fz = fuzzy_name_match(pred, g); fuzzy += int(fz)
        c = cer(pred, g); cers.append(c)
        details.append((r["img_path"], g, pred, fz, c))
    n = len(rows)
    return {"n": n, "exact": exact, "fuzzy": fuzzy,
            "fuzzy_pct": 100*fuzzy/max(1, n), "mean_cer": sum(cers)/max(1, len(cers)),
            "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--processor", default=DEFAULT_PROC)
    ap.add_argument("--csv", default=str(DATA_DIR / "labels.csv"))
    ap.add_argument("--output", default=str(ROOT / "training" / "hatformer-ft"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr_encoder", type=float, default=1e-5)
    ap.add_argument("--lr_decoder", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_text_length", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="0.0 = train on all 30 (pure overtrain/leak); >0 = held-out CER early stop")
    ap.add_argument("--val_stems", default=None,
                    help="Comma-separated crop stems to FORCE into val (e.g. the worst-CER "
                         "crops). Overrides random val selection; val_frac then ignored.")
    ap.add_argument("--arabic_only", action="store_true", help="drop the 1 English crop")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    rows = load_rows(args.csv, args.arabic_only)
    def _stem(r): return Path(r["img_path"]).stem
    if args.val_stems:
        want = {s.strip() for s in args.val_stems.split(",") if s.strip()}
        val_rows = [r for r in rows if _stem(r) in want]
        train_rows = [r for r in rows if _stem(r) not in want]
        found = {_stem(r) for r in val_rows}
        missing = want - found
        print(f"val_stems: forcing {len(val_rows)} worst-CER crops into VAL"
              + (f" | MISSING {missing}" if missing else ""))
    elif args.val_frac > 0:
        random.Random(args.seed).shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        val_rows, train_rows = rows[:n_val], rows[n_val:]
    else:
        val_rows, train_rows = rows, rows  # eval on all (overtrain)
    print(f"device={device} | crops total={len(rows)} | train={len(train_rows)} | "
          f"val={len(val_rows)} | val_frac={args.val_frac}")
    print("VAL crops:", sorted(_stem(r) for r in val_rows))

    processor = TrOCRProcessor.from_pretrained(args.processor)
    model, tok = build_model_tok(args.ckpt, args.tokenizer, device)
    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"model: {Path(args.ckpt).name} | {n_params:.0f}M params (ALL trainable) | "
          f"VRAM after load {torch.cuda.memory_allocated()/1e9:.2f}GB" if device=="cuda" else
          f"model: {Path(args.ckpt).name} | {n_params:.0f}M params (ALL trainable)")

    train_ds = CropDataset(train_rows, processor, tok, args.max_text_length, train=True, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=0)

    # differential LR: encoder (preserve Arabic visual feats) << decoder
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": dec_params, "lr": args.lr_decoder},
    ], betas=(0.9, 0.999), eps=1e-6, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps*args.warmup_ratio), total_steps)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} | "
          f"lr_enc={args.lr_encoder} lr_dec={args.lr_decoder}")
    best_cer = 1e9; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = 0.0; opt.zero_grad()
        for it, batch in enumerate(train_dl):
            pv = batch["pixel_values"].to(device); labels = batch["labels"].to(device)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(pixel_values=pv, labels=labels, interpolate_pos_encoding=True)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            if (it + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            run += out.loss.item()
        ev = evaluate(model, tok, processor, val_rows, device, args.beam, args.max_text_length)
        peak = torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0
        print(f"[ep {ep+1:>3}/{args.epochs}] loss={run/len(train_dl):.4f} | "
              f"val fuzzy={ev['fuzzy']}/{ev['n']}={ev['fuzzy_pct']:.1f}% CER={ev['mean_cer']:.3f} "
              f"| peakVRAM={peak:.1f}GB")
        if ev["mean_cer"] < best_cer:
            best_cer = ev["mean_cer"]
            model.save_pretrained(os.path.join(args.output, "best"))
            tok.save_pretrained(os.path.join(args.output, "best"))
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min | best val CER={best_cer:.3f}")
    # final report on val_rows with best
    print("Per-sample (last epoch):")
    for path, g, pred, fz, c in ev["details"]:
        print(f"  [{'FUZZY' if fz else '    '}] CER={c:.2f} | gt={g[:32]} | pred={pred[:32]}")


if __name__ == "__main__":
    main()
