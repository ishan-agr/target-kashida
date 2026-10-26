"""
Fine-tune the Invizo recognition model (CNN + Transformer, ONNX).
 
Pipeline:
  1. Convert recognition-CNN-TR-*.onnx -> a trainable PyTorch module (onnx2torch).
     This recovers the exact architecture AND the released weights, since the
     original .pth checkpoint was never published.
  2. Optionally freeze the CNN backbone (authors' recipe: FREEZE_CNN = True).
  3. Teacher-forcing training with CrossEntropy (ignore <pad>), early stop on CER.
  4. Re-export to ONNX with the SAME I/O contract the app expects:
        inputs : src (B,3,64,1024) float32, tgt (B,seq_len) int64
        output : (B,seq_len,98) float32
  5. Verify the exported ONNX loads and runs under onnxruntime.
 
Output classes stay at 98 (matches the released model) even though the shipped
OCRConfig.CHAR_TO_NUM only names 92 ids. Labels use ids 0-91; the extra 6
classes simply go untargeted. Do NOT resize the output layer.
 
Example:
  python finetune/finetune_recognition.py \
      --onnx "<HOME>/Downloads/recognition-CNN-TR-1.0.onnx" \
      --images-dir "C:/path/to/lines/images" \
      --labels-dir "C:/path/to/lines/labels" \
      --out "recognition-CNN-TR-2.0.onnx"
"""
import os
import sys
import time
import argparse
 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_patches  # noqa: F401,E402  registers missing onnx2torch converters (Trilu, ...)
from dataset import (  # noqa: E402
    LineOCRDataset, collate, NUM_TO_CHAR, PAD, SOS, EOS, SPACE, MODEL_SEQ_LEN,
)
 
try:
    from Levenshtein import distance as _lev
except ImportError:
    _lev = None
 
 
def decode_ids(ids):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i in (PAD, SOS):
            continue
        if i == SPACE:
            out.append(" ")
        elif i in NUM_TO_CHAR:
            out.append(NUM_TO_CHAR[i])
    return "".join(out)
 
 
@torch.no_grad()
def greedy_decode(model, src, device, max_steps):
    """Mirror OCRModule.greedy_decode: fixed MODEL_SEQ_LEN tgt, read pos `step`.
    Decoding is capped at max_steps (words are short) so a sample that never
    emits EOS can't force the full 249-step loop.
    """
    model.eval()
    b = src.size(0)
    tgt = torch.full((b, MODEL_SEQ_LEN), PAD, dtype=torch.long, device=device)
    tgt[:, 0] = SOS
    preds = [[] for _ in range(b)]
    done = torch.zeros(b, dtype=torch.bool, device=device)
    steps = min(max_steps, MODEL_SEQ_LEN - 1)
    for step in range(steps):
        logits = model(src, tgt)               # (B, 250, V)
        nxt = logits[:, step, :].argmax(-1)    # (B,)
        for i in range(b):
            preds[i].append(int(nxt[i]))
        tgt[:, step + 1] = nxt
        done |= (nxt == EOS)
        if bool(done.all()):
            break
    return preds                               # list[list[int]]
 
 
@torch.no_grad()
def eval_cer(model, loader, device, max_steps, max_val):
    if _lev is None:
        return float("nan")
    tot_d, tot_n, n = 0, 0, 0
    for imgs, _, texts in loader:
        imgs = imgs.to(device)
        preds = greedy_decode(model, imgs, device, max_steps)
        for ids, ref in zip(preds, texts):
            hyp = decode_ids(ids)
            tot_d += _lev(hyp, ref)
            tot_n += max(len(ref), 1)
            n += 1
        if max_val and n >= max_val:
            break
    return tot_d / max(tot_n, 1)
 
 
def export_onnx(model, path, opset):
    """Export to ONNX with the app's exact I/O; restores train mode afterwards."""
    was_training = model.training
    model.eval()
    dummy_src = torch.randn(1, 3, 64, 1024)
    dummy_tgt = torch.full((1, MODEL_SEQ_LEN), SOS, dtype=torch.long)
    torch.onnx.export(
        model, (dummy_src, dummy_tgt), path,
        input_names=["src", "tgt"], output_names=["output"],
        dynamic_axes={"src": {0: "batch_size"},
                      "tgt": {0: "batch_size"},
                      "output": {0: "batch_size"}},
        opset_version=opset,
        dynamo=False,   # stable TorchScript exporter -> graph matches the app's model
    )
    if was_training:
        model.train()
 
 
def freeze_cnn(model):
    """Freeze conv/batchnorm params (the CNN feature extractor) and set them eval.
 
    Robust to onnx2torch naming: matches by module type AND by class-name substring.
    """
    frozen = 0
    for m in model.modules():
        cls = type(m).__name__.lower()
        is_cnn = isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)) \
            or "conv" in cls or "batchnorm" in cls
        if is_cnn:
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen += p.numel()
            m.eval()
    return frozen
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="path to recognition-*.onnx")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", help="one <basename>.txt per image")
    ap.add_argument("--labels-file", help="single file: 'filename<TAB>text' per line")
    ap.add_argument("--out", default="recognition-CNN-TR-2.0.onnx")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=10, help="print progress every N batches")
    ap.add_argument("--limit", type=int, default=0, help="use only N samples (quick smoke test)")
    ap.add_argument("--max-decode-len", type=int, default=60,
                    help="cap greedy-decode steps during validation (words are short)")
    ap.add_argument("--val-samples", type=int, default=64,
                    help="evaluate CER on at most N val samples per epoch (0=all)")
    ap.add_argument("--freeze-cnn", dest="freeze_cnn", action="store_true", default=True)
    ap.add_argument("--no-freeze-cnn", dest="freeze_cnn", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
 
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[info] device = {device}")
 
    # ---- dataset -----------------------------------------------------------
    ds = LineOCRDataset(args.images_dir, args.labels_dir, args.labels_file,
                        max_len=args.max_len)
    if args.limit and args.limit < len(ds):
        import random
        random.Random(args.seed).shuffle(ds.samples)
        ds.samples = ds.samples[: args.limit]
        print(f"[info] --limit: using {len(ds)} samples (smoke test)")
    print(f"[info] usable samples: {len(ds)} | dropped: {len(ds.dropped)}")
    for fn, why in ds.dropped[:25]:
        print(f"   [drop] {fn}: {why}")
    if len(ds.dropped) > 25:
        print(f"   ... and {len(ds.dropped) - 25} more")
    if len(ds) == 0:
        sys.exit("[error] no usable samples - check paths / label encoding")
 
    n_val = max(1, int(len(ds) * args.val_split))
    n_tr = len(ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    tr, va = random_split(ds, [n_tr, n_val], generator=gen)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate)
    print(f"[info] train = {n_tr} | val = {n_val}")
    if _lev is None:
        print("[warn] python-Levenshtein not installed - CER will be NaN")
 
    # ---- model -------------------------------------------------------------
    import onnx
    from onnx2torch import convert
    print("[info] converting ONNX -> PyTorch (this can take a moment) ...")
    model = convert(onnx.load(args.onnx)).to(device)
 
    if args.freeze_cnn:
        n = freeze_cnn(model)
        print(f"[info] froze CNN params: {n:,}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[info] trainable params: {n_train:,}")
    if n_train == 0:
        sys.exit("[error] nothing trainable - try --no-freeze-cnn")
 
    opt = torch.optim.Adam(trainable, lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
 
    n_batches = len(tr_loader)
    best_cer, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_cnn:
            freeze_cnn(model)   # keep BN in eval after .train()
        run, nb, seen, t0 = 0.0, 0, 0, time.time()
        for imgs, seqs, _ in tr_loader:
            imgs, seqs = imgs.to(device), seqs.to(device)
            dec_in, target = seqs[:, :-1], seqs[:, 1:]
            logits = model(imgs, dec_in)                    # (B, T, V)
            loss = crit(logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            run += loss.item()
            nb += 1
            seen += imgs.size(0)
            if nb % args.log_every == 0 or nb == n_batches:
                rate = seen / max(time.time() - t0, 1e-6)
                print(f"[epoch {ep:>3}] batch {nb}/{n_batches} "
                      f"loss={run / nb:.4f}  {rate:.1f} img/s", flush=True)
        n_val_eval = min(args.val_samples or len(va_loader.dataset),
                         len(va_loader.dataset))
        print(f"[epoch {ep:>3}] training done, validating on {n_val_eval} samples "
              f"(<= {args.max_decode_len} decode steps) ...", flush=True)
        cer = eval_cer(model, va_loader, device, args.max_decode_len, args.val_samples)
        print(f"[epoch {ep:>3}] train_loss = {run / max(nb, 1):.4f} | "
              f"val_CER = {cer:.4f}", flush=True)
        if cer < best_cer:
            best_cer = cer
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
            export_onnx(model, args.out, args.opset)   # checkpoint: safe to stop now
            if args.freeze_cnn:
                freeze_cnn(model)                      # re-freeze BN after eval() in export
            print(f"          * new best CER {best_cer:.4f} -> saved {args.out}",
                  flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[info] early stopping at epoch {ep}")
                break
 
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[info] best val CER = {best_cer:.4f}")
 
    # ---- export ------------------------------------------------------------
    model.to("cpu")
    export_onnx(model, args.out, args.opset)
    print(f"[info] exported -> {os.path.abspath(args.out)}")
 
    # ---- verify round-trip -------------------------------------------------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        dummy_src = torch.randn(1, 3, 64, 1024)
        dummy_tgt = torch.full((1, MODEL_SEQ_LEN), SOS, dtype=torch.long)
        out = sess.run(None, {"src": dummy_src.numpy(),
                              "tgt": dummy_tgt.numpy().astype(np.int64)})[0]
        print(f"[info] verify OK: output shape {out.shape}")
        if out.shape[-1] != 98:
            print(f"[warn] expected 98 classes, got {out.shape[-1]}")
        print("[done] Rename with a higher version and drop into "
              "AiResources/models/ to deploy.")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] onnx verify failed: {e}")
 
 
if __name__ == "__main__":
    main()