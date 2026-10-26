"""Logit distillation: large on-prem VLM teacher -> <=3 GB sidecar student.

STATUS: designed for the write-up, NOT executed in the project. The no-external-API
rule rules out a frontier model as a production teacher. A frontier model could
still be used as an offline labelling oracle, subject to data-governance sign-off;
this script assumes an on-prem teacher instead. The teacher is a
bigger sibling run in BF16 on a rented card, so it shares the student's tokenizer and
we can match full distributions rather than just strings.

  loss = a * CE(student, y)                                   # hard labels (GT)
       + (1-a) * T^2 * KL( softmax(z_t/T) || softmax(z_s/T) ) # soft labels

The KL term only covers assistant tokens. This is the Hinton et al. (2015) objective
applied to sequences, i.e. token-level KD. The sequence-level variant (Kim & Rush,
2016) amounts to SFT on teacher greedy outputs, which is cheaper and was the
fallback.

Usage:
  python distill_sidecar.py --teacher Qwen/Qwen3.5-9B --student Qwen/Qwen3.5-2B \
      --data sidecar_train.jsonl --out kd_student
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup


def batch_iter(rows, processor, device):
    for r in rows:
        msgs = r["messages"]
        img = Image.open(next(c["image"] for c in msgs[0]["content"] if c["type"] == "image")
                         .removeprefix("file://")).convert("RGB")
        full = processor.apply_chat_template(msgs, tokenize=False)
        prompt = processor.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        enc = processor(text=[full], images=[img], return_tensors="pt").to(device)
        n_prompt = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100            # loss on the answer only (cf. train_peft.py caveat)
        yield enc, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="kd_student")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda"
    proc = AutoProcessor.from_pretrained(args.student)
    assert AutoProcessor.from_pretrained(args.teacher).tokenizer.get_vocab() == proc.tokenizer.get_vocab(), \
        "token-level KD needs a shared vocabulary; use sequence-level KD otherwise"
    teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, torch_dtype=torch.bfloat16).to(dev).eval()
    student = AutoModelForImageTextToText.from_pretrained(args.student, torch_dtype=torch.bfloat16).to(dev)
    student.gradient_checkpointing_enable()

    rows = [json.loads(l) for l in Path(args.data).read_text("utf-8").splitlines()]
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(rows) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)

    for ep in range(args.epochs):
        for step, (enc, labels) in enumerate(batch_iter(rows, proc, dev)):
            with torch.no_grad():
                zt = teacher(**enc).logits[:, :-1].float()
            out = student(**enc)
            zs = out.logits[:, :-1].float()
            y = labels[:, 1:]
            m = y != -100
            ce = F.cross_entropy(zs[m], y[m])
            kd = F.kl_div(F.log_softmax(zs[m] / args.T, -1), F.log_softmax(zt[m] / args.T, -1),
                          log_target=True, reduction="batchmean") * args.T ** 2
            loss = args.alpha * ce + (1 - args.alpha) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"ep{ep} step{step} ce={ce.item():.4f} kd={kd.item():.4f}")

    student.save_pretrained(args.out)
    proc.save_pretrained(args.out)


if __name__ == "__main__":
    main()
