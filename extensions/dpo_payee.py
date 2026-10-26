"""DPO on a VLM sidecar with self-mined hard negatives.

STATUS: designed for the write-up, NOT executed in the project. SFT kept producing
three failure classes that a likelihood loss cannot push down directly:
  (a) template text read as the payee ("ادفعوا لأمر", "or bearer")
  (b) the payer/drawer name copied into the payee slot
  (c) memorised training names emitted on unrelated cheques (seen after V15)
Each of those outputs is a natural `rejected` completion, and the ground truth is
`chosen`. DPO pushes up log pi(chosen) - log pi(rejected) relative to a frozen
reference, without needing a reward model:

  L = -log sigma( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                         - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

Pair-mining step:   python dpo_payee.py mine  --preds preds.jsonl --gt gt.jsonl
Training step:      python dpo_payee.py train --pairs pairs.jsonl --model <sft-ckpt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT = ("اقرأ اسم المستفيد المكتوب بخط اليد فقط في هذه الصورة. "
          "Return ONLY the handwritten payee name.")


def mine(preds_path: str, gt_path: str, out: str) -> None:
    """Every wrong sampled read becomes a rejected completion for its crop."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from src.v7a.text_utils import fuzzy_name_match  # the project's frozen scorer

    gt = {r["crop"]: r for r in map(json.loads, Path(gt_path).read_text("utf-8").splitlines())}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for p in map(json.loads, Path(preds_path).read_text("utf-8").splitlines()):
            g = gt.get(p["crop"])
            if not g:
                continue
            # p["samples"]: N reads at T>0 + any template/payer strings seen in the pipeline
            for bad in {s for s in p["samples"] if s and not fuzzy_name_match(s, g["payee"])}:
                f.write(json.dumps({"image": p["crop"], "chosen": g["payee"], "rejected": bad},
                                   ensure_ascii=False) + "\n")
                n += 1
            if g.get("payer"):  # synthetic class (b) negative
                f.write(json.dumps({"image": p["crop"], "chosen": g["payee"], "rejected": g["payer"]},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} preference pairs -> {out}")


def train(pairs_path: str, model_id: str, out_dir: str) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import DPOConfig, DPOTrainer

    rows = [json.loads(l) for l in Path(pairs_path).read_text("utf-8").splitlines()]

    def to_example(r):
        return {
            "images": [Image.open(r["image"]).convert("RGB")],
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            "chosen": [{"role": "assistant", "content": [{"type": "text", "text": r["chosen"]}]}],
            "rejected": [{"role": "assistant", "content": [{"type": "text", "text": r["rejected"]}]}],
        }

    ds = Dataset.from_list([to_example(r) for r in rows])
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    args = DPOConfig(
        output_dir=out_dir,
        beta=0.1,                    # KL strength vs the SFT reference
        loss_type="sigmoid",         # vanilla DPO; "ipo" if pairs are noisy
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        logging_steps=5,
        report_to="mlflow",          # see extensions/mlflow_tracking.py
    )
    # With a PEFT config TRL uses the adapter-disabled base as pi_ref: no second copy in VRAM.
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")
    DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
               processing_class=processor, peft_config=peft).train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine"); m.add_argument("--preds"); m.add_argument("--gt"); m.add_argument("--out", default="pairs.jsonl")
    t = sub.add_parser("train"); t.add_argument("--pairs"); t.add_argument("--model"); t.add_argument("--out", default="dpo_out")
    a = ap.parse_args()
    mine(a.preds, a.gt, a.out) if a.cmd == "mine" else train(a.pairs, a.model, a.out)
