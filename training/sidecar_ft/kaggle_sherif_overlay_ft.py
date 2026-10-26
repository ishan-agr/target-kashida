#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle QLoRA fine-tune — sherif v3, FULL cheque image -> payee-only.

Change from the June pipeline: model stays sherif v3, but this trains on the
225-image leak-free PRINTED-overlay split (full images, no crops) produced by
filter_overlay_trainset.py + prepare_sherif_overlay_data.py.

Config rationale (researched — see PIVOT_RESEARCH_APRIL_2026.md, TRAINING_DATA_LEAK_RESEARCH.md):
  * Base: sherif1313/Arabic-English-handwritten-OCR-v3 (Qwen2.5-VL-3B; multilingual base
    does NOT leak Chinese/HTML like GLM-OCR — leak doc §2). LoRA (not full-FT) also
    minimises base-knowledge memorisation (leak doc §6.3).
  * Vision FROZEN by default: task is PRINTED-Arabic payee ISOLATION, not perception —
    sherif's ViT already reads Arabic; unfreezing on 225 imgs risks forgetting. Flip with
    --unfreeze_vision to test AtlasOCR's +47% claim (their data was 30K, not 225).
  * r=32 / alpha=64: Ketaba-OCR-LoRA validated this exact rank on this exact base
    (CER 0.58 -> 0.082, 6.3x). --rank 64 = AtlasOCR compromise.
  * RSLoRA OFF (AtlasOCR: detrimental). lr=1e-4, 3 epochs (small homogeneous set).
  * TINY DATA + PRINTED overlay: expect strong fit on printed Arabic; transfer to the
    HANDWRITTEN val set is the open question — measure with eval_sherif_overlay.py.

Kaggle:
  1. Upload the images dir + sherif_overlay_train.jsonl as a dataset.
  2. GPU T4 (16GB) is enough (3B @ 4-bit, frozen vision).
  3. python kaggle_sherif_overlay_ft.py --data <jsonl> --image_dir <imgs> --output_dir ./out
"""
import argparse
import json
import os
import time


def install_dependencies():
    print("Installing dependencies...")
    os.system("pip install -q unsloth 2>/dev/null")
    os.system("pip install -q --no-deps trl peft accelerate bitsandbytes 2>/dev/null")
    print("Dependencies installed.")


def load_jsonl_as_dataset(data_path, image_dir):
    """Load JSONL, resolve bare image filenames -> absolute paths (process_vision_info
    opens local paths), and normalise any string content to list form."""
    from datasets import load_dataset

    def fix(example):
        for m in example["messages"]:
            if isinstance(m.get("content"), str):
                m["content"] = [{"type": "text", "text": m["content"]}]
            for part in m["content"]:
                if part.get("type") == "image":
                    fn = part.get("image", "")
                    if not os.path.isabs(fn):
                        part["image"] = os.path.join(image_dir, fn)
        return example

    ds = load_dataset("json", data_files=data_path, split="train")
    ds = ds.map(fix)
    return ds


def train(args):
    t0 = time.time()
    print("=" * 60)
    print("SHERIF V3 QLoRA — FULL IMAGE -> PAYEE (printed overlay split)")
    print("=" * 60)
    print(f"base={args.base_model}  data={args.data}  image_dir={args.image_dir}")
    print(f"r={args.rank} alpha={args.alpha} lr={args.lr} epochs={args.epochs} "
          f"vision={'UNFROZEN' if args.unfreeze_vision else 'FROZEN'} rslora={args.rslora}")

    import torch
    from unsloth import FastVisionModel

    print("\n[1/5] Loading sherif v3 (4-bit)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    # Transformers 5.x + Unsloth can random-init the tied lm_head (sherif ships no
    # lm_head.weight). Left unfixed, LoRA trains on top of a random head -> garbage.
    ie, oe = model.get_input_embeddings(), model.get_output_embeddings()
    if ie is not None and oe is not None and ie.weight.data_ptr() != oe.weight.data_ptr():
        oe.weight = ie.weight
        print("  [fix] re-tied lm_head to input embeddings (was untied/random)")
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\n[2/5] LoRA (r={args.rank}, alpha={args.alpha})...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.unfreeze_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        random_state=42, use_rslora=args.rslora, loftq_config=None)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {tr:,}/{tot:,} ({100*tr/tot:.2f}%)")

    print("\n[3/5] Loading data...")
    dataset = load_jsonl_as_dataset(args.data, args.image_dir)
    print(f"  {len(dataset)} samples")

    print("\n[4/5] Configuring trainer...")
    from trl import SFTTrainer, SFTConfig
    from unsloth import UnslothVisionDataCollator, is_bfloat16_supported

    FastVisionModel.for_training(model)
    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=5, save_strategy="epoch", save_total_limit=2,
        optim="adamw_8bit", weight_decay=0.01, max_grad_norm=1.0, seed=42,
        max_seq_length=args.max_seq_length,
        dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, dataloader_pin_memory=True, report_to="none")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset, args=cfg)

    print("\n[5/5] Training...")
    stats = trainer.train()
    print(f"\n  Done in {(time.time()-t0)/60:.1f} min | loss {stats.training_loss:.4f} "
          f"| peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    print("\nSaving...")
    adapter = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    merged = os.path.join(args.output_dir, "merged_bf16")
    model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
    json.dump(vars(args), open(os.path.join(args.output_dir, "train_config.json"), "w"), indent=2)
    print(f"  adapter -> {adapter}\n  merged  -> {merged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sherif_overlay_train.jsonl")
    ap.add_argument("--image_dir", default="images")
    ap.add_argument("--output_dir", default="./sherif-overlay-payee")
    ap.add_argument("--base_model", default="sherif1313/Arabic-English-handwritten-OCR-v3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--unfreeze_vision", action="store_true")
    ap.add_argument("--rslora", action="store_true")
    ap.add_argument("--install", action="store_true", help="pip install deps first (Kaggle)")
    args = ap.parse_args()
    if args.install:
        install_dependencies()
    train(args)


if __name__ == "__main__":
    main()
