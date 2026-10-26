#!/usr/bin/env python3
"""
QLoRA Training Script for Qwen3-VL-8B — Cheque Arabic OCR Fine-Tuning.

Uses Unsloth for 2x faster training with 60% less memory.

Supports two modes:
  A) WARM START (recommended): Continue from U4RASD/ar-ms-baseline adapter
     - Arabic manuscript LoRA already trained on Qwen3-VL-8B-Instruct
     - r=32, alpha=64, targets: q/k/v/o_proj + gate/up/down_proj
     - Provides Arabic handwriting foundation, needs cheque-specific tuning
  B) FRESH START: Train new LoRA adapter from scratch
     - r=32, alpha=64 (reduced from AtlasOCR r=128 to prevent quantization bin shift)
     - RoLoRA paper: r=128 W4 → 37.0% (31pp collapse) vs r=32 more stable

Training recipe (validated against AtlasOCR research):
- Vision encoder: UNFROZEN (critical for Arabic glyph recognition)
- RSLoRA: OFF (AtlasOCR finding: degrades OCR tasks)
- Gradient checkpointing: ON (saves VRAM)

Usage:
    # Warm start from Arabic manuscript adapter (RECOMMENDED)
    python train_qlora.py \
        --data merged_train.jsonl \
        --warm_start U4RASD/ar-ms-baseline \
        --output ./cheque-adapter-v14 \
        --rank 32 --alpha 64 --lr 1e-4 \
        --batch_size 2 --gradient_accumulation 4 \
        --epochs 3

    # Fresh start (no warm-start)
    python train_qlora.py \
        --data merged_train.jsonl \
        --output ./cheque-adapter-v14-fresh \
        --rank 32 --alpha 64 --lr 1e-4 \
        --epochs 3
"""

import argparse
import json
import os
import sys
from pathlib import Path


def train(
    data_path: str,
    output_dir: str,
    warm_start_adapter: str = "",
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    rank: int = 128,
    alpha: int = 128,
    lr: float = 2e-4,
    max_seq_length: int = 2048,
    vision_unfrozen: bool = True,
):
    """
    Run QLoRA fine-tuning with Unsloth.

    Args:
        data_path: Path to training JSONL
        output_dir: Output directory
        warm_start_adapter: HuggingFace adapter ID or local path for warm start
        epochs: Training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation
        rank: LoRA rank (ignored in warm start mode)
        alpha: LoRA alpha (ignored in warm start mode)
        lr: Learning rate
        max_seq_length: Max sequence length
        vision_unfrozen: Whether to fine-tune vision encoder
    """
    print("=" * 60)
    print("Qwen3-VL-8B QLoRA Fine-Tuning (Unsloth)")
    if warm_start_adapter:
        print(f"MODE: WARM START from {warm_start_adapter}")
    else:
        print(f"MODE: FRESH START (r={rank}, alpha={alpha})")
    print("=" * 60)

    from unsloth import FastVisionModel, UnslothVisionDataCollator
    import torch

    # ── Step 1: Load base model ──────────────────────────────────────
    print("\n[1/5] Loading base model with Unsloth...")

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    print(f"  Base model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 2: Apply or load LoRA adapters ──────────────────────────
    if warm_start_adapter:
        # WARM START: Load existing adapter and continue training
        print(f"\n[2/5] Loading warm-start adapter: {warm_start_adapter}")

        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            warm_start_adapter,
            is_trainable=True,  # Enable continued training
        )

        # Get adapter config for logging
        adapter_config = model.peft_config.get("default", None)
        if adapter_config:
            actual_rank = adapter_config.r
            actual_alpha = adapter_config.lora_alpha
            print(f"  Adapter loaded: r={actual_rank}, alpha={actual_alpha}")
            print(f"  Target modules: {adapter_config.target_modules}")
        else:
            actual_rank = rank
            actual_alpha = alpha
            print("  Adapter loaded (config not readable)")

        # Ensure all LoRA params are trainable
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        print("  Warm start ready - continuing training on cheque data")

    else:
        # FRESH START: Create new LoRA adapter
        print(f"\n[2/5] Applying fresh LoRA (r={rank}, alpha={alpha})...")

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=vision_unfrozen,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0,
            bias="none",
            random_state=42,
            use_rslora=False,  # AtlasOCR: RSLoRA OFF for OCR tasks
            loftq_config=None,
        )

        if vision_unfrozen:
            print("  Vision encoder: UNFROZEN (LoRA applied to ViT)")
        else:
            print("  Vision encoder: FROZEN")

        actual_rank = rank
        actual_alpha = alpha

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    print(f"  VRAM after LoRA: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 3: Load training data ───────────────────────────────────
    print(f"\n[3/5] Loading training data from {data_path}...")

    # Load and normalize: some Arabic dataset samples have content as plain
    # string instead of list, which breaks PyArrow's schema inference.
    # Normalize all content fields to list format before loading.
    import tempfile
    normalized_path = data_path
    with open(data_path, "r", encoding="utf-8") as f:
        first_lines = [f.readline() for _ in range(100)]
    needs_fix = any(
        isinstance(m.get("content"), str)
        for line in first_lines if line.strip()
        for m in json.loads(line).get("messages", [])
    )
    if needs_fix:
        print("  Normalizing mixed content formats...")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, dir="."
        )
        fixed = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for m in obj.get("messages", []):
                    if isinstance(m.get("content"), str):
                        m["content"] = [{"type": "text", "text": m["content"]}]
                        fixed += 1
                tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp.close()
        normalized_path = tmp.name
        print(f"  Fixed {fixed} string->list content fields")

    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files=normalized_path,
        split="train",
    )
    if normalized_path != data_path:
        os.remove(normalized_path)
    print(f"  Loaded {len(dataset)} training samples")

    # ── Step 4: Configure training ───────────────────────────────────
    print(f"\n[4/5] Configuring training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Max sequence length: {max_seq_length}")

    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported

    training_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        max_seq_length=max_seq_length,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # ── Step 5: Train ────────────────────────────────────────────────
    print(f"\n[5/5] Starting training...")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_config,
        data_collator=UnslothVisionDataCollator(model, tokenizer, max_seq_length=max_seq_length),
    )

    print(f"  VRAM before training: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    stats = trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    print(f"\nTraining complete!")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Final loss: {stats.training_loss:.4f}")
    print(f"  Time: {stats.metrics.get('train_runtime', 0)/60:.1f} min")

    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  Adapter saved to: {adapter_dir}")

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "base_model": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
            "warm_start_adapter": warm_start_adapter or None,
            "rank": actual_rank,
            "alpha": actual_alpha,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "vision_unfrozen": vision_unfrozen,
            "use_rslora": False,
            "max_seq_length": max_seq_length,
            "final_loss": stats.training_loss,
            "total_steps": stats.global_step,
            "training_samples": len(dataset),
        }, f, indent=2)

    print(f"\nNext steps:")
    print(f"  1. python merge_for_vllm.py --adapter {adapter_dir} --base_model Qwen/Qwen3-VL-8B-Instruct --output ./cheque-merged-v14 --skip_quantize")
    print(f"  2. vllm serve ./cheque-merged-v14_fp16 --dtype float8 --trust-remote-code  (FP8 primary)")
    print(f"  3. python quantize_awq_domain.py --model ./cheque-merged-v14_fp16 --output ./cheque-merged-v14_awq4  (AWQ secondary)")
    print(f"  NOTE: Do NOT use merge_adapter.py (has Qwen2_5_VL class bug). Use merge_for_vllm.py instead.")

    return adapter_dir


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen3-VL-8B cheque extraction"
    )
    parser.add_argument("--data", "-d", required=True, help="Training JSONL path")
    parser.add_argument("--output", "-o", default="./qwen3vl-cheque-qlora", help="Output dir")
    parser.add_argument(
        "--warm_start", "-w",
        default="",
        help="HuggingFace adapter ID for warm start (e.g., U4RASD/ar-ms-baseline). "
             "If empty, trains from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank (ignored in warm start mode). "
                             "Default 32 matches U4RASD warm-start config.")
    parser.add_argument("--alpha", type=int, default=64,
                        help="LoRA alpha (ignored in warm start mode). "
                             "Default 64 gives scaling factor 2.0 (64/32).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate. Default 1e-4 for warm-start "
                             "(preserves Arabic knowledge from U4RASD).")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Freeze vision encoder (NOT recommended)")

    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        warm_start_adapter=args.warm_start,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        vision_unfrozen=not args.freeze_vision,
    )


if __name__ == "__main__":
    main()