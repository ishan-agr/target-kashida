#!/usr/bin/env python3
"""
Kaggle Training Script — Sherif v3 QLoRA Fine-Tuning for Payee Field Isolation.

Target: Kaggle T4x2 (2x 16GB VRAM) or single T4/P100 (16GB).

This script:
  1. Installs dependencies (Unsloth, etc.)
  2. Downloads sherif v3 BF16 base model
  3. Loads pre-prepared training JSONL (from prepare_sherif_ft_data.py)
  4. Runs QLoRA fine-tuning with frozen vision encoder
  5. Saves merged BF16 model + GGUF quantized for CPU inference

Key design decisions:
  - Base: sherif v3 BF16 (NOT Qwen2.5-VL-3B-Instruct — sherif already has 47K Arabic HW samples)
  - Vision: FROZEN (sherif's ViT already knows Arabic chars, unfreezing risks forgetting with <700 samples)
  - LR: 1e-5 (low — NeurIPS 2025: low LR prevents forgetting in re-fine-tuning)
  - Rank: 32 (conservative — small dataset, less overfitting risk)
  - Epochs: 5 (small dataset benefits from more passes)

Kaggle usage:
  1. Upload cheque images + training JSONL as Kaggle dataset
  2. Select GPU T4x2 accelerator
  3. Run this script

  Or paste into notebook cells (see cell markers below).
"""

import argparse
import json
import os
import sys
import time


def install_dependencies():
    """Install Unsloth and dependencies on Kaggle."""
    print("Installing dependencies...")
    os.system("pip install -q unsloth[kaggle] 2>/dev/null")
    os.system("pip install -q --no-deps trl peft accelerate bitsandbytes xformers 2>/dev/null")
    os.system("pip install -q huggingface_hub 2>/dev/null")
    print("Dependencies installed.")


def train_sherif_ft(
    data_path: str,
    output_dir: str = "./sherif-ft-payee",
    base_model: str = "sherif1313/Arabic-English-handwritten-OCR-v3",
    epochs: int = 5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    rank: int = 32,
    alpha: int = 64,
    lr: float = 1e-5,
    max_seq_length: int = 1024,
    freeze_vision: bool = True,
    save_gguf: bool = True,
):
    """
    QLoRA fine-tuning of sherif v3 for payee field isolation.

    Args:
        data_path: Path to training JSONL (from prepare_sherif_ft_data.py)
        output_dir: Where to save the fine-tuned model
        base_model: HuggingFace model ID (sherif v3 BF16)
        epochs: Training epochs (5 recommended for small dataset)
        batch_size: Per-device batch size (1 for 16GB GPU)
        gradient_accumulation_steps: Effective batch = batch_size * this
        rank: LoRA rank (32 — conservative for small dataset)
        alpha: LoRA alpha (64 = 2x rank — standard scaling)
        lr: Learning rate (1e-5 — low to prevent catastrophic forgetting)
        max_seq_length: Max tokens (1024 — payee names are short)
        freeze_vision: Whether to freeze vision encoder (recommended: True)
        save_gguf: Whether to also save GGUF quantized for CPU inference
    """
    start_time = time.time()

    print("=" * 60)
    print("SHERIF V3 QLoRA FINE-TUNING — Payee Field Isolation")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Data: {data_path}")
    print(f"Config: r={rank}, alpha={alpha}, lr={lr}, epochs={epochs}")
    print(f"Vision encoder: {'FROZEN' if freeze_vision else 'UNFROZEN'}")
    print(f"Effective batch: {batch_size * gradient_accumulation_steps}")
    print()

    # ── Step 1: Load base model ──────────────────────────────────────
    print("[1/5] Loading sherif v3 with Unsloth (4-bit for training)...")
    import torch
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded. VRAM: {vram_after_load:.1f} GB")

    # ── Step 2: Apply LoRA adapter ───────────────────────────────────
    print(f"\n[2/5] Applying LoRA (r={rank}, alpha={alpha})...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not freeze_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,  # RSLoRA OFF — degrades OCR tasks (AtlasOCR finding)
        loftq_config=None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  VRAM after LoRA: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── Step 3: Load training data ───────────────────────────────────
    print(f"\n[3/5] Loading training data...")

    # Normalize content format (some samples may have string content)
    import tempfile
    normalized_path = data_path
    with open(data_path, "r", encoding="utf-8") as f:
        first_lines = [f.readline() for _ in range(50)]
    needs_fix = any(
        isinstance(m.get("content"), str)
        for line in first_lines if line.strip()
        for m in json.loads(line).get("messages", [])
    )
    if needs_fix:
        print("  Normalizing mixed content formats...")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for m in obj.get("messages", []):
                    if isinstance(m.get("content"), str):
                        m["content"] = [{"type": "text", "text": m["content"]}]
                tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp.close()
        normalized_path = tmp.name

    from datasets import load_dataset

    dataset = load_dataset("json", data_files=normalized_path, split="train")
    if normalized_path != data_path:
        os.remove(normalized_path)
    print(f"  Loaded {len(dataset)} training samples")

    # Source distribution
    sources = {}
    for sample in dataset:
        src = sample.get("_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    # ── Step 4: Configure training ───────────────────────────────────
    print(f"\n[4/5] Configuring training...")

    from trl import SFTTrainer, SFTConfig
    from unsloth import UnslothVisionDataCollator, is_bfloat16_supported

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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=training_config,
    )

    # ── Step 5: Train ────────────────────────────────────────────────
    print(f"\n[5/5] Starting training...")
    print(f"  VRAM before train: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"  GPU: {gpu_stats.name} ({gpu_stats.total_mem / 1e9:.1f} GB)")

    trainer_stats = trainer.train()

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # ── Save model ───────────────────────────────────────────────────
    print("\nSaving model...")

    # Save LoRA adapter
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  LoRA adapter: {adapter_dir}")

    # Save merged BF16 model (for direct use)
    merged_dir = os.path.join(output_dir, "merged_bf16")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"  Merged BF16: {merged_dir}")

    # Save GGUF for CPU inference (optional)
    if save_gguf:
        print("  Saving GGUF quantized (Q8_0 for CPU)...")
        try:
            gguf_dir = os.path.join(output_dir, "gguf")
            model.save_pretrained_gguf(
                gguf_dir, tokenizer,
                quantization_method="q8_0",
            )
            print(f"  GGUF Q8_0: {gguf_dir}")
        except Exception as e:
            print(f"  GGUF save failed (non-critical): {e}")
            print("  You can convert manually with llama.cpp later.")

    # Save training config for reproducibility
    config = {
        "base_model": base_model,
        "rank": rank,
        "alpha": alpha,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "freeze_vision": freeze_vision,
        "training_samples": len(dataset),
        "training_loss": trainer_stats.training_loss,
        "training_time_minutes": elapsed / 60,
        "source_distribution": sources,
    }
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")

    print(f"\nDone! Total time: {elapsed/60:.1f} minutes")
    return output_dir


# ---------------------------------------------------------------------------
# 5-Fold Cross-Validation (HONEST held-out evaluation)
# ---------------------------------------------------------------------------
#
# WHY: the 31 Arabic payee crops are simultaneously our only in-domain training
# signal AND our only benchmark (the 41.4% sherif baseline). Training on them
# and re-scoring them is a data leak (see TRAINING_DATA_LEAK_RESEARCH.md).
# Group-K-fold by cheque_id gives an UNBIASED estimate: each Arabic cheque is
# scored only by a model that never saw it (in any prompt/augmentation variant).
# Base samples (KHATT + English + negatives) stay in train for every fold.
# Eval uses the crops_gt MANUAL crops + oracle best-of-2 prompts, so the number
# is directly comparable to the 41.4% baseline.
#
# CV is for MEASUREMENT only — it trains K throwaway models and saves no
# artifact. After CV gives you an honest number, run the normal single-run
# train_sherif_ft() on ALL data to produce the deployable adapter.

import re as _re
import unicodedata as _ud

EVAL_PROMPTS = [
    "Read the handwritten name in this image. Return ONLY the name, nothing else.",
    "اقرأ الاسم المكتوب بخط اليد في هذه الصورة. أعد الاسم فقط، لا شيء آخر.",
]
_ARABIC_RE = _re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def _has_arabic(t):
    return bool(_ARABIC_RE.search(t or ""))


def _normalize_name(name):
    """Identical to benchmark/sherif_v3_manual_crops.normalize_name."""
    if not name:
        return ""
    name = _ud.normalize("NFKC", name).lower().strip()
    name = _re.sub(r'[.,()\[\]{}"\'/\\]+', ' ', name)
    name = _re.sub(r'\s+', ' ', name).strip()
    name = _re.sub(r'[أإآٱ]', 'ا', name)
    name = _re.sub(r'ؤ', 'و', name)
    name = _re.sub(r'ئ', 'ي', name)
    name = _re.sub(r'[ً-ٰٟ]', '', name)
    name = name.replace('ة', 'ه')
    return name.strip()


def _fuzzy_name_match(extracted, ground_truth):
    """Identical to benchmark/sherif_v3_manual_crops.fuzzy_name_match (the baseline scorer)."""
    if not extracted or not ground_truth:
        return False
    e, g = _normalize_name(extracted), _normalize_name(ground_truth)
    if not e or not g:
        return False
    if e == g or e in g or g in e:
        return True
    for part in g.split('/'):
        part = part.strip()
        if part and (e == part or e in part or part in e):
            return True
    e_words, g_words = set(e.split()), set(g.split())
    if not g_words:
        return False
    return len(e_words & g_words) / len(g_words) >= 0.5


def _build_sherif_model(base_model, rank, alpha, freeze_vision, max_seq_length):
    """Build a fresh 4-bit sherif + LoRA for one CV fold."""
    import torch
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not freeze_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=rank, lora_alpha=alpha, lora_dropout=0, bias="none",
        random_state=42, use_rslora=False, loftq_config=None,
    )
    return model, tokenizer


def _sherif_infer(model, tokenizer, pil_image, prompt, max_new_tokens=64):
    """Single-image inference on an Unsloth FastVisionModel."""
    import torch
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(pil_image, input_text, add_special_tokens=False,
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             use_cache=True, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    if text and "<" in text:
        text = _re.sub(r"<[^>]+>", " ", text)
        text = _re.sub(r"\s+", " ", text).strip()
    return text.strip() if text and text.strip() else None


def _evaluate_held_out(model, tokenizer, held_out, crops_gt_dir):
    """Score held-out cheques on their crops_gt MANUAL crop, oracle best-of-2 prompts.

    held_out: list of dicts {cheque_id, crop_name, gt_payee}
    Returns (n_pass, n_total, details).
    """
    from unsloth import FastVisionModel
    from PIL import Image
    FastVisionModel.for_inference(model)
    n_pass = 0
    details = []
    for c in held_out:
        crop_path = os.path.join(crops_gt_dir, c["crop_name"])
        if not os.path.exists(crop_path):
            details.append({**c, "status": "no_crop"})
            continue
        img = Image.open(crop_path)
        best = False
        preds = {}
        for p in EVAL_PROMPTS:
            try:
                txt = _sherif_infer(model, tokenizer, img, p)
            except Exception as e:
                txt = None
                preds["error"] = str(e)
            m = _fuzzy_name_match(txt, c["gt_payee"])
            preds[p[:20]] = {"pred": txt, "match": m}
            best = best or m
        n_pass += int(best)
        details.append({**c, "oracle_match": best, "preds": preds})
    return n_pass, len(held_out), details


def _group_kfold_cheque_ids(eval_cheques, k, seed=42):
    """Deterministically split eval-able cheque_ids into k folds."""
    import random as _r
    ids = sorted(eval_cheques.keys())
    rng = _r.Random(seed)
    rng.shuffle(ids)
    return [ids[i::k] for i in range(k)]


def train_sherif_cv(
    data_path, crops_gt_dir, eval_gt_json,
    base_model="sherif1313/Arabic-English-handwritten-OCR-v3",
    cv_folds=5, epochs=5, batch_size=1, gradient_accumulation_steps=4,
    rank=32, alpha=64, lr=1e-5, max_seq_length=1024, freeze_vision=True,
    output_dir="./sherif-cv",
):
    """Group-K-fold CV over Arabic cheque_ids. Honest held-out fuzzy estimate."""
    import json as _json
    import torch
    from datasets import Dataset

    print("=" * 60)
    print(f"SHERIF V3 — {cv_folds}-FOLD CROSS-VALIDATION (honest held-out)")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    # Load all training samples
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(_json.loads(line))
    print(f"Loaded {len(samples)} samples from {data_path}")

    # Arabic cheque samples are foldable (grouped by cheque_id); everything else is base
    arabic_by_id = {}
    base_samples = []
    for s in samples:
        meta = s.get("_meta", {})
        src = s.get("_source", "")
        if src in ("cheque_payee", "cheque_payee_aug") and meta.get("is_arabic"):
            arabic_by_id.setdefault(meta.get("cheque_id"), []).append(s)
        else:
            base_samples.append(s)
    print(f"Arabic cheque_ids (foldable): {len(arabic_by_id)} | base (always-train): {len(base_samples)}")

    # GT map + eval-able cheques (Arabic GT AND a crops_gt manual crop present)
    gt = _json.load(open(eval_gt_json, encoding="utf-8"))
    eval_cheques = {}
    for c in gt["cheques"]:
        payee = c["fields"].get("payee_name", "")
        if not _has_arabic(payee):
            continue
        cid = c.get("id", c["image_path"])
        crop_name = c["image_path"].replace("/", "_")
        if os.path.exists(os.path.join(crops_gt_dir, crop_name)) and cid in arabic_by_id:
            eval_cheques[cid] = {"cheque_id": cid, "crop_name": crop_name, "gt_payee": payee}
    print(f"Eval-able Arabic cheques (have crops_gt crop + training samples): {len(eval_cheques)}")

    folds = _group_kfold_cheque_ids(eval_cheques, cv_folds)
    for i, fl in enumerate(folds):
        print(f"  fold {i}: {len(fl)} cheques")

    from trl import SFTTrainer, SFTConfig
    from unsloth import UnslothVisionDataCollator, is_bfloat16_supported, FastVisionModel

    all_details = []
    total_pass = total_n = 0
    for fold_idx, held_ids in enumerate(folds):
        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx+1}/{cv_folds} — holding out {len(held_ids)} cheques")
        print("=" * 60)
        held_set = set(held_ids)

        # Train set = base + arabic samples whose cheque_id is NOT held out
        train_samples = list(base_samples)
        for cid, sl in arabic_by_id.items():
            if cid not in held_set:
                train_samples.extend(sl)
        # Normalize string content -> list (collator expects list)
        for s in train_samples:
            for m in s.get("messages", []):
                if isinstance(m.get("content"), str):
                    m["content"] = [{"type": "text", "text": m["content"]}]
        ds = Dataset.from_list(train_samples)
        print(f"  train samples: {len(ds)} | held-out cheques: {len(held_ids)}")

        model, tokenizer = _build_sherif_model(base_model, rank, alpha, freeze_vision, max_seq_length)
        FastVisionModel.for_training(model)
        cfg = SFTConfig(
            output_dir=os.path.join(output_dir, f"fold{fold_idx}"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs, learning_rate=lr,
            lr_scheduler_type="cosine", warmup_ratio=0.1,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=10, save_strategy="no", optim="adamw_8bit",
            weight_decay=0.01, max_grad_norm=1.0, seed=42,
            max_seq_length=max_seq_length, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False, report_to="none",
        )
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=ds, args=cfg,
        )
        stats = trainer.train()
        print(f"  fold {fold_idx} loss={stats.training_loss:.4f} "
              f"peakVRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        held_out = [eval_cheques[cid] for cid in held_ids]
        n_pass, n_total, details = _evaluate_held_out(model, tokenizer, held_out, crops_gt_dir)
        print(f"  fold {fold_idx} held-out: {n_pass}/{n_total} = "
              f"{100*n_pass/max(1,n_total):.1f}% oracle fuzzy")
        for d in details:
            tag = "PASS" if d.get("oracle_match") else d.get("status", "FAIL")
            print(f"    [{tag}] {d['cheque_id']}  gt={d['gt_payee'][:40]}")
        all_details.extend(details)
        total_pass += n_pass
        total_n += n_total

        # Free for next fold
        del model, trainer, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    print("\n" + "=" * 60)
    print(f"CV RESULT — {total_pass}/{total_n} = {100*total_pass/max(1,total_n):.1f}% "
          f"held-out oracle fuzzy  (baseline raw sherif = 41.4%)")
    print("=" * 60)
    report = {
        "config": {"base_model": base_model, "cv_folds": cv_folds, "epochs": epochs,
                   "rank": rank, "alpha": alpha, "lr": lr, "freeze_vision": freeze_vision,
                   "data_path": data_path, "eval_gt_json": eval_gt_json},
        "held_out_pass": total_pass, "held_out_total": total_n,
        "held_out_fuzzy_pct": round(100*total_pass/max(1, total_n), 2),
        "baseline_raw_pct": 41.4,
        "details": all_details,
    }
    rp = os.path.join(output_dir, "cv_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report: {rp}")
    return report


# ---------------------------------------------------------------------------
# Kaggle notebook cell version
# ---------------------------------------------------------------------------

KAGGLE_NOTEBOOK_CELLS = """
# =====================================================================
# Cell 1: Install dependencies
# =====================================================================
%%capture
!pip install unsloth[kaggle]
!pip install --no-deps trl peft accelerate bitsandbytes xformers
!pip install huggingface_hub datasets pillow

# =====================================================================
# Cell 2: Upload and configure data
# =====================================================================
# Option A: Upload sherif_ft_train.jsonl as Kaggle dataset
# Option B: Upload cheque images + GT and run prepare_sherif_ft_data.py

DATA_PATH = "/kaggle/input/your-dataset/sherif_ft_train.jsonl"
OUTPUT_DIR = "/kaggle/working/sherif-ft-payee"

# =====================================================================
# Cell 3: Train
# =====================================================================
# Paste the train_sherif_ft() function from kaggle_sherif_ft.py here,
# or upload the script and import it:
#
# import sys
# sys.path.insert(0, "/kaggle/input/your-dataset")
# from kaggle_sherif_ft import train_sherif_ft
#
# train_sherif_ft(
#     data_path=DATA_PATH,
#     output_dir=OUTPUT_DIR,
#     epochs=5,
#     lr=1e-5,
#     rank=32,
#     alpha=64,
#     freeze_vision=True,
# )

# =====================================================================
# Cell 4: Upload to HuggingFace (optional)
# =====================================================================
# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_folder(
#     folder_path=f"{OUTPUT_DIR}/merged_bf16",
#     repo_id="your-username/sherif-v3-payee-ft",
#     repo_type="model",
# )
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sherif v3 QLoRA fine-tuning for payee field isolation"
    )
    parser.add_argument("--data", required=True, help="Training JSONL path")
    parser.add_argument("--output", default="./sherif-ft-payee", help="Output directory")
    parser.add_argument("--base_model", default="sherif1313/Arabic-English-handwritten-OCR-v3",
                        help="Base model ID")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--unfreeze_vision", action="store_true",
                        help="Unfreeze vision encoder (risky with small data)")
    parser.add_argument("--no_gguf", action="store_true",
                        help="Skip GGUF quantization")
    parser.add_argument("--install_deps", action="store_true",
                        help="Install Unsloth/dependencies first (for Kaggle)")
    parser.add_argument("--print_notebook", action="store_true",
                        help="Print Kaggle notebook cell template")
    # 5-fold CV (honest held-out evaluation) ----------------------------------
    parser.add_argument("--cv_folds", type=int, default=0,
                        help="If >1, run group-K-fold CV (by cheque_id) for an honest "
                             "held-out number instead of a single train run.")
    parser.add_argument("--crops_gt_dir",
                        default="benchmark/sherif_v3_arabic_outward/crops_gt",
                        help="Manual benchmark crops used for held-out eval (CV mode).")
    parser.add_argument("--eval_gt_json",
                        default="cheque-poc/test-data/ground_truth_v15-m.json",
                        help="GT json for held-out eval labels (CV mode).")

    args = parser.parse_args()

    if args.print_notebook:
        print(KAGGLE_NOTEBOOK_CELLS)
        sys.exit(0)

    if args.install_deps:
        install_dependencies()

    if args.cv_folds and args.cv_folds > 1:
        train_sherif_cv(
            data_path=args.data,
            crops_gt_dir=args.crops_gt_dir,
            eval_gt_json=args.eval_gt_json,
            base_model=args.base_model,
            cv_folds=args.cv_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            rank=args.rank,
            alpha=args.alpha,
            lr=args.lr,
            max_seq_length=args.max_seq_length,
            freeze_vision=not args.unfreeze_vision,
            output_dir=args.output,
        )
        sys.exit(0)

    train_sherif_ft(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        freeze_vision=not args.unfreeze_vision,
        save_gguf=not args.no_gguf,
    )
