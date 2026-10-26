#!/usr/bin/env python3
"""Quantize cheque-merged-v15 using auto-round with mixed domain+general calibration.

Uses 50/50 mix of cheque_train_v15.jsonl (domain) and NeelNanda/pile-10k (general).
Domain samples are concatenated to reach seqlen since individual cheque samples are short.
"""

import json
import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from auto_round import AutoRound

model_dir = "./cheque-merged-v15_fp16"
output_dir = "./cheque-merged-v15_awq4"
calib_jsonl = "./cheque_train_v15.jsonl"
SEQLEN = 2048

# ── Load model on GPU ──────────────────────────────────────────────
print("Loading model on GPU...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── Build mixed calibration data (domain + general) ───────────────
print(f"\nBuilding mixed calibration dataset (seqlen={SEQLEN})...")

# Domain data: cheque training samples with chat template, concatenated
raw_domain = []
with open(calib_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        sample = json.loads(line)
        msgs = sample.get("messages", [])
        clean_msgs = []
        for msg in msgs:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [item["text"] for item in content if item.get("type") == "text"]
                clean_msgs.append({"role": msg["role"], "content": " ".join(text_parts)})
            elif isinstance(content, str):
                clean_msgs.append({"role": msg["role"], "content": content})
        if clean_msgs:
            try:
                formatted = tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=False)
                raw_domain.append(formatted)
            except Exception:
                texts = [m["content"] for m in clean_msgs if m["content"]]
                raw_domain.append(" ".join(texts))

print(f"  Raw domain samples: {len(raw_domain)}")

# Concatenate short domain samples into seqlen-sized chunks
domain_data = []
buffer = ""
for text in raw_domain:
    buffer += text + "\n"
    # Rough char estimate: ~4 chars per token, so seqlen*4 chars ≈ seqlen tokens
    if len(buffer) > SEQLEN * 3:
        domain_data.append(buffer)
        buffer = ""
if buffer:
    domain_data.append(buffer)

print(f"  Domain chunks (concatenated): {len(domain_data)}")

# General data: pile-10k (already long enough)
from datasets import load_dataset
pile = load_dataset("NeelNanda/pile-10k", split="train")
general_data = []
for sample in pile:
    text = sample.get("text", "")
    if len(text) > SEQLEN * 2:  # Only keep long enough samples
        general_data.append(text)
    if len(general_data) >= 128:
        break
print(f"  General samples (pile-10k): {len(general_data)}")

# Mix: domain + general
calib_data = domain_data + general_data
print(f"  Total calibration samples: {len(calib_data)}")

# Verify token lengths
sample_lens = []
for text in calib_data[:10]:
    toks = tokenizer(text, truncation=False)["input_ids"]
    sample_lens.append(len(toks))
print(f"  Sample token lengths (first 10): min={min(sample_lens)}, max={max(sample_lens)}, avg={sum(sample_lens)//len(sample_lens)}")

# ── Quantize on GPU ───────────────────────────────────────────────
print(f"\nStarting quantization (GPU-accelerated, 200 iters)...")
print(f"  GPU: {torch.cuda.get_device_name(0)}")

autoround = AutoRound(
    model=model,
    tokenizer=tokenizer,
    processor=processor,
    dataset=calib_data,
    scheme="W4A16",
    seqlen=SEQLEN,
    nsamples=256,
    batch_size=8,
    iters=200,
    low_gpu_mem_usage=False,
)

model, _ = autoround.quantize_and_save(output_dir, format="auto_awq")

total_size = sum(
    os.path.getsize(os.path.join(output_dir, f))
    for f in os.listdir(output_dir)
    if f.endswith((".safetensors", ".bin"))
)
print(f"\nQuantized model saved to: {output_dir}")
print(f"Model size: {total_size / 1e9:.1f} GB")
