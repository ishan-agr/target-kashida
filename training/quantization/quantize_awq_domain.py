#!/usr/bin/env python3
"""
Domain-Calibrated AWQ Quantization — Uses cheque training data as calibration.

Key difference from previous attempt: calibration data is Arabic cheque samples
instead of English ultrachat. AWQ computes per-channel activation scales from
this data — domain match means Arabic-important channels get correct scaling.

Usage:
    python quantize_awq_domain.py \
        --model ./cheque-merged-v14_fp16 \
        --calib_data ./cheque_train_v14.jsonl \
        --output ./cheque-merged-v14_awq4
"""

import argparse
import json
import os


def quantize(model_dir: str, calib_jsonl: str, output_dir: str, n_samples: int = 256):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading model from {model_dir}...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Build domain-matched calibration data from cheque training samples
    print(f"Loading calibration data from {calib_jsonl}...")
    calib_data = []
    with open(calib_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            # Extract text content from messages
            texts = []
            for msg in sample.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item["text"])
            if texts:
                calib_data.append(" ".join(texts))
            if len(calib_data) >= n_samples:
                break

    print(f"  Calibration samples: {len(calib_data)}")

    print("Quantizing with domain-matched calibration...")
    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4},
        calib_data=calib_data,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith((".safetensors", ".bin"))
    )
    print(f"  Quantized model size: {total_size / 1e9:.1f} GB")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWQ quantization with domain-matched calibration data"
    )
    parser.add_argument("--model", required=True, help="Merged FP16 model directory")
    parser.add_argument(
        "--calib_data", default="cheque_train_v14.jsonl", help="Calibration JSONL"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--n_samples", type=int, default=256, help="Number of calibration samples"
    )
    args = parser.parse_args()
    quantize(args.model, args.calib_data, args.output, args.n_samples)
