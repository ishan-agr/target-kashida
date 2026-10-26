#!/usr/bin/env python3
"""
Merge LoRA Adapter + AWQ Quantize → vLLM-ready model.

Single script that does: Load FP16 base → Merge adapter → AWQ quantize → Upload to HF.
Run this on the GPU server (or RunPod) to prepare models for vLLM serving.

Supports both adapters:
  1. U4RASD/ar-ms-baseline (Qwen3-VL-8B, r=32, Arabic manuscript)
  2. presightai/arabic-image-to-markdown (Qwen2.5-VL-7B, r=8, Arabic docs)

Usage:
    # Merge U4RASD adapter into Qwen3-VL-8B + quantize
    python merge_for_vllm.py \
        --adapter U4RASD/ar-ms-baseline \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --output ./u4rasd_merged_awq \
        --upload your-hf-username/Qwen3-VL-8B-ar-ms-AWQ

    # Merge presightai adapter into Qwen2.5-VL-7B + quantize
    python merge_for_vllm.py \
        --adapter presightai/arabic-image-to-markdown-qwen2.5vl-7b-instruct-lora \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --output ./presightai_merged_awq \
        --upload your-hf-username/Qwen25-VL-7B-arabic-doc-AWQ

    # Skip quantization (serve FP16 directly — needs more VRAM)
    python merge_for_vllm.py \
        --adapter U4RASD/ar-ms-baseline \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --output ./u4rasd_merged_fp16 \
        --skip_quantize

    # Quantize an already-merged FP16 model (skip merge step)
    python merge_for_vllm.py \
        --skip_merge \
        --merged_dir ./u4rasd_merged_awq_fp16 \
        --output ./u4rasd_merged_awq \
        --upload cyankiwi/Qwen3-VL-8B-ar-ms-AWQ

    # Upload FP16 merged model for benchmarking (no quantization)
    python merge_for_vllm.py \
        --skip_merge --skip_quantize \
        --merged_dir ./u4rasd_merged_awq_fp16 \
        --output ./u4rasd_merged_awq_fp16 \
        --upload cyankiwi/Qwen3-VL-8B-ar-ms-FP16
"""

import argparse
import gc
import json
import os
import sys
import time


def detect_model_class(base_model: str) -> str:
    """Detect the correct model class based on base model name or config.json."""
    name_lower = base_model.lower()

    # Check by name first
    if "qwen3" in name_lower and "vl" in name_lower:
        return "Qwen3VLForConditionalGeneration"
    elif ("qwen2.5" in name_lower or "qwen2_5" in name_lower) and "vl" in name_lower:
        return "Qwen2_5_VLForConditionalGeneration"
    elif "qwen2" in name_lower and "vl" in name_lower:
        return "Qwen2VLForConditionalGeneration"

    # Check config.json for local directories
    config_path = os.path.join(base_model, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
        architectures = config.get("architectures", [])
        # Check architectures list
        for arch in architectures:
            if "Qwen3VL" in arch:
                return "Qwen3VLForConditionalGeneration"
            elif "Qwen2_5_VL" in arch or "Qwen25VL" in arch:
                return "Qwen2_5_VLForConditionalGeneration"
            elif "Qwen2VL" in arch:
                return "Qwen2VLForConditionalGeneration"
        # Check model_type
        if "qwen3_vl" in model_type:
            return "Qwen3VLForConditionalGeneration"
        elif "qwen2_5_vl" in model_type:
            return "Qwen2_5_VLForConditionalGeneration"
        elif "qwen2_vl" in model_type:
            return "Qwen2VLForConditionalGeneration"
        # Check merge_metadata.json if present
        meta_path = os.path.join(base_model, "merge_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return meta.get("model_class", "AutoModelForCausalLM")

    # Check by name without VL requirement (for model IDs like Qwen3-VL-8B)
    if "qwen3" in name_lower:
        return "Qwen3VLForConditionalGeneration"
    elif "qwen2.5" in name_lower or "qwen2_5" in name_lower:
        return "Qwen2_5_VLForConditionalGeneration"

    return "AutoModelForCausalLM"


def merge_adapter(
    adapter_id: str,
    base_model: str,
    output_dir: str,
):
    """Load base model + LoRA adapter, merge, save FP16."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    model_class_name = detect_model_class(base_model)

    print("=" * 60)
    print(f"STEP 1: MERGE ADAPTER")
    print(f"  Base: {base_model}")
    print(f"  Adapter: {adapter_id}")
    print(f"  Model class: {model_class_name}")
    print("=" * 60)

    # Dynamic model class import
    if model_class_name == "Qwen3VLForConditionalGeneration":
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    elif model_class_name == "Qwen2_5_VLForConditionalGeneration":
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import AutoModelForCausalLM as ModelClass

    # Load FP16 base
    print(f"\n  Loading base model (FP16)...")
    t0 = time.time()
    model = ModelClass.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Load processor — try AutoProcessor first, fall back to tokenizer-only
    # if torchvision is missing (causes VideoProcessor lookup to crash)
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    except TypeError:
        print("  WARNING: AutoProcessor failed (missing torchvision?), loading tokenizer only")
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"  Loaded in {time.time()-t0:.0f}s. "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load adapter
    print(f"\n  Loading adapter: {adapter_id}")
    model = PeftModel.from_pretrained(model, adapter_id, torch_dtype=torch.float16)

    # Log adapter info
    config = model.peft_config.get("default")
    if config:
        print(f"  Adapter config: r={config.r}, alpha={config.lora_alpha}")
        print(f"  Target modules: {config.target_modules}")

    # Merge
    print(f"\n  Merging adapter into base weights...")
    model = model.merge_and_unload()
    print(f"  Merge complete")

    # Save merged FP16
    merged_fp16_dir = output_dir + "_fp16"
    os.makedirs(merged_fp16_dir, exist_ok=True)
    print(f"\n  Saving merged FP16 to: {merged_fp16_dir}")
    model.save_pretrained(merged_fp16_dir, safe_serialization=True)
    processor.save_pretrained(merged_fp16_dir)

    # Save metadata
    with open(os.path.join(merged_fp16_dir, "merge_metadata.json"), "w") as f:
        json.dump({
            "base_model": base_model,
            "adapter": adapter_id,
            "model_class": model_class_name,
            "dtype": "float16",
            "adapter_r": config.r if config else None,
            "adapter_alpha": config.lora_alpha if config else None,
        }, f, indent=2)

    total_size = sum(
        os.path.getsize(os.path.join(merged_fp16_dir, f))
        for f in os.listdir(merged_fp16_dir)
        if f.endswith((".safetensors", ".bin"))
    )
    print(f"  FP16 model size: {total_size / 1e9:.1f} GB")

    # Free GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_fp16_dir, processor


def quantize_awq(merged_fp16_dir: str, output_dir: str):
    """AWQ quantize the merged FP16 model for vLLM serving."""
    print("\n" + "=" * 60)
    print(f"STEP 2: AWQ QUANTIZE")
    print(f"  Input: {merged_fp16_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    import torch

    try:
        # Method 1: llm-compressor (vLLM recommended)
        # Try multiple import paths (API changed across versions)
        oneshot = None
        recipe = None

        # v0.10.0+ API
        try:
            from llmcompressor import oneshot as _oneshot
            oneshot = _oneshot
            print("\n  Using llm-compressor v0.10+ API...")
        except ImportError:
            pass

        # Older API
        if oneshot is None:
            try:
                from llmcompressor.transformers import oneshot as _oneshot
                oneshot = _oneshot
                print("\n  Using llm-compressor (legacy API)...")
            except ImportError:
                pass

        if oneshot is None:
            raise ImportError("llmcompressor oneshot not found")

        # Build ignore list — MUST exclude vision encoder for VLMs.
        # The ViT MLP layers have dimension 4304 which is not divisible by
        # group_size 128 (4304 % 128 = 80). This is standard practice:
        # only quantize the language model backbone, keep vision encoder FP16.
        #
        # IMPORTANT: llm-compressor v0.10.0 uses EXACT name matching, not
        # prefix matching. Use "re:..." regex patterns to match all sublayers.
        # Reference: https://github.com/vllm-project/llm-compressor/blob/main/examples/quantizing_moe/deepseek_r1_example.py
        # Reference: https://github.com/vllm-project/llm-compressor/issues/1059
        ignore_layers = [
            "lm_head",
            "re:model\\.visual\\..*",   # All vision encoder sublayers (regex)
            "re:visual\\..*",           # Alternative prefix (regex)
            "re:model\\.merger\\..*",   # Vision-language merger sublayers (regex)
            "re:merger\\..*",           # Alternative prefix (regex)
            "model.visual",             # Exact match fallback
            "visual",                   # Exact match fallback
            "model.merger",             # Exact match fallback
            "merger",                   # Exact match fallback
        ]
        print(f"  Ignoring layers: {ignore_layers}")

        # Build recipe — try new API first, fall back to dict
        try:
            from llmcompressor.modifiers.quantization import QuantizationModifier
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=ignore_layers,
            )
        except ImportError:
            # v0.10+ uses dict/YAML recipe
            recipe = [{"QuantizationModifier": {
                "targets": "Linear",
                "scheme": "W4A16",
                "ignore": ignore_layers,
            }}]

        # Set trust_remote_code via env var (llmcompressor 0.10+ removed kwarg)
        os.environ["TRUST_REMOTE_CODE"] = "true"

        # Pre-load model with the correct class to avoid AutoModelForCausalLM error.
        # llm-compressor uses AutoModelForCausalLM internally which doesn't support
        # vision-language models like Qwen3VLForConditionalGeneration.
        model_class_name = detect_model_class(merged_fp16_dir)
        print(f"  Pre-loading model with class: {model_class_name}")

        # Read config.json to detect model type
        model_type = ""
        config_path = os.path.join(merged_fp16_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            model_type = model_config.get("model_type", "")
            arch_list = model_config.get("architectures", [])
            print(f"  Config model_type: {model_type}, architectures: {arch_list}")

        # Load the model with the correct class
        if "Qwen3VL" in model_class_name or "qwen3_vl" in str(model_type):
            from transformers import Qwen3VLForConditionalGeneration
            print("  Loading as Qwen3VLForConditionalGeneration...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                merged_fp16_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif "Qwen2_5_VL" in model_class_name or "qwen2_5_vl" in str(model_type):
            from transformers import Qwen2_5_VLForConditionalGeneration
            print("  Loading as Qwen2_5_VLForConditionalGeneration...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                merged_fp16_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForCausalLM
            print("  Loading as AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(
                merged_fp16_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

        # Pass the pre-loaded model object (not path) to oneshot
        try:
            oneshot(
                model=model,
                recipe=recipe,
                output_dir=output_dir,
            )
        except TypeError as e:
            # Some versions use save_dir instead of output_dir
            if "output_dir" in str(e) or "unexpected keyword" in str(e):
                print(f"  Retrying with save_dir kwarg...")
                oneshot(
                    model=model,
                    recipe=recipe,
                    save_dir=output_dir,
                )
            else:
                raise

        # DO NOT patch config.json quant_method.
        # llm-compressor saves in "compressed-tensors" format with its own
        # weight naming convention (e.g. weight_packed). vLLM 0.15+ supports
        # compressed-tensors natively. Patching to "awq" causes KeyError
        # because vLLM then looks for AWQ-format weight names that don't exist.
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            qc = config.get("quantization_config", {})
            print(f"  Quantization config: quant_method={qc.get('quant_method', 'N/A')}")
            print(f"  NOTE: Keeping compressed-tensors format (vLLM native support)")

        print(f"  AWQ quantization complete")

    except (ImportError, Exception) as e:
        print(f"\n  llm-compressor failed: {e}")
        # Method 2: AutoAWQ fallback
        print("  Trying AutoAWQ...")
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(merged_fp16_dir, trust_remote_code=True)
            model = AutoAWQForCausalLM.from_pretrained(
                merged_fp16_dir,
                device_map="auto",
                trust_remote_code=True,
            )
            model.quantize(
                tokenizer,
                quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4},
            )
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  AutoAWQ quantization complete")

        except ImportError:
            print("  ERROR: Neither llm-compressor nor AutoAWQ available!")
            print("  Install: pip install llmcompressor  OR  pip install autoawq")
            print("  Falling back to FP16 (will need more VRAM to serve)")
            # Just copy the FP16 model as-is
            import shutil
            if merged_fp16_dir != output_dir:
                shutil.copytree(merged_fp16_dir, output_dir, dirs_exist_ok=True)
            return output_dir

    # Copy tokenizer files if missing
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(merged_fp16_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)

    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith((".safetensors", ".bin"))
    )
    print(f"  Quantized model size: {total_size / 1e9:.1f} GB")

    return output_dir


def upload_to_hf(model_dir: str, repo_id: str, private: bool = True):
    """Upload quantized model to HuggingFace Hub."""
    print("\n" + "=" * 60)
    print(f"STEP 3: UPLOAD TO HUGGINGFACE")
    print(f"  Model: {model_dir}")
    print(f"  Repo: {repo_id}")
    print("=" * 60)

    from huggingface_hub import HfApi
    api = HfApi()

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model")

    print(f"  Uploaded to: https://huggingface.co/{repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter + AWQ quantize for vLLM serving"
    )
    parser.add_argument(
        "--adapter", "-a", default="",
        help="HuggingFace adapter ID (e.g., U4RASD/ar-ms-baseline). "
             "Required unless --skip_merge is used.",
    )
    parser.add_argument(
        "--base_model", "-b", default="",
        help="Base model ID (e.g., Qwen/Qwen3-VL-8B-Instruct). "
             "Required unless --skip_merge is used.",
    )
    parser.add_argument(
        "--output", "-o", default="./merged_awq",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--upload", "-u", default="",
        help="HuggingFace repo ID to upload (e.g., user/Qwen3-VL-8B-cheque-AWQ)",
    )
    parser.add_argument(
        "--skip_quantize", action="store_true",
        help="Skip AWQ quantization (serve FP16 directly)",
    )
    parser.add_argument(
        "--skip_merge", action="store_true",
        help="Skip merge step — quantize an already-merged FP16 model directly. "
             "Use --merged_dir to specify the FP16 model path.",
    )
    parser.add_argument(
        "--merged_dir", default="",
        help="Path to already-merged FP16 model (use with --skip_merge). "
             "Default: {output}_fp16",
    )
    parser.add_argument(
        "--private", action="store_true", default=True,
        help="Upload as private repo (default: True)",
    )

    args = parser.parse_args()

    # Validate required args when not skipping merge
    if not args.skip_merge:
        if not args.adapter:
            parser.error("--adapter is required (unless --skip_merge is used)")
        if not args.base_model:
            parser.error("--base_model is required (unless --skip_merge is used)")

    t_start = time.time()

    if args.skip_merge:
        # Skip merge — use existing FP16 model directly
        merged_dir = args.merged_dir or (args.output + "_fp16")
        if not os.path.exists(merged_dir):
            print(f"  ERROR: Merged FP16 dir not found: {merged_dir}")
            print(f"  Run without --skip_merge first, or specify --merged_dir")
            sys.exit(1)
        print(f"  Skipping merge. Using existing FP16 model: {merged_dir}")
    else:
        # Step 1: Merge
        merged_dir, processor = merge_adapter(args.adapter, args.base_model, args.output)

    # Step 2: Quantize (unless skipped)
    if args.skip_quantize:
        final_dir = merged_dir
        print(f"\n  Skipped quantization. FP16 model at: {merged_dir}")
    else:
        final_dir = quantize_awq(merged_dir, args.output)

    # Step 3: Upload (if requested)
    if args.upload:
        upload_to_hf(final_dir, args.upload, private=args.private)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed/60:.1f} min")
    print(f"  Model ready at: {final_dir}")
    if args.upload:
        print(f"  HuggingFace: https://huggingface.co/{args.upload}")
    print(f"\n  To serve with vLLM:")
    model_name = args.upload or final_dir
    print(f"    vllm serve {model_name} --dtype auto --trust-remote-code")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()