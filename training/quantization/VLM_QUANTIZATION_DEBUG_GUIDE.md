# VLM AWQ Quantization Debug Guide

> **Context**: Quantizing merged Qwen3-VL-8B + LoRA adapter for vLLM serving
> **Tool**: llm-compressor 0.10+ (vLLM recommended)
> **Date**: March 2026

---

## Error 1: `AutoModelForCausalLM` doesn't recognize VL models

```
ValueError: Unrecognized configuration class <class 'Qwen3VLConfig'>
for this kind of AutoModel: AutoModelForCausalLM.
```

### Root Cause

llm-compressor's `oneshot()` internally calls `AutoModelForCausalLM.from_pretrained()` when given a path string. Vision-Language models (Qwen3-VL, Qwen2.5-VL, LLaVA, etc.) use `*ForConditionalGeneration` classes, not `*ForCausalLM`.

### Fix

Pre-load the model with the correct class and pass the model object:

```python
from transformers import Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    merged_fp16_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

oneshot(
    model=model,  # Pass object, not path
    recipe=recipe,
    output_dir=output_dir,
)
```

### References

- llm-compressor source: `llmcompressor/entrypoints/utils.py:175` (`initialize_model_from_path`)
- transformers `AutoModelForCausalLM` supported configs: [auto_factory.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/auto_factory.py)
- GitHub Issue: [vllm-project/llm-compressor#1059](https://github.com/vllm-project/llm-compressor/issues/1059) — VLM quantization support

---

## Error 2: Vision encoder layers not divisible by group_size

```
ValueError: The following layers have weight column counts not divisible by
group_size. columns % group_size == 0
  - model.visual.blocks.0.mlp.linear_fc2 (columns=4304, group_size=128)
  ... (27 layers)
```

### Root Cause

Qwen3-VL's ViT encoder has `intermediate_size=4304` in its MLP layers. AWQ W4A16 quantization uses `group_size=128` by default. `4304 % 128 = 80 != 0`, so quantization fails.

This is a **known constraint** of group quantization: weight dimensions must be divisible by the group size. The fix is NOT to change the group size — it's to **skip the vision encoder entirely**.

### Why skipping the vision encoder is correct

1. **Standard practice**: All production VLM quantization pipelines only quantize the LLM backbone and keep the ViT in FP16. The vision encoder is typically 10-15% of total params.

2. **cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit** (the model you're already running) was quantized this way — ViT is FP16, LLM backbone is 4-bit AWQ.

3. **Accuracy**: Quantizing the vision encoder degrades visual feature extraction quality. For document/cheque processing where OCR accuracy matters, keeping ViT in FP16 is critical.

4. **VRAM impact**: Qwen3-VL-8B ViT is ~500M params. At FP16 that's ~1GB. The LLM backbone at 4-bit is ~4GB. Total ~5GB vs ~5GB all-4bit — negligible difference.

### Fix

Add vision encoder layers to the `ignore` list using **regex patterns** (`re:` prefix).

**CRITICAL**: llm-compressor v0.10.0 uses **exact name matching** by default, NOT prefix matching.
`"model.visual"` will NOT match `"model.visual.blocks.0.mlp.linear_fc2"`.
You MUST use `"re:model\\.visual\\..*"` for prefix-style matching.

```python
ignore_layers = [
    "lm_head",                    # Output projection (standard, exact match)
    "re:model\\.visual\\..*",     # All vision encoder sublayers (regex)
    "re:visual\\..*",             # Alternative prefix (regex)
    "re:model\\.merger\\..*",     # Vision-language merger sublayers (regex)
    "re:merger\\..*",             # Alternative prefix (regex)
    "model.visual",               # Exact match fallback
    "visual",                     # Exact match fallback
    "model.merger",               # Exact match fallback
    "merger",                     # Exact match fallback
]

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=ignore_layers,
)
```

**Why both regex and exact?** The regex patterns (`re:...`) match all sublayer paths like
`model.visual.blocks.0.mlp.linear_fc2`. The exact names (`model.visual`) catch the parent
module itself. Including both ensures coverage regardless of how llm-compressor resolves
module names internally.

### Production References

**cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit quantization config:**
```json
{
  "quantization_config": {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": true,
    "modules_to_not_convert": ["visual", "merger"]
  }
}
```

**vLLM documentation on VLM quantization:**
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/auto_awq.html): "For multimodal models, only the language model layers should be quantized."
- [llm-compressor VLM example](https://github.com/vllm-project/llm-compressor/tree/main/examples): All VLM examples ignore vision encoder modules.

**Alternative: `bypass_divisibility_checks=True`**
llm-compressor offers this flag, but it's NOT recommended — it can produce corrupt weight matrices that crash at inference. Always prefer ignoring non-divisible layers.

---

## Error 3: `trust_remote_code` kwarg rejected

```
ValueError: Some keys are not used by the HfArgumentParser: ['trust_remote_code']
```

### Root Cause

llm-compressor v0.10+ uses `HfArgumentParser` internally which doesn't accept `trust_remote_code` as a keyword argument. Older versions (0.5.x) accepted it directly.

### Fix

Set via environment variable:

```python
os.environ["TRUST_REMOTE_CODE"] = "true"
```

### References

- llm-compressor v0.10.0 changelog: `trust_remote_code` removed from oneshot kwargs
- HuggingFace `HfArgumentParser`: Only accepts dataclass fields, not arbitrary kwargs

---

## Error 4: AutoAWQ `PytorchGELUTanh` ImportError

```
ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'
```

### Root Cause

AutoAWQ is deprecated and incompatible with newer transformers versions (4.51+). The `PytorchGELUTanh` class was renamed/removed.

### Fix

```bash
pip uninstall autoawq autoawq-kernels -y
```

Use llm-compressor instead (vLLM's recommended quantization tool).

---

## Error 5: AutoProcessor TypeError (missing torchvision)

```
TypeError: argument of type 'NoneType' is not iterable
```

### Root Cause

`AutoProcessor.from_pretrained()` tries to load a `VideoProcessor` which requires `torchvision`. If torchvision is missing, the video processor lookup returns `None`, causing the error.

### Fix

```bash
pip install torchvision
```

Or add a fallback:

```python
try:
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
except TypeError:
    from transformers import AutoTokenizer
    processor = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
```

---

## Complete Working Quantization Pipeline

Here's the validated end-to-end command sequence:

```bash
# 1. Install dependencies
pip install torch transformers peft accelerate torchvision
pip install llmcompressor>=0.10.0
pip install huggingface_hub sentencepiece protobuf
pip uninstall autoawq autoawq-kernels -y  # Remove deprecated package

# 2. Set environment
export TRUST_REMOTE_CODE=true
export TOKENIZERS_PARALLELISM=false

# 3. Run merge + quantize
python merge_for_vllm.py \
    --adapter U4RASD/ar-ms-baseline \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --output ./u4rasd_merged_awq \
    --upload cyankiwi/Qwen3-VL-8B-ar-ms-AWQ
```

### Expected output flow

```
STEP 1: MERGE ADAPTER
  Base: Qwen/Qwen3-VL-8B-Instruct
  Adapter: U4RASD/ar-ms-baseline
  Model class: Qwen3VLForConditionalGeneration
  Loading base model (FP16)...
  Loaded in ~120s. Params: 8,767,123,696
  Loading adapter: U4RASD/ar-ms-baseline
  Adapter config: r=32, alpha=64
  Merging adapter into base weights...
  Merge complete
  Saving merged FP16 to: ./u4rasd_merged_awq_fp16
  FP16 model size: ~16.3 GB

STEP 2: AWQ QUANTIZE
  Pre-loading model with Qwen3VLForConditionalGeneration
  Ignoring layers: [lm_head, model.visual, visual, model.merger, merger]
  Quantizing language model backbone (W4A16, group_size=128)
  ... ~20-40 minutes on A100/A10 ...
  Quantized model size: ~5.2 GB

STEP 3: UPLOAD TO HUGGINGFACE
  Uploaded to: https://huggingface.co/cyankiwi/Qwen3-VL-8B-ar-ms-AWQ
```

### Serving the quantized model

```bash
# Option 1: Docker compose
MODEL=cyankiwi/Qwen3-VL-8B-ar-ms-AWQ docker compose up

# Option 2: Direct vLLM serve
vllm serve cyankiwi/Qwen3-VL-8B-ar-ms-AWQ \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768
```

---

## Production Case Studies

### Case Study 1: cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit

The model currently running in production. Quantized from `Qwen/Qwen3-VL-8B-Instruct`:

- **Method**: AWQ 4-bit, group_size=128
- **Vision encoder**: Kept in FP16 (not quantized)
- **Language model**: W4A16 quantized
- **VRAM**: ~21GB on A10 24GB (with vLLM overhead)
- **Serving**: vLLM with `enforce_eager=true`, `enable_chunked_prefill=true`
- **Known limitation**: Arabic text degraded at 4-bit (56x CER increase per QARI-OCR paper)

### Case Study 2: Qwen2.5-VL-7B AWQ (vLLM official example)

vLLM's official example for quantizing Qwen2.5-VL:
- Source: [vLLM examples/quantization](https://github.com/vllm-project/vllm/tree/main/examples)
- **Key config**: `modules_to_not_convert: ["visual", "merger", "lm_head"]`
- **Recipe**: W4A16 via llm-compressor
- **Result**: ~4.5GB model, serves on 8GB GPU

### Case Study 3: QARI-OCR Arabic quantization study

From arXiv:2506.02295 (June 2025):
- Tested Qwen2-VL-2B at FP16, 8-bit, 4-bit on Arabic OCR
- **FP16**: CER 0.061, WER 0.160
- **8-bit**: CER 0.091, WER 0.255 (49% CER increase)
- **4-bit**: CER 3.452, WER 4.516 (56x CER increase)
- **Conclusion**: 4-bit quantization is catastrophic for Arabic. 8-bit is acceptable.
- **Implication**: Our merged adapter model will still struggle with Arabic at AWQ 4-bit. The adapter improves Arabic capabilities at FP16, but quantization destroys it.

---

## Technical References

### llm-compressor

| Resource | URL |
|----------|-----|
| GitHub repo | https://github.com/vllm-project/llm-compressor |
| PyPI | https://pypi.org/project/llmcompressor/ |
| v0.10.0 release notes | https://github.com/vllm-project/llm-compressor/releases/tag/v0.10.0 |
| VLM quantization issues | https://github.com/vllm-project/llm-compressor/issues?q=visual+encoder |
| Oneshot API docs | https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/oneshot.py |
| W4A16 scheme docs | https://github.com/vllm-project/llm-compressor/blob/main/docs/schemes.md |

### vLLM quantization

| Resource | URL |
|----------|-----|
| vLLM quantization guide | https://docs.vllm.ai/en/latest/quantization/ |
| AWQ support in vLLM | https://docs.vllm.ai/en/latest/quantization/auto_awq.html |
| compressed-tensors format | https://github.com/neuralmagic/compressed-tensors |
| vLLM VLM support | https://docs.vllm.ai/en/latest/models/vlm.html |

### Qwen VL models

| Resource | URL |
|----------|-----|
| Qwen3-VL technical report | https://arxiv.org/abs/2504.XXXXX |
| Qwen2.5-VL technical report | https://arxiv.org/abs/2502.13923 |
| Qwen3-VL HuggingFace | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct |
| Qwen3-VL config (vision encoder) | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json |
| DeepStack architecture | https://deepwiki.com/QwenLM/Qwen3-VL/4.2-model-architecture |

### AWQ quantization

| Resource | URL |
|----------|-----|
| AWQ paper (Lin et al., MLSys 2024) | https://arxiv.org/abs/2306.00978 |
| AWQ GitHub | https://github.com/mit-han-lab/llm-awq |
| W4A16 explanation | Weight 4-bit, Activation 16-bit. Only weights are quantized. |
| Group quantization | Weights grouped into blocks of `group_size` (default 128), each group gets its own scale/zero-point |

### Arabic OCR quantization research

| Resource | URL |
|----------|-----|
| QARI-OCR paper | https://arxiv.org/abs/2506.02295 |
| KITAB-Bench (Arabic OCR benchmark) | https://arxiv.org/html/2502.14949v2 |
| Don't merge LoRA into 4-bit LLM | https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997 |

---

## Debugging Checklist

When quantization fails, check in this order:

1. **Model class**: Is llm-compressor loading with the right class? VLMs need `*ForConditionalGeneration`, not `AutoModelForCausalLM`. Fix: Pre-load and pass model object.

2. **Ignore list**: Are vision encoder layers excluded? Check `model.visual.*` and `model.merger.*`. Fix: Add to recipe `ignore` list.

3. **Group size divisibility**: All quantized layers must have dimensions divisible by `group_size` (default 128). Fix: Ignore non-divisible layers.

4. **trust_remote_code**: Set via `os.environ["TRUST_REMOTE_CODE"] = "true"`, not as a kwarg.

5. **API version**: llm-compressor 0.10+ has different import paths than 0.5.x. Check `from llmcompressor import oneshot` vs `from llmcompressor.transformers import oneshot`.

6. **VRAM**: AWQ quantization needs the full FP16 model in memory + quantization buffers. A100 80GB recommended. A10 24GB may OOM on 8B models.

7. **tokenizer**: Some models have tokenizer regex warnings (Mistral regex pattern). These are non-fatal warnings, ignore them.

8. **config.json patching**: After quantization, check if `quant_method` is set correctly for vLLM. If it says `compressed-tensors`, patch to `awq` for older vLLM versions.
