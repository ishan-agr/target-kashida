#!/bin/bash
# =============================================================================
# RunPod Environment Setup for Qwen3-VL-8B QLoRA Fine-Tuning
# =============================================================================
#
# Platform: RunPod A100 80GB ($1.19/hr) or A100 40GB ($0.79/hr)
# Minimum VRAM: 24GB (Unsloth QLoRA)
# Recommended VRAM: 40GB+ for batch_size > 1
#
# Usage: bash setup.sh
# =============================================================================

set -e

echo "=========================================="
echo "RunPod Setup: Qwen3-VL-8B QLoRA Training"
echo "=========================================="

# Check GPU
nvidia-smi || { echo "ERROR: No GPU found"; exit 1; }

# Install Unsloth (primary training framework)
echo "Installing Unsloth..."
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.15.0" peft accelerate bitsandbytes

# Install additional dependencies
echo "Installing dependencies..."
pip install datasets pillow transformers torchvision
pip install qwen-vl-utils  # Qwen3-VL image processing

# Install llm-compressor for AWQ quantization (post-training)
echo "Installing llm-compressor..."
pip install llmcompressor

# Install HuggingFace CLI for model upload
pip install huggingface_hub

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import unsloth; print(f'Unsloth version: {unsloth.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"
python -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)')
"

echo ""
echo "=========================================="
echo "Setup complete. Ready for training."
echo ""
echo "Next steps:"
echo "  1. Upload training data: training/data/merged_train.jsonl"
echo "  2. Run training: python train_qlora.py"
echo "  3. Merge adapter: python merge_adapter.py"
echo "  4. Quantize AWQ: python quantize_awq.py"
echo "  5. Upload model: python upload_model.py"
echo "=========================================="
