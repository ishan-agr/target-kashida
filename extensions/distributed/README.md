# Distributed training configs (reference; not executed in the project)

Every real run in this project was single-GPU: a Kaggle T4 16 GB for the 3B sidecar, and
one RunPod A100 80 GB for the 8B QLoRA runs (V14/V15). Scaling the same recipe to more
GPUs is mostly a question of where the 4-bit base weights live.

| Strategy | What is sharded | Works with a bnb-4bit QLoRA base? | When I would use it |
|---|---|---|---|
| DDP (`ddp.yaml`) | nothing; each rank holds a full replica | yes, simplest | Base fits on one card and you only want more images per second (the 8B NF4 base is about 6 GB) |
| FSDP + QLoRA (`fsdp_qlora.yaml`) | params, grads and optimizer state, including the 4-bit storage | yes, with `bnb_4bit_quant_storage=bfloat16` (the Answer.AI FSDP-QLoRA approach) | Bigger base (Qwen3.5-27B teacher for distillation) on 2 to 4 × 24 GB cards |
| DeepSpeed ZeRO-3 (`zero3.json`) | params, grads and optimizer state, plus optional CPU offload | yes, with the same quant-storage trick; offload trades speed for memory | Full fine-tune of the 3B sidecar on cheap cards |

## VLM-specific gotchas

- **Variable visual sequence length.** A 2000 px cheque costs about 4x the vision
  tokens of a 1000 px one (`../tokenizer_fertility.py`). Bucket batches by pixel area,
  otherwise the slowest rank sets the step time. V14 crashed at 16% with
  `max_seq_length=2048` for exactly this reason: image tokens alone overflowed it.
- **Vision tower under FSDP.** Wrap the ViT blocks (`Qwen3VLVisionBlock`) and the
  decoder layers as separate FSDP units. The merger is small, so leave it in the root
  unit. Freezing the tower with FSDP requires `use_orig_params=True`.
- **Gradient checkpointing.** Use `use_reentrant=False`. The reentrant variant plus
  frozen embeddings produces "does not require grad" errors with PEFT, which is why
  `train_peft.py` calls `enable_input_require_grads()`.
- **Eval.** Run generation-based CER evaluation on rank 0 only. `generate()` under
  FSDP needs `summon_full_params` or FSDP2, so it is easier to save the adapter and
  evaluate in a separate process.

Launch:

```bash
accelerate launch --config_file fsdp_qlora.yaml ../../training/qlora_main_vlm/train_peft.py \
    --data train.jsonl --output out_fsdp
```
