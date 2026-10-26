import argparse, json, os, sys
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from PIL import Image
from functools import partial

def collate_fn(examples, processor):
      """Custom collator that tokenizes multimodal messages using the processor."""
      texts = []
      images_list = []
      for ex in examples:
          msgs = ex["messages"]
          # Convert array-style content back to standard format for chat template
          clean_msgs = []
          for msg in msgs:
              content = msg["content"]
              if isinstance(content, list):
                  # Check for images
                  has_image = False
                  text_parts = []
                  for item in content:
                      if item.get("type") == "image":
                          has_image = True
                          img_path = item["image"].replace("file://", "")
                          try:
                              img = Image.open(img_path).convert("RGB")
                              images_list.append(img)
                          except Exception as e:
                              print(f"WARNING: Failed to load image {img_path}: {e}")
                              # Use a small dummy image
                              images_list.append(Image.new("RGB", (64, 64)))
                      elif item.get("type") == "text":
                          text_parts.append(item["text"])
                  if has_image:
                      clean_msgs.append({
                          "role": msg["role"],
                          "content": [{"type": "image"}] + [{"type": "text", "text": t} for t in text_parts]
                      })
                  else:
                      clean_msgs.append({"role": msg["role"], "content": "\n".join(text_parts)})
              else:
                  clean_msgs.append(msg)

          text = processor.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=False)
          texts.append(text)

      batch = processor(
          text=texts,
          images=images_list if images_list else None,
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=2048,
      )
      # Labels = input_ids (causal LM), mask padding with -100
      labels = batch["input_ids"].clone()
      labels[labels == processor.tokenizer.pad_token_id] = -100
      batch["labels"] = labels
      return batch


def train(data_path, output_dir, warm_start="", epochs=3, lr=2e-4):
      print("=" * 60)
      print("Qwen3-VL-8B QLoRA Fine-Tuning (PEFT + Trainer)")
      print("=" * 60)

      bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16,
          bnb_4bit_use_double_quant=True,
      )

      base_id = "Qwen/Qwen3-VL-8B-Instruct"
      print(f"\n[1/5] Loading {base_id} in 4-bit...")
      model = AutoModelForImageTextToText.from_pretrained(
          base_id,
          quantization_config=bnb_config,
          torch_dtype=torch.bfloat16,
          device_map="auto",
          trust_remote_code=True,
      )
      processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
      if processor.tokenizer.pad_token is None:
          processor.tokenizer.pad_token = processor.tokenizer.eos_token
      print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

      if warm_start:
          print(f"\n[2/5] Loading warm-start adapter: {warm_start}")
          model = PeftModel.from_pretrained(model, warm_start, is_trainable=True)
          for n, p in model.named_parameters():
              if "lora" in n.lower():
                  p.requires_grad = True
          # Required for gradient checkpointing with quantized models
          model.enable_input_require_grads()
      else:
          print(f"\n[2/5] Applying fresh LoRA (r=128)...")
          lora_config = LoraConfig(
              r=128, lora_alpha=128, lora_dropout=0, bias="none",
              target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
              task_type="CAUSAL_LM",
          )
          model = get_peft_model(model, lora_config)

      # Required for gradient checkpointing with quantized models
      model.enable_input_require_grads()

      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
      total = sum(p.numel() for p in model.parameters())
      print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

      print(f"\n[3/5] Loading data...")
      dataset = load_dataset("json", data_files=data_path, split="train")
      print(f"  {len(dataset)} samples")

      print(f"\n[4/5] Configuring training...")
      args = TrainingArguments(
          output_dir=output_dir,
          per_device_train_batch_size=1,
          gradient_accumulation_steps=8,
          num_train_epochs=epochs,
          learning_rate=lr,
          lr_scheduler_type="cosine",
          warmup_ratio=0.1,
          bf16=True,
          logging_steps=5,
          save_strategy="epoch",
          save_total_limit=2,
          optim="adamw_8bit",
          weight_decay=0.01,
          max_grad_norm=1.0,
          gradient_checkpointing=True,
          gradient_checkpointing_kwargs={"use_reentrant": False},
          report_to="none",
          remove_unused_columns=False,
          dataloader_pin_memory=False,
      )

      print(f"\n[5/5] Training...")
      trainer = Trainer(
          model=model,
          args=args,
          train_dataset=dataset,
          data_collator=partial(collate_fn, processor=processor),
      )
      stats = trainer.train()

      print(f"\nDone! Loss: {stats.training_loss:.4f}, Steps: {stats.global_step}")
      adapter_dir = os.path.join(output_dir, "adapter")
      model.save_pretrained(adapter_dir)
      processor.save_pretrained(adapter_dir)
      print(f"Adapter saved to: {adapter_dir}")

if __name__ == "__main__":
      p = argparse.ArgumentParser()
      p.add_argument("--data", required=True)
      p.add_argument("--output", default="./cheque-qlora")
      p.add_argument("--warm_start", default="")
      p.add_argument("--epochs", type=int, default=3)
      p.add_argument("--lr", type=float, default=2e-4)
      a = p.parse_args()
      train(a.data, a.output, a.warm_start, a.epochs, a.lr)