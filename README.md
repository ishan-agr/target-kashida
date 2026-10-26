# Cheque VLM Fine-Tuning Journey

A technical write-up and companion code for an on-prem bank-cheque extraction system. The system is built
around a 4-bit Qwen3-VL-8B served with vLLM on a single 24 GB GPU, plus small specialist OCR models
("sidecars") for handwritten Arabic.

**Start here → [`TECHNICAL_WRITEUP.pdf`](TECHNICAL_WRITEUP.pdf)** (16 pages, 7 figures). 

## Layout

```
TECHNICAL_WRITEUP.pdf           article-formatted PDF (A4, figures, clickable outline)
TECHNICAL_WRITEUP.md            the long-form post (~5.7k words), source of the PDF

figures/                        7 SVG figures referenced by the write-up
  make_charts.py                5 matplotlib charts (values are formulas or validated benchmark numbers)
  diagrams/diagrams_svg.py      hand-laid system diagram
  diagrams/*.mmd                Mermaid serving flowchart (+ mermaid/puppeteer config)

pdf_build/                      md → HTML (pandoc + article.css) → PDF (headless Chrome)
  build_pdf.py, render_pdf.mjs, article.css, TECHNICAL_WRITEUP.html (self-contained web version)

pipeline/                       production inference pipeline (v7a)
  src/v7a/                      ingestion, layout (anchor-relative), 5 concurrent VLM passes,
                                fusion, deterministic MICR/IBAN, amount-words parser,
                                confidence, validation, audit, FastAPI
  src/preprocessing/            SWT/CCA payee-crop cleanup (classical CV)
  docs/                         pipeline blueprint + ADRs
  test_v7a.py                   unit tests

training/
  data_prep/                    GT → chat JSONL (3 prompt variants), Arabic dataset streaming,
                                Albumentations augmentation, data-mix builders (V14 ratio), label fix
  qlora_main_vlm/               Unsloth QLoRA (train_qlora.py) + plain HF/PEFT fallback (train_peft.py)
  quantization/                 merge LoRA → FP16 → W4A16 (llm-compressor), domain-calibrated AWQ,
                                AutoRound; VLM quantization debug guide
  sidecar_ft/                   Qwen2.5-VL-3B sidecar FT on Kaggle T4: payee-isolation data,
                                group-K-fold CV, leak-free overlay filter, eval scripts
  hatformer/                    full FT of a 334M TrOCR-style Arabic HTR model
  invizo/                       ONNX → PyTorch (onnx2torch) fine-tune + re-export (vendored; see header)
  layout_detector/              YOLOv8n zone detector (superseded by anchor-relative layout)
  active_learning/              SHA-256 frozen eval manifest

benchmark/                      83-cheque pipeline harness, MICR E-13B validator A/B, date normalizer A/B,
                                manual-crop sidecar benchmark, frontier reference benchmark

extensions/                     reference implementations for the domains the project did NOT execute
  tokenizer_fertility.py        Arabic token fertility + vision-token budget
  dedup_leakage.py              pHash + MinHash-LSH near-duplicate / leakage audit → GroupKFold ids
  distill_sidecar.py            token-level KD (teacher → ≤3 GB student) with prompt masking
  dpo_payee.py                  DPO with self-mined hard negatives (TRL)
  triton_w4a16_dequant.py       Triton INT4 group-dequant kernel + reference check
  mlflow_tracking.py            benchmark → MLflow (scorer hash, eval-set hash, git SHA, champion tag)
  distributed/                  DDP, FSDP+QLoRA, DeepSpeed ZeRO-3 configs + VLM-specific notes

research_notes_INTERNAL/        raw dated research logs (Feb–Jul 2026); see warning below
```

## Status labels

- Code under `pipeline/`, `training/` and `benchmark/` was **run** during the project.
- Code under `extensions/` was **written for the write-up and not executed on project data**. Each file
  header says so.


## Known caveats in the run code

- `training/qlora_main_vlm/train_peft.py` computes loss on prompt tokens too (only padding is masked).
  `extensions/distill_sidecar.py` shows the masked version.
- `setup.sh` and `kaggle_sherif_ft.py` use `get_device_properties(0).total_mem`. The attribute is
  `total_memory`.
- `train_qlora.py`: the `train()` signature defaults (r=128/α=128) differ from the CLI defaults (r=32/α=64).
  The CLI defaults are the ones that were used.

