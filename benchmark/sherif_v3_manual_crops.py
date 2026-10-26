"""
Sherif v3 OCR — Re-run with manually cropped payee GT images.

Uses crops from benchmark/sherif_v3_arabic_outward/crops_gt/ instead of
auto-generated ROI crops. Same sherif v3 model on GPU, same scoring.
"""

import base64
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
GT_JSON = PROJECT_ROOT / "cheque-poc" / "test-data" / "ground_truth_v14.json"
CROPS_GT_DIR = PROJECT_ROOT / "benchmark" / "sherif_v3_arabic_outward" / "crops_gt"
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "sherif_v3_arabic_outward"

MODEL_NAME = "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3"
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def has_arabic(text):
    return bool(ARABIC_RE.search(text or ""))


def load_sherif_gpu():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoConfig
    print(f"Loading {MODEL_NAME} on GPU...")
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.tie_word_embeddings = True
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, trust_remote_code=True, config=config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    elapsed = (time.perf_counter() - start) * 1000
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  OK: {params:.0f}M params, {elapsed:.0f}ms, VRAM: {vram:.2f}GB")
    return processor, model


def run_sherif_inference(processor, model, pil_image, prompt):
    from qwen_vl_utils import process_vision_info
    start = time.perf_counter()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"data:image/png;base64,{img_b64}"},
            {"type": "text", "text": prompt},
        ],
    }]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100,
                                 return_dict_in_generate=True, output_scores=True)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_len:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    token_probs = []
    if outputs.scores:
        for step_idx, score_tensor in enumerate(outputs.scores):
            probs = torch.softmax(score_tensor[0], dim=-1)
            token_id = generated_ids[0, step_idx].item()
            token_prob = probs[token_id].item()
            token_str = processor.decode([token_id])
            token_probs.append({
                "token": token_str, "token_id": token_id,
                "prob": round(token_prob, 6),
                "log_prob": round(torch.log(probs[token_id]).item(), 6),
            })

    if token_probs:
        avg_lp = sum(tp["log_prob"] for tp in token_probs) / len(token_probs)
        seq_confidence = round(np.exp(avg_lp), 6)
    else:
        seq_confidence = 0.0

    latency = (time.perf_counter() - start) * 1000
    if text and "<" in text:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    clean = text.strip() if text and text.strip() else None
    return clean, latency, seq_confidence, token_probs


def normalize_name(name):
    if not name:
        return ""
    import unicodedata
    name = unicodedata.normalize("NFKC", name).lower().strip()
    name = re.sub(r'[.,()\[\]{}"\'/\\]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'[أإآٱ]', 'ا', name)
    name = re.sub(r'ؤ', 'و', name)
    name = re.sub(r'ئ', 'ي', name)
    name = re.sub(r'[\u064B-\u065F\u0670]', '', name)
    name = name.replace('ة', 'ه')
    return name.strip()


def fuzzy_name_match(extracted, ground_truth):
    if not extracted or not ground_truth:
        return False
    e, g = normalize_name(extracted), normalize_name(ground_truth)
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


def anls_score(pred, gt, tau=0.5):
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    p, g = normalize_name(pred), normalize_name(gt)
    if not p or not g:
        return 0.0
    m, n = len(p), len(g)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if p[i-1] == g[j-1] else 1 + min(dp[j], dp[j-1], prev)
            prev = temp
    nls = 1.0 - dp[n] / max(m, n)
    return nls if nls >= tau else 0.0


def main():
    print("=" * 70)
    print("Sherif v3 OCR — MANUAL CROPS Benchmark")
    print(f"Model: {MODEL_NAME}")
    print(f"Crops: {CROPS_GT_DIR}")
    print("=" * 70)

    # Load GT and match to manual crops
    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    cheques = []
    for entry in gt_data["cheques"]:
        rel_path = entry["image_path"]
        crop_name = rel_path.replace("/", "_")
        crop_path = CROPS_GT_DIR / crop_name
        if not crop_path.exists():
            continue
        payee = entry["fields"].get("payee_name", "")
        if not has_arabic(payee):
            continue  # skip English
        cheques.append({
            "crop_path": str(crop_path),
            "crop_name": crop_name,
            "gt_payee": payee,
            "category": entry["category"],
            "image_path": rel_path,
            "fields": entry["fields"],
        })

    print(f"\nMatched {len(cheques)} Arabic cheques with manual crops")
    outward = [c for c in cheques if c["category"] == "outward"]
    inward = [c for c in cheques if c["category"] == "inward"]
    print(f"  Outward: {len(outward)}, Inward: {len(inward)}")

    # Load model
    processor, model = load_sherif_gpu()

    # Prompts
    prompts = [
        ("default", "Read the handwritten name in this image. Return ONLY the name, nothing else."),
        ("arabic_focused", "اقرأ الاسم المكتوب بخط اليد في هذه الصورة. أعد الاسم فقط، لا شيء آخر."),
    ]

    # Also load previous auto-crop results for comparison
    prev_results = {}
    prev_files = sorted(OUTPUT_DIR.glob("sherif_v3_arabic_payee_*.json"))
    if prev_files:
        with open(prev_files[-1], "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        for r in prev_data.get("results", []):
            prev_results[r.get("image", "")] = r

    print(f"\n{'='*70}")
    print(f"Running sherif v3 on {len(cheques)} Arabic cheques (MANUAL CROPS)")
    print(f"{'='*70}")

    results = []
    for idx, cheque in enumerate(cheques):
        crop_path = cheque["crop_path"]
        crop_name = cheque["crop_name"]
        gt_payee = cheque["gt_payee"]
        category = cheque["category"]
        img_name = Path(cheque["image_path"]).name

        print(f"\n[{idx+1}/{len(cheques)}] {category.upper()} — {crop_name}")
        print(f"  GT: {gt_payee}")

        try:
            crop_img = Image.open(crop_path)
        except Exception as e:
            print(f"  CROP LOAD ERROR: {e}")
            results.append({"crop_name": crop_name, "image": img_name,
                            "category": category, "gt_payee": gt_payee, "error": str(e)})
            continue

        cheque_result = {
            "crop_name": crop_name,
            "image": img_name,
            "image_path": cheque["image_path"],
            "category": category,
            "gt_payee": gt_payee,
            "crop_path": crop_path,
            "crop_size": f"{crop_img.width}x{crop_img.height}",
            "prompt_results": {},
        }

        for prompt_name, prompt_text in prompts:
            print(f"  [{prompt_name}]", end=" ")
            try:
                text, latency, confidence, token_probs = run_sherif_inference(
                    processor, model, crop_img, prompt=prompt_text)
                match = fuzzy_name_match(text, gt_payee)
                anls = anls_score(text, gt_payee)

                # Compare with auto-crop result
                prev = prev_results.get(img_name, {}).get("prompt_results", {}).get(prompt_name, {})
                prev_match = prev.get("fuzzy_match", None)
                delta = ""
                if prev_match is not None:
                    if match and not prev_match:
                        delta = " << IMPROVED"
                    elif not match and prev_match:
                        delta = " << REGRESSED"

                print(f"{latency:.0f}ms | conf={confidence:.4f} | "
                      f"{'PASS' if match else 'FAIL'} | ANLS={anls:.3f} | "
                      f"{(text or 'EMPTY')[:60]}{delta}")

                cheque_result["prompt_results"][prompt_name] = {
                    "extracted": text,
                    "latency_ms": round(latency, 1),
                    "confidence": confidence,
                    "fuzzy_match": match,
                    "anls": round(anls, 4),
                    "token_count": len(token_probs),
                    "token_probs": token_probs,
                    "prev_auto_crop": {
                        "extracted": prev.get("extracted"),
                        "fuzzy_match": prev.get("fuzzy_match"),
                        "anls": prev.get("anls"),
                    } if prev else None,
                }
            except Exception as e:
                print(f"ERROR: {e}")
                cheque_result["prompt_results"][prompt_name] = {"error": str(e)}

        results.append(cheque_result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — MANUAL CROPS vs AUTO CROPS")
    print(f"{'='*70}")

    for prompt_name, _ in prompts:
        print(f"\n--- Prompt: {prompt_name} ---")
        for label, subset_filter in [("OUTWARD", "outward"), ("INWARD", "inward"), ("ALL", None)]:
            if subset_filter:
                valid = [r for r in results if r["category"] == subset_filter
                         and prompt_name in r.get("prompt_results", {})
                         and "error" not in r["prompt_results"][prompt_name]]
            else:
                valid = [r for r in results if prompt_name in r.get("prompt_results", {})
                         and "error" not in r["prompt_results"][prompt_name]]
            if not valid:
                continue

            correct = sum(1 for r in valid if r["prompt_results"][prompt_name]["fuzzy_match"])
            total = len(valid)
            avg_anls = np.mean([r["prompt_results"][prompt_name]["anls"] for r in valid])
            avg_lat = np.mean([r["prompt_results"][prompt_name]["latency_ms"] for r in valid])

            # Prev auto-crop stats
            prev_correct = sum(1 for r in valid
                               if r["prompt_results"][prompt_name].get("prev_auto_crop", {})
                               and r["prompt_results"][prompt_name]["prev_auto_crop"].get("fuzzy_match"))
            prev_anls_vals = [r["prompt_results"][prompt_name]["prev_auto_crop"]["anls"]
                              for r in valid
                              if r["prompt_results"][prompt_name].get("prev_auto_crop")
                              and r["prompt_results"][prompt_name]["prev_auto_crop"].get("anls") is not None]
            prev_avg_anls = np.mean(prev_anls_vals) if prev_anls_vals else 0

            print(f"  {label:8s} Manual: {correct}/{total} ({100*correct/total:.1f}%) ANLS={avg_anls:.3f} | "
                  f"Auto: {prev_correct}/{total} ({100*prev_correct/total:.1f}%) ANLS={prev_avg_anls:.3f} | "
                  f"lat={avg_lat:.0f}ms")

    # Per-cheque delta table
    print(f"\n--- Per-cheque changes (default prompt) ---")
    improved = []
    regressed = []
    for r in results:
        pr = r.get("prompt_results", {}).get("default", {})
        if "error" in pr:
            continue
        prev = pr.get("prev_auto_crop", {})
        if not prev:
            continue
        cur_match = pr["fuzzy_match"]
        prev_match = prev.get("fuzzy_match", False)
        cur_anls = pr["anls"]
        prev_anls = prev.get("anls", 0)
        if cur_match and not prev_match:
            improved.append(r)
            tag = "IMPROVED"
        elif not cur_match and prev_match:
            regressed.append(r)
            tag = "REGRESSED"
        elif cur_anls > prev_anls + 0.05:
            improved.append(r)
            tag = "anls++"
        elif cur_anls < prev_anls - 0.05:
            regressed.append(r)
            tag = "anls--"
        else:
            tag = "same"
        print(f"  {tag:10s} {r['category']:8s} {r['crop_name']:45s} "
              f"auto={prev_anls:.3f} manual={cur_anls:.3f} | {(pr.get('extracted','')or'')[:40]}")

    print(f"\n  Improved: {len(improved)}, Regressed: {len(regressed)}")

    # Save
    output_file = OUTPUT_DIR / f"sherif_v3_manual_crops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "config": {
            "model": MODEL_NAME,
            "device": "GPU",
            "gpu": torch.cuda.get_device_name(0),
            "timestamp": datetime.now().isoformat(),
            "crop_source": "manual_gt",
            "crop_dir": str(CROPS_GT_DIR),
            "total_cheques": len(cheques),
            "prompts": {k: v for k, v in prompts},
        },
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()
