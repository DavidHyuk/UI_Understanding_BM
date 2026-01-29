import argparse
import torch
import json
import os
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
from datasets import load_dataset

def parse_coords(text):
    """Parses coordinates from text (e.g. <loc0500> or 0.5)"""
    # Pattern for special location tokens <loc0000>
    loc_tokens = re.findall(r"<loc(\d{3,4})>", text)
    if loc_tokens:
        # Assuming 1024 bins which is common for Paligemma/Gemma
        return [int(t) / 1024.0 for t in loc_tokens]
    
    # Pattern for floats
    floats = re.findall(r"0\.\d+", text)
    if floats:
        return [float(f) for f in floats]
    
    return []

def eval_screenspot(pred_text, gt_data):
    """
    Evaluates Success Rate for ScreenSpot.
    GT is a bounding box [x1, y1, x2, y2].
    Prediction should be a point inside the box.
    """
    if not gt_data:
        return {"success": 0.0}

    # ScreenSpot usually provides bbox
    gt_bbox = gt_data if isinstance(gt_data, list) and len(gt_data) == 4 else None
    
    if not gt_bbox:
        return {"success": 0.0}

    pred_coords = parse_coords(pred_text)
    pred_point = None

    # Assuming model outputs [y, x] or [y1, x1, y2, x2] (Gemma style)
    # Dataset is [x1, y1, x2, y2]
    if len(pred_coords) >= 2:
        # Use first point [y, x] -> convert to [x, y]
        y, x = pred_coords[0], pred_coords[1]
        pred_point = [x, y]
    elif len(pred_coords) >= 4:
        # Use center of box
        y1, x1, y2, x2 = pred_coords[:4]
        pred_point = [(x1 + x2) / 2, (y1 + y2) / 2]

    if not pred_point:
        return {"success": 0.0}

    # Check intersection
    px, py = pred_point
    xmin, ymin, xmax, ymax = gt_bbox
    
    # Success if point is inside bbox
    if xmin <= px <= xmax and ymin <= py <= ymax:
        return {"success": 1.0}
        
    return {"success": 0.0}

def eval_sroie(pred_text, gt_text):
    """Evaluates Accuracy and F1 for SROIE text extraction."""
    def normalize(s):
        if not s: return ""
        return re.sub(r"\s+", " ", s.lower().strip())

    p = normalize(pred_text)
    g = normalize(gt_text)

    # Accuracy (Exact Match)
    acc = 1.0 if p == g else 0.0

    # F1 Score
    p_words = p.split()
    g_words = g.split()
    
    if not g_words:
        return {"accuracy": acc, "f1": 0.0}

    common = 0
    g_words_copy = list(g_words)
    for word in p_words:
        if word in g_words_copy:
            common += 1
            g_words_copy.remove(word)

    precision = common / len(p_words) if p_words else 0.0
    recall = common / len(g_words) if g_words else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"accuracy": acc, "f1": f1}

DATASET_CONFIGS = {
    "screenspot": {
        "id": "HongxinLi/ScreenSpot_v2",
        "prompt_fn": lambda ex: f"Detect the element described: {ex['instruction']}",
        "gt_fn": lambda ex: ex.get("bbox") or ex.get("point", "N/A"),
        "instruction_key": "instruction",
        "eval_fn": eval_screenspot
    },
    "sroie": {
        "id": "rajistics/sroie",
        "split": "train",
        "prompt_fn": lambda ex: "Extract all text",
        "gt_fn": lambda ex: ex.get("text", "N/A"),
        "instruction_key": None,
        "eval_fn": eval_sroie
    }
}

def run_benchmark(model_id, dataset_name, device, output_dir):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    dataset_id = config["id"]
    # Default to 'test' if split not specified in config
    split = config.get("split", "test")

    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading dataset: {dataset_id} (split: {split})")
    try:
        dataset = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load with trust_remote_code=True: {e}")
        print("Retrying without trust_remote_code...")
        dataset = load_dataset(dataset_id, split=split)

    results = []
    metrics_accum = {}
    
    print(f"Starting inference on {dataset_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"benchmark_results_{dataset_name}.json")
    summary_file = os.path.join(output_dir, f"benchmark_summary_{dataset_name}.json")

    total_samples = len(dataset)
    for i, example in enumerate(dataset):
        image = example["image"]
        
        # Determine prompt text
        prompt_text = config["prompt_fn"](example)
        
        # For logging/results
        instruction_display = example[config["instruction_key"]] if config["instruction_key"] else "N/A"

        # 1. Apply Official Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Check if apply_chat_template works (requires tokenizer.chat_template to be set)
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            # Fallback for models without chat template in config
            # print(f"Warning: apply_chat_template failed ({e}), using fallback manual prompt.")
            prompt = f"<image>\n{prompt_text}"

        # 2. Preprocess
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # Match input dtype with model dtype (BF16) where applicable
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # 3. Inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                top_p=None,
                top_k=None
            )
            
            # Extract only new tokens
            input_len = inputs["input_ids"].shape[1]
            # Don't skip special tokens to catch <loc> tags
            generated_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=False)

        print(f"[{i+1}/{total_samples}] Prompt: {prompt_text} -> Pred: {generated_text}")
        
        # Calculate Metrics
        ground_truth = config["gt_fn"](example)
        metrics = config["eval_fn"](generated_text, ground_truth)
        
        # Accumulate metrics
        for k, v in metrics.items():
            if k not in metrics_accum:
                metrics_accum[k] = []
            metrics_accum[k].append(v)

        # Structure result
        results.append({
            "id": i,
            "instruction": instruction_display,
            "prompt": prompt_text,
            "prediction": generated_text,
            "ground_truth": ground_truth,
            "metrics": metrics
        })

    # Calculate average metrics
    summary = {k: sum(v)/len(v) for k, v in metrics_accum.items() if len(v) > 0}
    summary["num_samples"] = len(dataset)
    
    print(f"Summary Metrics: {summary}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Benchmark complete. Results saved to {output_file}")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark with Gemma 3n")
    parser.add_argument("--model_id", type=str, default="models/gemma-3n", help="Hugging Face model ID or path")
    parser.add_argument("--dataset_name", type=str, default="screenspot", choices=list(DATASET_CONFIGS.keys()), help="Dataset to evaluate on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    run_benchmark(args.model_id, args.dataset_name, args.device, args.output_dir)
