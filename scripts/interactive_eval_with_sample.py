import argparse
import torch
import os
import sys
import time
import re
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
from datasets import load_dataset

def parse_coords(text):
    """Parses coordinates from text (e.g. <loc0500>, 0.5, or JSON box_2d)"""
    # Try parsing JSON first
    try:
        # Extract JSON part if mixed with text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, list) and len(data) > 0:
                # Look for box_2d in the first item
                if "box_2d" in data[0]:
                    coords = data[0]["box_2d"]
                    # Check if normalized or integer
                    if any(c > 1.0 for c in coords):
                        # Match token-based scale (usually 1024 for Gemma/PaliGemma)
                        return [c / 1024.0 for c in coords]
                    return coords
    except:
        pass

    loc_tokens = re.findall(r"<loc(\d{3,4})>", text)
    if loc_tokens:
        return [int(t) / 1024.0 for t in loc_tokens]
    floats = re.findall(r"0\.\d+", text)
    if floats:
        return [float(f) for f in floats]
    return []

def eval_screenspot(pred_text, gt_data):
    if not gt_data: return {"success": 0.0}
    gt_bbox = gt_data if isinstance(gt_data, list) and len(gt_data) == 4 else None
    if not gt_bbox: return {"success": 0.0}
    pred_coords = parse_coords(pred_text)
    pred_point = None
    if len(pred_coords) >= 2:
        y, x = pred_coords[0], pred_coords[1]
        pred_point = [x, y]
    elif len(pred_coords) >= 4:
        y1, x1, y2, x2 = pred_coords[:4]
        pred_point = [(x1 + x2) / 2, (y1 + y2) / 2]
    if not pred_point: return {"success": 0.0}
    px, py = pred_point
    xmin, ymin, xmax, ymax = gt_bbox
    if xmin <= px <= xmax and ymin <= py <= ymax:
        return {"success": 1.0}
    return {"success": 0.0}

def eval_sroie(pred_text, gt_text):
    def normalize(s):
        if not s: return ""
        s = re.sub(r"<[^>]+>", " ", s)
        return re.sub(r"\s+", " ", s.lower().strip())
    p = normalize(pred_text)
    g = normalize(gt_text)
    
    p_words = p.split()
    g_words = g.split()
    
    # WER Calculation (Levenshtein on words)
    def word_edit_distance(ref, hyp):
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        return dp[m][n]

    dist = word_edit_distance(g_words, p_words)
    wer = dist / len(g_words) if g_words else (0.0 if not p_words else 1.0)

    # F1 Calculation
    if not g_words: return {"wer": wer, "f1": 0.0}
    common = 0
    g_words_copy = list(g_words)
    for word in p_words:
        if word in g_words_copy:
            common += 1
            g_words_copy.remove(word)
    precision = common / len(p_words) if p_words else 0.0
    recall = common / len(g_words) if g_words else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"wer": wer, "f1": f1}

# Configuration for datasets
DATASET_CONFIGS = {
    "screenspot": {
        "id": "HongxinLi/ScreenSpot_v2",
        "split": "test",
        "prompt_fn": lambda ex: f"Detect the specific element described: {ex['instruction']} Output the bounding box coordinates for this element only.",
        "gt_fn": lambda ex: ex.get("bbox") or ex.get("point", "N/A"),
        "eval_fn": eval_screenspot
    },
    "sroie": {
        "id": "rajistics/sroie",
        "split": "train",
        "prompt_fn": lambda ex: "Extract the total amount, date, company name, and address. Output each value on a new line.",
        "gt_fn": lambda ex: ex.get("text", "N/A"),
        "eval_fn": eval_sroie
    }
}

def interactive_session(model_id, dataset_name, device, output_dir):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    dataset_id = config["id"]
    split = config["split"]

    print(f"Loading model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading dataset: {dataset_id} ({split} split)...")
    try:
        dataset = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: {e}. Retrying without trust_remote_code...")
        dataset = load_dataset(dataset_id, split=split)

    print(f"\nModel and Dataset loaded successfully! (Dataset size: {len(dataset)})")
    print(f"Interactive mode started. Results saved to: {output_dir}")

    max_idx = len(dataset) - 1
    while True:
        print("\n" + "="*50)
        user_input = input(f"Enter sample index to evaluate (0-{max_idx}, or 'q' to quit): ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        
        try:
            index = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
            continue

        if index < 0 or index >= len(dataset):
            print(f"Index out of bounds! Valid range: 0 - {len(dataset)-1}")
            continue

        print(f"\nProcessing sample [{index}]...")
        example = dataset[index]
        image = example["image"]
        prompt_text = config["prompt_fn"](example)
        ground_truth = config["gt_fn"](example)

        # 1. Save Image for Inspection
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, f"sample_{dataset_name}_{index}.png")
        image.save(image_path)
        print(f"[1] Input Image saved to: {image_path}")

        # 2. Prepare Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print(f"\n[2] Prompt:\n{'-'*20}\n{prompt}\n{'-'*20}")

        # 3. Inference
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,
                top_p=None,
                top_k=None
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
            
        input_len = inputs["input_ids"].shape[1]
        # SROIE requires clean text (no special tokens), while ScreenSpot needs <loc> tokens
        should_skip_special = (dataset_name == "sroie")
        generated_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=should_skip_special)

        # 4. Evaluation
        metrics = config["eval_fn"](generated_text, ground_truth)

        print(f"\n[3] Output (Inference time: {elapsed_time:.2f}s):\n{'-'*20}\n{generated_text}\n{'-'*20}")
        print(f"\n[4] Evaluation Results:\n{'-'*20}")
        print(f"Ground Truth: {ground_truth}")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v}")
        print(f"{'-'*20}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive evaluation of dataset samples with Gemma 3n")
    parser.add_argument("--model_id", type=str, default="models/gemma-3n", help="Hugging Face model ID or path")
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(DATASET_CONFIGS.keys()), help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_dir", type=str, default="results/interactive", help="Directory to save images")
    
    args = parser.parse_args()
    interactive_session(args.model_id, args.dataset_name, args.device, args.output_dir)
