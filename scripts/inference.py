import argparse
import torch
import json
import os
import sys

# Add the project root to sys.path to allow importing demo.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.utils import DATASET_CONFIGS, load_model_and_processor, get_dataset

def run_benchmark(model_id, dataset_name, device, output_dir):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]

    # Load resources using centralized utility
    model, processor = load_model_and_processor(model_id, device)
    
    # Load data using centralized utility
    dataset = get_dataset(dataset_name)

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
        
        # Check if apply_chat_template works
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            prompt = f"<image>\n{prompt_text}"

        # 2. Preprocess
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
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
            should_skip_special = (dataset_name == "sroie")
            generated_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=should_skip_special)

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
