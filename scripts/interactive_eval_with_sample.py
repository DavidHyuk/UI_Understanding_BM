import argparse
import torch
import os
import sys
import time
import json
from PIL import Image

# Add the project root to sys.path to allow importing scripts.src.common.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.src.common.utils import DATASET_CONFIGS, load_model_and_processor, get_dataset, calculate_corpus_metrics

def interactive_session(model_id, dataset_name, device, output_dir):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]

    # Load resources using centralized utility
    model, processor = load_model_and_processor(model_id, device)
    
    # Load data using centralized utility
    dataset = get_dataset(dataset_name)

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

        # 2. Inference
        print(f"\n[2] Prompt:\n{'-'*20}\n{prompt_text}\n{'-'*20}")
        
        start_time = time.time()
        try:
            should_skip_special = (dataset_name == "sroie")
            generated_text = model.generate_content(
                prompt_text=prompt_text,
                image=image,
                max_new_tokens=512,
                skip_special_tokens=should_skip_special
            )
        except Exception as e:
            print(f"Inference error: {e}")
            generated_text = ""
            
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 4. Evaluation
        metrics = config["eval_fn"](generated_text, ground_truth)
        
        # Calculate additional corpus-level metrics if available (e.g. CIDEr, METEOR for captioning)
        if dataset_name == "widget_captioning":
            try:
                corpus_res = [{"prediction": generated_text, "ground_truth": ground_truth}]
                extra_metrics = calculate_corpus_metrics(corpus_res, dataset_name)
                metrics.update(extra_metrics)
            except Exception as e:
                print(f"Warning: Failed to calculate extra metrics: {e}")

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
