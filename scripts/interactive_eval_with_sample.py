import argparse
import torch
import os
import sys
import time
import json
from PIL import Image

# Add the project root to sys.path to allow importing demo.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.utils import DATASET_CONFIGS, load_model_and_processor, get_dataset

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
