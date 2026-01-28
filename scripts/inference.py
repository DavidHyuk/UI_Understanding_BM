import argparse
import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
from datasets import load_dataset

def run_benchmark(model_id, dataset_id, device, output_dir):
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, split="test")

    results = []
    print("Starting inference...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "benchmark_results.json")

    for i, example in enumerate(dataset):
        image = example["image"]
        instruction = example["instruction"]
        
        # 1. Apply Official Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Detect the element described: {instruction}"}
                ]
            }
        ]
        
        # Check if apply_chat_template works (requires tokenizer.chat_template to be set)
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            # Fallback for models without chat template in config
            print(f"Warning: apply_chat_template failed ({e}), using fallback manual prompt.")
            prompt = f"<image>\nDetect the element described: {instruction}"

        # 2. Preprocess
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # Match input dtype with model dtype (BF16) where applicable
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # 3. Inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False
            )
            
            # Extract only new tokens
            input_len = inputs["input_ids"].shape[1]
            generated_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        print(f"[{i}] Instruction: {instruction} -> Pred: {generated_text}")
        
        # Structure result
        results.append({
            "id": i,
            "instruction": instruction,
            "prediction": generated_text,
            "ground_truth": example.get("bbox") or example.get("point", "N/A")
        })

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Benchmark complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ScreenSpot V2 benchmark with Gemma 3n")
    parser.add_argument("--model_id", type=str, default="models/gemma-3n", help="Hugging Face model ID or path")
    parser.add_argument("--dataset_id", type=str, default="HongxinLi/ScreenSpot_v2", help="Hugging Face dataset ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    run_benchmark(args.model_id, args.dataset_id, args.device, args.output_dir)
