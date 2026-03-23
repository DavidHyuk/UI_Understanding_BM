import argparse
import torch
import torch.distributed as dist
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import json
import os
import sys
import time

# Add the project root to sys.path to allow importing scripts.src.common.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.src.common.utils import DATASET_CONFIGS, load_model_and_processor, get_dataset

def run_benchmark(model_id, dataset_name, device, output_base_dir, use_ddp=False, num_samples=None):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    rank = 0
    world_size = 1
    if use_ddp:
        if not dist.is_initialized():
            # Increase timeout to 30 minutes for long inference tasks
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

    # Generate timestamp and output_dir, syncing across ranks in DDP
    if use_ddp:
        # We need to broadcast the timestamp from rank 0
        timestamp_tensor = torch.zeros(15, dtype=torch.long, device=device) # YYYYMMDD_HHMMSS is 15 chars
        ts_str = ""
        if rank == 0:
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            for i, c in enumerate(ts_str):
                timestamp_tensor[i] = ord(c)
        dist.broadcast(timestamp_tensor, src=0)
        if rank != 0:
            ts_str = "".join([chr(int(c.item())) for c in timestamp_tensor])
        timestamp = ts_str
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    output_dir = os.path.join(output_base_dir, dataset_name, timestamp)

    # Generate timestamp and output_dir, syncing across ranks in DDP
    if use_ddp:
        # We need to broadcast the tim`estamp from rank 0
        timestamp_tensor = torch.zeros(15, dtype=torch.long, device=device) # YYYYMMDD_HHMMSS is 15 chars
        ts_str = ""
        if rank == 0:
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            for i, c in enumerate(ts_str):
                timestamp_tensor[i] = ord(c)
        dist.broadcast(timestamp_tensor, src=0)
        if rank != 0:
            ts_str = "".join([chr(int(c.item())) for c in timestamp_tensor])
        timestamp = ts_str
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    output_dir = os.path.join(output_base_dir, dataset_name, timestamp)

    config = DATASET_CONFIGS[dataset_name]

    # Load data using centralized utility FIRST to avoid datasets/PyTorch deadlocks in DDP
    dataset = get_dataset(dataset_name)

    # In DDP, ensure only rank 0 downloads the model first to avoid HF cache conflicts
    model = None
    processor = None
    if use_ddp:
        if rank == 0:
            model, processor = load_model_and_processor(model_id, device)
        dist.barrier()
        if rank != 0:
            time.sleep(rank * 3) # Stagger loading to prevent HF hub concurrent cache access/API rate limits
            model, processor = load_model_and_processor(model_id, device)
    else:
        model, processor = load_model_and_processor(model_id, device)
    
    # Limit samples if requested
    if num_samples is not None and num_samples > 0:
        total_available = len(dataset)
        num_samples = min(num_samples, total_available)
        if rank == 0:
            print(f"Limiting evaluation to first {num_samples} samples (Total: {total_available})")
        dataset = dataset.select(range(num_samples))

    # Shard dataset if using DDP
    if use_ddp:
        # We manually shard to avoid complexity of DataLoader for simple inference
        total_size = len(dataset)
        indices = list(range(total_size))
        # Simple sharding
        rank_indices = indices[rank::world_size]
        # In datasets, we can use select
        dataset = dataset.select(rank_indices)
        if rank == 0:
            print(f"DDP Mode: World Size {world_size}, Sharding {total_size} samples...")

    results = []
    metrics_accum = {}
    
    if rank == 0:
        print(f"Starting inference on {dataset_name}...")
    
    # Create output directory
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Barrier to ensure dir exists
    if use_ddp:
        if rank == 0: print(f"DEBUG: Rank {rank} entering pre-loop barrier")
        dist.barrier()
        if rank == 0: print(f"DEBUG: Rank {rank} passed pre-loop barrier")

    output_file = os.path.join(output_dir, f"benchmark_results_rank{rank}.json")
    
    total_samples = len(dataset)
    for i, example in enumerate(dataset):
        image = example["image"]
        
        # Apply transformation if defined (e.g. drawing bbox for widget captioning)
        if config.get("transform_fn"):
            image = config["transform_fn"](example)
            
        # Determine prompt text
        prompt_text = config["prompt_fn"](example)
        
        # For logging/results
        instruction_display = example[config["instruction_key"]] if config["instruction_key"] else "N/A"

        # 1. Inference using Unified Interface
        try:
            should_skip_special = (dataset_name == "sroie")
            # For SROIE, use a repetition penalty to avoid infinite generation loops
            penalty = 1.1 if dataset_name == "sroie" else 1.0
            
            start_time = time.time()
            generated_text = model.generate_content(
                prompt_text=prompt_text,
                image=image,
                max_new_tokens=256,
                skip_special_tokens=should_skip_special,
                repetition_penalty=penalty
            )
            elapsed_time = time.time() - start_time
            # Approx token count (4 chars per token)
            approx_tokens = len(generated_text) / 4
            speed = approx_tokens / elapsed_time if elapsed_time > 0 else 0
            
        except Exception as e:
            print(f"Error during inference sample {i}: {e}")
            generated_text = ""
            elapsed_time = 0
            approx_tokens = 0
            speed = 0

        print(f"[{i+1}/{total_samples}] Time: {elapsed_time:.2f}s ({approx_tokens:.1f} tokens, {speed:.1f} t/s) | Prompt: {prompt_text} -> Pred: {generated_text}")
        
        # Calculate Metrics
        ground_truth = config["gt_fn"](example)
        # Pass image size if available, as needed for coordinate normalization
        image_size = image.size if hasattr(image, "size") else None
        metrics = config["eval_fn"](generated_text, ground_truth, image_size=image_size)
        
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

    # Save rank-specific results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    if use_ddp:
        print(f"DEBUG: Rank {rank} entering post-loop barrier")
        dist.barrier()
        print(f"DEBUG: Rank {rank} passed post-loop barrier")
    
    # Rank 0 gathers and aggregates
    if rank == 0:
        all_results = []
        final_metrics_accum = {}
        
        # Collect from all ranks
        for r in range(world_size):
            rank_file = os.path.join(output_dir, f"benchmark_results_rank{r}.json")
            # Wait a bit for file writing to finish if needed
            for _ in range(5):
                if os.path.exists(rank_file): break
                time.sleep(1)
            
            with open(rank_file, "r") as f:
                rank_data = json.load(f)
                all_results.extend(rank_data)
                for item in rank_data:
                    for k, v in item["metrics"].items():
                        if k not in final_metrics_accum:
                            final_metrics_accum[k] = []
                        final_metrics_accum[k].append(v)
            
            # Clean up rank-specific file
            if r > 0: os.remove(rank_file)
            else:
                # Rename rank 0 file to final name
                final_output_file = os.path.join(output_dir, f"benchmark_results_{dataset_name}_{timestamp}.json")
                os.rename(rank_file, final_output_file)

        # Calculate average metrics
        summary = {k: sum(v)/len(v) for k, v in final_metrics_accum.items() if len(v) > 0}
        
        # Calculate corpus-level metrics (e.g. CIDEr, METEOR)
        try:
            from scripts.src.common.utils import calculate_corpus_metrics
            corpus_metrics = calculate_corpus_metrics(all_results, dataset_name)
            summary.update(corpus_metrics)
        except Exception as e:
            print(f"Warning: Failed to calculate corpus metrics: {e}")

        summary["num_samples"] = len(all_results)
        
        summary_file = os.path.join(output_dir, f"benchmark_summary_{dataset_name}_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # TensorBoard logging
        try:
            writer = SummaryWriter(log_dir=output_dir)
            # Log metrics
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    # Using a fixed step 0 as this is a single evaluation run.
                    # TensorBoard will group by dataset_name automatically.
                    writer.add_scalar(f"{dataset_name}/{k}", v, 0)
            
            # Log a few sample results as text for quick inspection
            sample_size = min(5, len(all_results))
            for i in range(sample_size):
                res = all_results[i]
                text_content = f"**Prompt:** {res['prompt']}  \n" \
                               f"**Pred:** {res['prediction']}  \n" \
                               f"**GT:** {res['ground_truth']}"
                writer.add_text(f"Samples/{dataset_name}/{i}", text_content, 0)
                
            writer.close()
            print(f"TensorBoard logs saved to {output_dir}")
        except Exception as e:
            print(f"Warning: Failed to log to TensorBoard: {e}")
            
        print(f"\nBenchmark complete on {world_size} GPUs.")
        print(f"Summary Metrics: {summary}")
        print(f"Full results saved to {os.path.join(output_dir, f'benchmark_results_{dataset_name}_{timestamp}.json')}")
        print(f"Summary saved to {summary_file}")
    
    if use_ddp:
        print(f"DEBUG: Rank {rank} destroying process group")
        dist.destroy_process_group()
        print(f"DEBUG: Rank {rank} process group destroyed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark with Gemma 3n")
    parser.add_argument("--model_id", type=str, default="models/gemma-3n", help="Hugging Face model ID or path")
    parser.add_argument("--dataset_name", type=str, default="screenspot", choices=list(DATASET_CONFIGS.keys()), help="Dataset to evaluate on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel for inference")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    run_benchmark(args.model_id, args.dataset_name, args.device, args.output_dir, use_ddp=args.ddp, num_samples=args.num_samples)
