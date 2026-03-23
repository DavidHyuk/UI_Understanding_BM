import argparse
import os
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir, ignore_patterns=None):
    print(f"Downloading model {model_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    
    if ignore_patterns is None:
        ignore_patterns = ["*.pth", "*.pt", "*.msgpack", "*.h5"]

    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_patterns
    )
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("--model_id", type=str, default="google/gemma-3n-E4B-it", help="Hugging Face model ID (e.g., google/gemma-3-4b-it or Qwen/Qwen2-VL-7B-Instruct)")
    parser.add_argument("--local_dir", type=str, default="models/gemma-3n", help="Local directory to save the model (e.g., models/gemma-3n or models/Qwen2-VL-7B-Instruct)")
    
    args = parser.parse_args()
    download_model(args.model_id, args.local_dir)

