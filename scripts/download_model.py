import argparse
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir):
    print(f"Downloading model {model_id} to {local_dir}...")
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("--model_id", type=str, default="google/gemma-3n-E4B-it", help="Hugging Face model ID (e.g., google/gemma-3-4b-it)")
    parser.add_argument("--local_dir", type=str, default="models/gemma-3", help="Local directory to save the model")
    
    args = parser.parse_args()
    download_model(args.model_id, args.local_dir)
