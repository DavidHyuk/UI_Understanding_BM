import argparse
from datasets import load_dataset

def download_dataset(dataset_id, local_dir=None):
    print(f"Downloading dataset {dataset_id}...")
    dataset = load_dataset(dataset_id)
    if local_dir:
        dataset.save_to_disk(local_dir)
        print(f"Dataset saved to {local_dir}")
    else:
        print("Dataset downloaded to default Hugging Face cache.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face")
    parser.add_argument("--dataset_id", type=str, default="HongxinLi/ScreenSpot_v2", help="Hugging Face dataset ID")
    parser.add_argument("--local_dir", type=str, default="data/screenspot_v2", help="Local directory to save the dataset (optional). If not set, uses HF cache.")
    
    args = parser.parse_args()
    download_dataset(args.dataset_id, args.local_dir)
