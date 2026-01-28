import argparse
from datasets import load_dataset

# Configuration for supported datasets
DATASETS = {
    "screenspot": {
        "id": "HongxinLi/ScreenSpot_v2",
        "default_dir": "data/screenspot_v2"
    },
    "sroie": {
        "id": "rajistics/sroie",
        "default_dir": "data/sroie"
    }
}

def download_dataset(dataset_id, local_dir=None):
    print(f"Downloading dataset {dataset_id}...")
    # trust_remote_code=True is often needed for datasets with custom loading scripts
    try:
        dataset = load_dataset(dataset_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if local_dir:
        dataset.save_to_disk(local_dir)
        print(f"Dataset saved to {local_dir}")
    else:
        print("Dataset downloaded to default Hugging Face cache.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face")
    
    # Selection by predefined name
    parser.add_argument("--dataset_name", type=str, choices=list(DATASETS.keys()), default="screenspot",
                        help=f"Name of the dataset to download. Supported: {', '.join(DATASETS.keys())}")
    
    # Manual overrides
    parser.add_argument("--dataset_id", type=str, help="Hugging Face dataset ID (overrides dataset_name configuration)")
    parser.add_argument("--local_dir", type=str, help="Local directory to save the dataset (overrides default)")
    
    args = parser.parse_args()
    
    # Determine final dataset ID and local directory
    if args.dataset_id:
        ds_id = args.dataset_id
        ds_dir = args.local_dir
    else:
        config = DATASETS[args.dataset_name]
        ds_id = config["id"]
        # Use provided local_dir if available, otherwise use default from config
        ds_dir = args.local_dir if args.local_dir else config["default_dir"]
    
    download_dataset(ds_id, ds_dir)
