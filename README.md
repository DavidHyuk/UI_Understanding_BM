# UI Understanding Benchmark: ScreenSpot V2 with Gemma 3n

This project provides a comprehensive benchmarking framework for evaluating the **Gemma 3n** (and other Gemma 3 variants) open-source model on the **ScreenSpot V2** dataset for UI understanding and grounding tasks.

It is optimized for high-performance GPU environments (including **NVIDIA Blackwell GB10**) and supports **aarch64** (ARM64) architectures like **DGX Spark**.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Downloading Models](#downloading-models)
- [Downloading & Exploring Dataset](#downloading--exploring-dataset)
- [Running Inference & Evaluation](#running-inference--evaluation)
- [Results & Output](#results--output)

---

## Project Overview
ScreenSpot V2 is a benchmark for evaluating a model's ability to identify UI elements on Mobile, Web, and Desktop platforms based on natural language instructions. This project implements an inference pipeline that:
- Uses **Gemma 3n** (VLM) for multimodal understanding.
- Supports **BF16 (BFloat16)** precision for stability and performance.
- Applies official **Chat Templates** for correct prompting.
- Provides tools for dataset visualization and local management.

## Environment Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd UI_Understanding_BM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: `requirements.txt` includes `torch`, `transformers` (v5+), `accelerate`, `datasets`, `jupyter`, and `matplotlib`.*

### Specialized Environment: aarch64 (e.g., DGX Spark)
If running on **DGX Spark (aarch64)**, optimized PyTorch installation is recommended.
1.  **Use NVIDIA Wheels:**
    Depending on your CUDA version, you might need to install PyTorch from the NVIDIA index to ensure ARM64 optimizations:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
    # Or use the default pip install if your environment (e.g., conda) is pre-configured.
    ```

---

## Downloading Models

Gemma 3 models are available on Hugging Face. Use the provided script to download them to a local directory.

### Authentication
If the model is gated (like Gemma 3), you must be authenticated.
- **Option A:** `huggingface-cli login`
- **Option B (Recommended for DGX):** Export your token:
  ```bash
  export HF_TOKEN=your_token_here
  ```

### Download Command
```bash
# Example: Download Gemma 3n (E4B-it variant) to 'models/gemma-3n'
python scripts/download_model.py --model_id google/gemma-3n-E4B-it --local_dir models/gemma-3n
```

---

## Downloading & Exploring Dataset

This project uses the **ScreenSpot V2** dataset (`HongxinLi/ScreenSpot_v2`). It also supports **SROIE** (`rajistics/sroie`) for receipt information extraction.

### 1. Download to Local Directory
To download the dataset to `data/screenspot_v2` for easy access:
```bash
python scripts/download_data.py
```
*This will save the dataset to `data/screenspot_v2` by default.*

To download the **SROIE** dataset:
```bash
python scripts/download_data.py --dataset_name sroie
```
*This will save the dataset to `data/sroie` by default.*

### 2. Visualize Data (Jupyter Notebook)
A notebook is provided to inspect images and ground truth bounding boxes.
1.  Open `notebooks/explore_dataset.ipynb` in VSCode or Jupyter Lab.
2.  Run the cells to visualize random examples from the test set.

---

## Running Inference & Evaluation

The inference script (`scripts/inference.py`) is optimized for **Blackwell GPUs (GB10)** using **BF16**.

### Features
- **BF16 Precision:** Prevents NaN/Inf issues common with FP16 on new architectures.
- **Chat Template:** Automatically applies the correct VLM prompt format.
- **Greedy Decoding:** Ensures deterministic results for benchmarking.

### Command
```bash
python scripts/inference.py --model_id models/gemma-3n --dataset_name screenspot
```

### Parameters
- `--model_id`: Path to local model (e.g., `models/gemma-3n`) or HF ID.
- `--dataset_name`: Name of dataset to evaluate (default: `screenspot`). Choices: `screenspot`, `sroie`.
- `--output_dir`: Directory to save results (default: `results`).
- `--device`: `cuda` or `cpu`.

---

## Results & Output

After inference, the results are saved to a JSON file:
- **File:** `results/benchmark_results.json`
- **Format:**
  ```json
  [
    {
      "id": 0,
      "instruction": "Click the search icon",
      "prediction": "point [120, 45]",
      "ground_truth": [118, 42, 125, 50]
    },
    ...
  ]
  ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
