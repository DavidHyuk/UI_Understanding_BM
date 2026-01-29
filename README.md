# UI Understanding Benchmark: ScreenSpot V2 & SROIE with Gemma 3n

This project provides a comprehensive benchmarking framework for evaluating the **Gemma 3n** (and other Gemma 3 variants) open-source model on UI understanding, grounding, and OCR tasks.

It is optimized for high-performance GPU environments (including **NVIDIA Blackwell GB10**) and supports **aarch64** (ARM64) architectures like **DGX Spark**.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Downloading Models](#downloading-models)
- [Downloading & Exploring Datasets](#downloading--exploring-datasets)
- [Running Inference & Evaluation](#running-inference--evaluation)
- [Debugging & Interaction Testing](#debugging--interaction-testing)
- [Results & Output](#results--output)

---

## Project Overview
This benchmark evaluates a model's ability to identify UI elements (ScreenSpot V2) and extract information from documents/receipts (SROIE).
- **Gemma 3n (VLM)**: Used for multimodal understanding and grounding.
- **BF16 (BFloat16)**: Supported for stability and high performance on DGX Spark.
- **Official Chat Templates**: Applied for correct model prompting.
- **Extensible Framework**: Supports multiple datasets and custom metrics.

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

---

## Downloading Models

Use the provided script to download Gemma 3 models to a local directory.

### Authentication
If the model is gated, export your token:
```bash
export HF_TOKEN=your_token_here
```

### Download Command
```bash
python scripts/download_model.py --model_id google/gemma-3n-E4B-it --local_dir models/gemma-3n
```

---

## Downloading & Exploring Datasets

The framework currently supports **ScreenSpot V2** and **SROIE**.

### 1. Download to Local Directory
Use `scripts/download_data.py` with the `--dataset_name` flag.

**Download ScreenSpot V2 (default):**
```bash
python scripts/download_data.py --dataset_name screenspot
```

**Download SROIE:**
```bash
python scripts/download_data.py --dataset_name sroie
```

### 2. Visualize Data (Jupyter Notebook)
Open `notebooks/explore_dataset.ipynb` to inspect images and ground truth labels (bboxes, text).
- Supports switching between `screenspot` and `sroie`.
- Displays random samples with their associated ground truth labels.

---

## Running Inference & Evaluation

The `scripts/inference.py` script runs the full benchmark on a selected dataset.

### Command
```bash
# Run SROIE evaluation
python scripts/inference.py --model_id models/gemma-3n --dataset_name sroie

# Run ScreenSpot evaluation
python scripts/inference.py --model_id models/gemma-3n --dataset_name screenspot
```

### Features
- **Real-time Logging**: Tracks progress with `[current/total]` sample count.
- **Automated Evaluation**: 
    - **SROIE**: Computes Accuracy (Exact Match) and F1 Score.
    - **ScreenSpot**: Computes Grounding Success Rate (Point-in-BBox).
- **Optimized for DGX**: Uses BF16 and greedy decoding for maximum throughput.

---

## Debugging & Interaction Testing

To interactively inspect samples, view the exact prompt sent to the model, and measure inference time, use `scripts/interactive_eval_with_sample.py`.

### Command
```bash
python scripts/interactive_eval_with_sample.py --dataset_name sroie
```

### Features
- **Interactive Loop**: Enter any sample index to run inference immediately.
- **Timing**: Measures and displays the inference time for each sample.
- **Per-Sample Metrics**: Displays ground truth and computed scores (Acc/F1/Success) for the specific sample.
- **Visual Debugging**: Saves the input image to `results/interactive/` for verification.

---

## Results & Output

Results are saved in the `results/` directory:
- **`benchmark_results_{dataset}.json`**: Detailed predictions and metrics for every sample.
- **`benchmark_summary_{dataset}.json`**: Aggregated average scores (Accuracy, F1, etc.).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
