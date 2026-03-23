# UI Understanding Benchmark Framework


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
This benchmark evaluates LVLMs across multiple specialized datasets for UI understanding:
- **Detection & Grounding**: Identifying UI elements based on instructions.
- **Information Extraction**: Parsing structured data from screens or documents.
- **Visual Question Answering**: Answering complex queries about screen content.
- **Widget Captioning**: Describing the functional role of UI elements.

### Supported Models
- **Gemma 3n / 4B / 12B / 27B**: Integrated via Hugging Face.
- **Qwen2-VL-7B-Instruct**: Integrated for comparative evaluation.

### Key Features
- **BF16 (BFloat16)**: Stability and high performance on NVIDIA Blackwell (GB10) and DGX Spark.
- **DDP Support**: Multi-GPU dataset sharding for high-throughput evaluation.
- **Interactive Debugging**: Per-sample inspection with real-time metrics and prompt visualization.
- **Refined Evaluation**: Enhanced metrics like Substring ANLS and Functional Match for more accurate performance measurement.

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

Use the provided script to download models to a local directory.

### Authentication
If the model is gated, export your token:
```bash
export HF_TOKEN=your_token_here
```

### Download Command
```bash
# Download Gemma 3n
python scripts/download_model.py --model_id google/gemma-3n-E4B-it --local_dir models/gemma-3n

# Download Qwen2-VL-7B-Instruct
python scripts/download_model.py --model_id Qwen/Qwen2-VL-7B-Instruct --local_dir models/Qwen2-VL-7B-Instruct
```

---

## Downloading & Exploring Datasets

The framework supports several standard benchmarks for UI understanding.

### 1. Download to Local Directory
Use `scripts/download_data.py` with the `--dataset_name` flag.

| Dataset | Description | Key Metric |
| :--- | :--- | :--- |
| `screenspot` | UI Element Detection | Success Rate |
| `sroie` | Document OCR/Extraction | F1, WER |
| `screenqa` | Screen-based VQA | ANLS |
| `widget_captioning` | UI Element Description | CIDEr, Functional Match |

**Example Command:**
```bash
# Download ScreenQA
python scripts/download_data.py --dataset_name screenqa
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
# Run SROIE evaluation (Full)
python scripts/inference.py --model_id models/gemma-3n --dataset_name sroie

# Run ScreenQA (Sampled - first 1000 items)
```

### Multi-GPU Inference (DDP)
For high-throughput benchmarking on multi-GPU systems, use the DDP-enabled script. This automatically shards the dataset across all available GPUs.

```bash
bash scripts/run_inference_ddp.sh

# Run DDP inference with specific model and dataset (Full "test" split for screenspot)
bash scripts/run_inference_ddp.sh models/gemma-3n screenspot

# Run with specific number of samples (e.g., first 100 samples)
bash scripts/run_inference_ddp.sh models/gemma-3n screenspot 100
```

> **Note**: The script automatically loads the specific split defined for each dataset in `scripts/src/common/utils.py` (e.g., `test` for ScreenSpot/ScreenQA, `train` for SROIE). If `NUM_SAMPLES` is omitted, it evaluates the entire configured split.

Alternatively, use `torchrun` directly:
```bash
```

### Features
- **Real-time Logging**: Tracks progress with `[current/total]` sample count.
- **Automated Evaluation**:
    - **SROIE**: WER (Word Error Rate), F1 Score.
    - **ScreenSpot**: Grounding Success Rate (Point-in-BBox).
    - **ScreenQA**: ANLS (with substring matching support), F1 Score.
    - **Widget Captioning**: CIDEr, METEOR, ROUGE-L, Functional Match.
- **Optimized for DGX**: Uses BF16 and greedy decoding for maximum throughput.

---

## Interactive Web Demo

A Streamlit-based web application is provided for a more user-friendly interaction with the model and datasets. It is optimized for multi-GPU systems.

### Command
```bash
bash run_demo.sh
```

### Features
- **Visual Interface**: Easily switch between datasets and samples.
- **Inference Comparison**: View side-by-side comparison of generated output vs. ground truth.
- **Diff Highlighting**: Automatically highlights differences in text extraction tasks.
- **Metric Dashboard**: Real-time display of performance metrics for the current sample.

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
- **Per-Sample Metrics**: Displays ground truth and computed scores (**WER/F1/Success**) for the specific sample.
- **Visual Debugging**: Saves the input image to `results/interactive/` for verification.

---

## Detailed Metric Definitions
For more information on how each dataset is evaluated and the custom logic applied (like Substring ANLS or Functional Match), see:
👉 [**Assets/Eval_Metrics.md**](assets/eval_metrics.md)

---

## Results & Output

Results are saved in the `results/` directory, organized by model and timestamp:
- **`benchmark_results_{dataset}.json`**: Detailed predictions and metrics for every sample.
- **`benchmark_summary_{dataset}.json`**: Aggregated average scores (ANLS, Success Rate, CIDEr, etc.).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
