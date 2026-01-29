# On-Device AI Model Deployment Plan (ExecuTorch)

## Objective
Convert the "Gemma 3n" (VLM) model into a format compatible with Android using **ExecuTorch**, as MediaPipe currently does not support Gemma 3's architecture or multimodal features.

## Strategy
We will use **ExecuTorch** to export the PyTorch model into a `.pte` (PyTorch Edge) binary. 
*   **Quantization**: We will use 4-bit (INT4) or 8-bit (INT8) quantization via the **XNNPACK** backend to ensure efficient execution on mobile CPUs.
*   **Runtime**: The model will be loaded using the ExecuTorch Android Inference API.

## Prerequisites
*   Python environment with `executorch` installed.
*   Model weights downloaded (e.g., in `models/gemma-3`).
*   Development environment: **aarch64 (ARM64)** Linux.

---

## Architecture-Specific Instructions (aarch64 / Native Build)

ExecuTorch requires specific C++ components to be built for the target architecture. Since you are on **aarch64**, you can build these components natively.

### 1. Install ExecuTorch & Dependencies
```bash
# Clone the executorch repository
git clone --recursive https://github.com/pytorch/executorch.git
cd executorch

# Install python dependencies
pip install -r requirements.txt

# Setup the build environment
./install_requirements.sh
```

### 2. Build XNNPACK Backend
For optimized performance on Android, we need the XNNPACK delegate.
```bash
mkdir cmake-out
cmake -B cmake-out . -DEXECUTORCH_BUILD_XNNPACK=ON
cmake --build cmake-out -j
```

---

## Implementation Steps

### 1. Model Export Script (`scripts/export_to_executorch.py`)
We will use `torch.export` to capture the model's computation graph and then use the ExecuTorch XNNPACK backend for quantization and optimization.

**Key components of the script:**
*   Loading `AutoModelForMultimodalLM` with `torch.bfloat16`.
*   Defining a wrapper to handle image and text inputs together.
*   Quantizing to 4-bit (using `torchao` or ExecuTorch quantization tools).
*   Exporting to `gemma3.pte`.

### 2. Execution
Run the conversion script:
```bash
python scripts/export_to_executorch.py \
  --checkpoint models/gemma-3 \
  --output_path scripts/deployed_model/gemma3.pte \
  --quantization int4
```

### 3. Output
The script will generate `scripts/deployed_model/gemma3.pte`.
**Deliverable**: This file is the optimized binary for the ExecuTorch runtime on Android.

## Build Script for aarch64 (`scripts/build_executorch_aarch64.sh`)

To automate the setup on your DGX Spark (aarch64) environment, we will provide a script that:
1.  Installs system dependencies (`cmake`, `ninja`, `python3-dev`).
2.  Clones and builds ExecuTorch with XNNPACK support natively.
3.  Installs the resulting python wheels for the export process.

## Android Integration Notes

1.  **Library**: Include `libexecutorch.so` (built for Android) in your project.
2.  **JNI Wrapper**: Use the ExecuTorch Java/Kotlin API to load and run the model.
3.  **Data Preparation**:
    *   **Images**: Resize and normalize to the format expected by Gemma 3n (usually 896x896 or similar).
    *   **Text**: Use the same tokenizer as the Hugging Face model (can be exported as a separate JSON or used via a C++ tokenizer implementation).

---

## Comparison: Quantization Options

| Method | Size | Performance | Accuracy |
| :--- | :--- | :--- | :--- |
| **INT8** | ~1 byte/param | Balanced | High |
| **INT4** | ~0.5 byte/param | Fastest | Moderate |

*Recommendation: Use **INT4** for Gemma 3n on standard Android devices to keep memory usage under 4GB.*
