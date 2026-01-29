# On-Device AI Model Deployment Plan

## Objective
Convert the "Gemma 3n" (VLM) model used in `scripts/interactive_eval_with_sample.py` into a format compatible with Android (MediaPipe LLM Inference).

## Strategy
We will use the **MediaPipe GenAI Converter** to transform the Hugging Face PyTorch checkpoint into a MediaPipe Bundle (`.bin`). This binary file contains the quantized model weights and metadata required for on-device inference.

## Prerequisites
*   Python environment with `mediapipe` installed.
*   The source model must be downloaded locally (e.g., in `models/gemma-3n`).

---

## Architecture-Specific Instructions (aarch64 / DGX Spark)

If you are running on an **aarch64 (ARM64)** Linux environment (like NVIDIA DGX Spark), the default `pip install mediapipe` may not include the necessary C++ bindings (`GenerateCpuTfLite`) for the model converter.

### Option A: Use an x86_64 Machine (Recommended for Speed)
Run the conversion on a standard Intel/AMD Linux machine or Google Colab, then copy the resulting `.bin` file to your target device.

### Option B: Build MediaPipe from Source (For DGX Spark / Native ARM64)
If you must convert locally on aarch64, you need to build the MediaPipe wheel from source to include the full framework bindings.

1.  **Ensure Docker is installed.**
2.  **Run the build script**:
    ```bash
    chmod +x scripts/build_mediapipe_wheel.sh
    ./scripts/build_mediapipe_wheel.sh
    ```
3.  **Install the generated wheel**:
    ```bash
    pip install mediapipe_source/dist/mediapipe-*.whl
    ```

---

## Implementation Steps

### 1. Model Conversion Script (`scripts/convert_to_mediapipe.py`)
This script uses `mediapipe.tasks.python.genai.converter` to perform the conversion. It automatically detects your MediaPipe version's parameter naming conventions.

### 2. Execution
Run the following command from the root of the repository:

```bash
python scripts/convert_to_mediapipe.py \
  --input_model_dir models/gemma-3n \
  --output_dir scripts/deployed_model \
  --model_type GEMMA_2B
```

*Note: For VLM (PaliGemma), ensure your built MediaPipe version supports `PALIGEMMA` or `PALIGEMMA_3B` as the `model_type`.*

### 3. Output
The script will generate `scripts/deployed_model/model.bin`.
**Deliverable**: You can take this `model.bin` file and place it in your Android Studio project's `src/main/assets/` directory.

## Android Integration Notes (For Reference)
When you implement the Android app:
1.  **Dependency**: Add `implementation("com.google.mediapipe:tasks-genai:0.10.14")` (or newer) to your `build.gradle.kts`.
2.  **Inference**: Use `LlmInference.createFromOptions(...)`.
3.  **Prompting**: Ensure you match the prompt format used during training/fine-tuning.
