# ScreenSpot Prompt Engineering

## Issue Description
The initial evaluation of the ScreenSpot dataset resulted in 0 metrics and missing visualizations.
- **Goal:** Evaluate the model on ScreenSpot dataset (GUI element grounding).
- **Initial Behavior:** The model generated natural language descriptions (e.g., "The element is the close button...") but failed to provide numerical coordinates.
- **Impact:** The `parse_coords` utility returned empty lists, causing metrics (IoU/Success) to be 0 and the visualization to fail (no "Red" prediction box).

## Improvements

### 1. Prompt Refinement
We modified the instruction to be explicit about the required output format and scope. The model needed clear direction to output coordinates rather than just describing the element.

- **Original Prompt:**
  ```python
  f"Detect the element described: {ex['instruction']}"
  ```
- **New Prompt:**
  ```python
  f"Detect the specific element described: {ex['instruction']} Output the bounding box coordinates for this element only."
  ```

### 2. Output Parsing
After updating the prompt, the model began producing JSON-structured output containing bounding boxes. The original parser only supported `<loc>` tokens or plain float numbers, causing it to fail on the new JSON format.

- **Observed Output Format:**
  ```json
  [
    {"box_2d": [7, 923, 33, 947], "label": "button"},
    ...
  ]
  ```
- **Parser Update (`demo/utils.py`):**
  - Added support for parsing JSON objects embedded in the text.
  - Extracts `box_2d` from the first item in the JSON list.
  - Normalizes integer coordinates (dividing by 1024.0) to the 0-1 range required by the evaluation script.

## Results
- **Success:** The system now correctly extracts coordinates from the model's structured response.
- **Metrics:** Metrics are now correctly calculated based on the parsed coordinates.
- **Visualization:** The web app now renders the prediction (Red box) alongside the Ground Truth (Green box), enabling proper visual debugging.

### 3. Coordinate Normalization Fix

## Issue
Even with correct JSON parsing, IOU scores were 0 for many samples.
- **Symptoms:** Bounding boxes looked correct visually but scores were 0.
- **Cause:** The model outputs absolute pixel coordinates (e.g., `[233, 155...]`) relative to the image size. The evaluation script default normalization (dividing by 1024) was incorrect for images that were not 1024x1024 (e.g. 540x960). Ground Truth is normalized (0-1).

## Fix
- **Evaluation Logic:** Updated `eval_screenspot` to accept `image_size`.
- **Normalization:** If the model outputs pixel coordinates (> 1.0), they are now normalized using the actual image dimensions (`c / height` or `c / width`) instead of the fixed 1024 scale.

## Verification
- **Result:** IOU scores improved from 0.0 to ~1.0 for valid predictions. The coordinates now correctly match the normalized Ground Truth.
- **Data Inspection:** Verified via `scripts/debug/check_screenspot_data.py` that dataset image sizes vary (e.g., 960x540, 1920x1080), confirming the necessity of dynamic normalization.
