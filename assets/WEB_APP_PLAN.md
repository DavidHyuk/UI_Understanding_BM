# Web App Development Plan for AI Model Demo

## Objective
Transform the existing terminal-based interactive evaluation script (`scripts/interactive_eval_with_sample.py`) into a professional-looking web application for demonstrating AI capabilities to stakeholders.

## Technology Stack
- **Framework**: Streamlit (Python-based, rapid development, interactive data/image visualization).
- **Model**: Hugging Face Transformers (`AutoModelForMultimodalLM`).
- **Processing**: PyTorch, PIL.

## Architecture

### 1. Directory Structure
```
demo/
├── app.py              # Main Streamlit application entry point
└── utils.py            # Shared logic for model loading, inference, and evaluation (refactored from original script)
```

### 2. Functional Components
- **Model Handler**: 
  - Singleton pattern to load the model and processor once (caching).
  - Inference function accepting image + prompt.
- **Dataset Handler**:
  - Load ScreenSpot and SROIE datasets.
  - Retrieve samples by index.
- **Visualizer**:
  - Draw bounding boxes/points for ScreenSpot tasks.
  - Display side-by-side comparisons (Ground Truth vs. Prediction).
- **Evaluator**:
  - Compute metrics (Success rate for ScreenSpot, WER/F1 for SROIE).

### 3. User Interface Design
- **Sidebar**:
  - **Settings**: Model path input, Device selection (CPU/CUDA).
  - **Task Selection**: Dropdown for "ScreenSpot" (GUI Navigation) vs "SROIE" (Document Extraction).
- **Main Content**:
  - **Header**: Project Title & Description.
  - **Sample Control**: Slider/Input for dataset index navigation. "Random Sample" button.
  - **Inference Area**: "Run Model" button.
  - **Results Display (2-Column Layout)**:
    - **Column 1 (Input)**: Original Image with User Prompt displayed below.
    - **Column 2 (Output)**: Model generated text, Parsed result (Coords/Text), Evaluation Metrics (highlighted Green/Red).
  - **Visual Feedback**:
    - For ScreenSpot: Overlay predicted point/box on the image.

## Development Steps

1.  [x] **Setup Environment**: Install Streamlit and verify dependencies.
2.  [x] **Refactor Logic**: Extract core logic (loading, eval, coordinate parsing) from `scripts/interactive_eval_with_sample.py` into `demo/utils.py`.
3.  [x] **Basic UI Skeleton**: Create `demo/app.py` with the sidebar and layout structure.
4.  [x] **Integrate Model**: Implement model loading with `@st.cache_resource` to prevent reloading on every interaction.
5.  [x] **Implement Inference Loop**: Connect UI inputs to the inference engine and display raw text output.
6.  [x] **Visualization & Evaluation**: Add bounding box drawing and metric calculation display.
7.  [x] **Polish**: Improve UI aesthetics (markdown, status indicators, error handling).
