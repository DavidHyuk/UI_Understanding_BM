# Code Refactoring Plan

The goal is to centralize the dataset configurations and utility functions to avoid code duplication across `demo/app.py`, `scripts/inference.py`, and `scripts/interactive_eval_with_sample.py`.

## Tasks

- [x] Centralize `DATASET_CONFIGS` and utility functions in `demo/utils.py`
- [x] Refactor `scripts/inference.py` to use shared utilities
- [x] Refactor `scripts/interactive_eval_with_sample.py` to use shared utilities
- [x] Verify that all components (app and scripts) function correctly after refactoring
- [x] Remove redundant code in scripts

## Phase 2: Unified Model Interface (Strategy Pattern)


### Goals
1.  **Unified Interface**: `model.generate_content(prompt_text, image, **kwargs)`
2.  **Encapsulation**: Hide model-specific preprocessing and generation logic (e.g., `<ctrl99>` tokens vs Chat Templates) inside wrapper classes.
3.  **Simplification**: `demo/app.py` and `scripts/inference.py` should not know which model is running.

### Design

```mermaid
classDiagram
    class BaseModelWrapper {
        <<interface>>
        +load(model_path, device)
        +generate_content(prompt_text, image, **kwargs)
        +get_device()
    }

        +load(model_path, device)
        +generate_content(prompt_text, image, **kwargs)
    }

    class HuggingFaceWrapper {
        +load(model_path, device)
        +generate_content(prompt_text, image, **kwargs)
    }
    
    class ModelFactory {
        +get_model(model_id, device)
    }

    BaseModelWrapper <|-- HuggingFaceWrapper
    ModelFactory ..> BaseModelWrapper : Creates
```

### Implementation Steps

1.  **Create `demo/model_wrappers.py`**:
    *   [x] Abstract Base Class.
    *   [x] Factory class.

2.  **Refactor `demo/utils.py`**:
    *   [x] Delegate loading to `ModelFactory`.
    *   [x] Remove raw model loading logic.

3.  **Refactor Clients**:
    *   [x] Update `demo/app.py` to use the new unified interface.
    *   [x] Update `scripts/inference.py` to use the new unified interface.

## Phase 3: Structural Organization

Moved shared utilities and model wrappers to `scripts/src/common/` to better reflect their usage across the project (both demo and scripts).

- [x] Create `scripts/src/common/` directory.
- [x] Move `demo/utils.py` -> `scripts/src/common/utils.py`
- [x] Move `demo/model_wrappers.py` -> `scripts/src/common/model_wrappers.py`
- [x] Update imports in `demo/app.py`, `scripts/inference.py`, `scripts/interactive_eval_with_sample.py` and debug scripts.
