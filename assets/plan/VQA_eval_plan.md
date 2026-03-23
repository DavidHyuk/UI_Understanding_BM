# VQA Evaluation Plan for Mobile Screens (Updated)

This plan outlines the addition of Visual Question Answering (VQA) benchmarks and Captioning metrics to the evaluation suite.

## 1. Dataset Recommendations

### A. ScreenQA (VQA)
*   **Hugging Face ID:** `google/screen_qa`
*   **Primary Metrics:** ANLS (Average Normalized Levenshtein Similarity), Normalized F1.
*   **Note:** Exact Match is deprecated as it is too restrictive for VQA.

### B. Widget Captioning (Captioning)
*   **Hugging Face ID:** `m-v-p/widget_captioning`
*   **Primary Metrics:** CIDEr, METEOR, ROUGE-L.
*   **Secondary Metrics:** BLEU-4.

---

## 2. Metric Definitions & Implementation

### A. ANLS (Average Normalized Levenshtein Similarity)
ANLS measures the edit distance between strings, normalized by the length of the longer string, with a threshold logic.
*   **Formula:** $s = 1 - \frac{Lev(p, g)}{\max(|p|, |g|)}$
*   **Threshold:** If $s < 0.5$, the score is 0 (to penalize completely wrong answers).

### B. Captioning Metrics (CIDEr, METEOR, ROUGE-L)
*   **ROUGE-L:** Longest Common Subsequence based overlap.
*   **CIDEr/METEOR:** These require specialized libraries. We recommend using the `evaluate` library or `pycocoevalcap`.

---

## 3. Implementation Plan

### Step 1: Update `scripts/src/common/utils.py`
1.  Implement `calculate_anls(pred, gt)` using Levenshtein distance.
2.  Implement `calculate_rouge_l(pred, gt)`.
3.  Update `eval_vqa` and `eval_captioning` functions.

### Step 2: Library Dependencies
To support advanced metrics, the following packages should be installed:
```bash
pip install evaluate rouge-score nltk pycocoevalcap
```

### Step 3: Updated Configuration
```python
DATASET_CONFIGS = {
    "screenqa": {
        "id": "google/screen_qa",
        "eval_fn": eval_screenqa, # Uses ANLS + F1
        ...
    },
    "widget_captioning": {
        "id": "m-v-p/widget_captioning",
        "eval_fn": eval_captioning, # Uses CIDEr, METEOR, ROUGE-L
        ...
    }
}
```

---

## 4. Next Action
1.  Implement the ANLS logic in `scripts/src/common/utils.py`.
2.  Implement ROUGE-L logic or integration.
3.  Modify `DATASET_CONFIGS` to use these new evaluation functions.
