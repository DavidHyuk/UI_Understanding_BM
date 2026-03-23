# Evaluation Metrics and Dataset Overview (English)

This document provides a detailed overview of the datasets and evaluation metrics used in the UI Understanding Benchmark.

---

## 1. ScreenSpot V2
**Dataset Description:** Focuses on UI element detection and grounding. The model is given a screenshot and a natural language instruction (e.g., "Click on the settings icon") and must identify the correct pixel coordinates.

*   **Metric: Success Rate (Accuracy)**
    *   **Description:** Measures if the model's predicted point falls within the bounding box of the target element.
    *   **Logic:**
        *   If the model outputs a point `[y, x]`, it is used directly.
        *   If the model outputs a box `[y1, x1, y2, x2]`, the **center point** is calculated and used.
        *   If the final point is inside `[xmin, ymin, xmax, ymax]`, score = 1.0, otherwise 0.0.
    *   **Interpretation:** A strict "Found/Not Found" metric for grounding tasks.
*   **Metric: IoU (Intersection over Union)**
    *   **Description:** Measures the overlap between the predicted bounding box and the ground truth box.
    *   **Logic:** `Area(Intersection) / Area(Union)`. Only applicable if the model outputs a box (4 coordinates).
    *   **Interpretation:** A continuous measure of how well the predicted area aligns with the target. Higher is better (0.0 to 1.0).

## 2. SROIE (Scanned Receipt OCR)
**Dataset Description:** Evaluates Optical Character Recognition (OCR) and structured information extraction from receipt images. Key fields include Company, Address, Date, and Total.

*   **Metric: F1 Score**
    *   **Description:** Evaluates word-level precision and recall for extracted fields.
    *   **Interpretation:** Higher is better (0.0 to 1.0). Captures how many key words were correctly extracted.
*   **Metric: WER (Word Error Rate)**
    *   **Description:** Standard OCR metric measuring edit distance at the word level.
    *   **Interpretation:** Lower is better. 0.0 indicates a perfect transcript.

## 3. ScreenQA
**Dataset Description:** A Visual Question Answering (VQA) benchmark specifically for mobile screens. Questions range from simple counts to complex spatial reasoning.

*   **Metric: ANLS (Average Normalized Levenshtein Similarity)**
    *   **Description:** Standard VQA metric based on character-level edit distance.
    *   **Custom Enhancement:** Added **Substring Matching**. If the prediction is a subset of the ground truth (e.g., "12" matching "There are 12 items"), it receives a full 1.0 score.
    *   **Interpretation:** Primary metric for ScreenQA. Higher values indicate more accurate text answers.
*   **Metric: F1 Score**
    *   **Description:** Measures word overlap between prediction and reference.

## 4. Widget Captioning
**Dataset Description:** Requires the model to generate a functional description of a highlighted UI element (e.g., a "back button" or "search bar").

*   **Metric: FUNCTIONAL_MATCH (Custom)**
    *   **Description:** A flexible metric that grants 1.0 if the core functional keywords are present in the prediction.
    *   **Logic:** Uses substring matching and significant word overlap (>= 50%) to validate semantic correctness.
*   **Metric: CIDEr / METEOR / ROUGE-L**
    *   **Description:** Standard image captioning metrics. CIDEr focuses on consensus and rarity, METEOR on synonyms, and ROUGE-L on structural overlap.

---
---

# 데이터셋 개요 및 평가 지표 (한국어)

이 문서는 UI 이해 벤치마크에 포함된 데이터셋과 각 평가 지표에 대한 상세 설명을 제공합니다.

---

## 1. ScreenSpot V2
**데이터셋 설명:** UI 요소 검출 및 그라운딩(Grounding)에 중점을 둡니다. 화면 스크린샷과 자연어 명령(예: "설정 아이콘 클릭")이 주어지면 모델은 올바른 픽셀 좌표를 찾아야 합니다.

*   **지표: Success Rate (성공률)**
    *   **설명:** 모델이 예측한 지점이 대상 요소의 바운딩 박스 내부에 포함되는지 측정합니다.
    *   **로직:**
        *   모델이 점(`[y, x]`)을 출력하면 해당 지점을 사용합니다.
        *   모델이 박스(`[y1, x1, y2, x2]`)를 출력하면 박스의 **중심점**을 계산하여 사용합니다.
        *   최종 지점이 정답 박스 `[xmin, ymin, xmax, ymax]` 안에 있으면 1.0점, 아니면 0.0점입니다.
    *   **해석:** 그라운딩 성능을 측정하는 가장 엄격한 지표입니다.
*   **지표: IoU (Intersection over Union)**
    *   **설명:** 모델이 예측한 바운딩 박스와 정답 박스가 얼마나 겹치는지 측정합니다.
    *   **로직:** `교집합 넓이 / 합집합 넓이`. 모델이 박스(4좌표)를 출력한 경우에만 계산됩니다.
    *   **해석:** 0.0에서 1.0 사이의 값으로, 박스의 위치와 크기가 정답과 얼마나 일치하는지 보여줍니다.

## 2. SROIE (영수증 OCR)
**데이터셋 설명:** 영수증 이미지에서의 광학 문자 인식(OCR) 및 구조화된 정보(상호, 주소, 날짜, 합계) 추출 능력을 평가합니다.

*   **지표: F1 Score**
    *   **설명:** 추출된 필드에 대해 단어 단위의 정밀도(Precision)와 재현율(Recall)을 평가합니다.
    *   **해석:** 높을수록 좋습니다 (0.0 ~ 1.0). 정답 단어들을 얼마나 정확하게 찾아냈는지 보여줍니다.
*   **지표: WER (Word Error Rate)**
    *   **설명:** 단어 단위의 편집 거리를 측정하는 표준 OCR 지표입니다.
    *   **해석:** 낮을수록 좋습니다. 0.0은 정답과 완벽히 일치함을 의미합니다.

## 3. ScreenQA
**데이터셋 설명:** 모바일 화면에 특화된 시각적 질의응답(VQA) 벤치마크입니다. 단순 개수 세기부터 복잡한 공간 추론까지 다양한 질문을 포함합니다.

*   **지표: ANLS (Average Normalized Levenshtein Similarity)**
    *   **설명:** 글자 단위 편집 거리를 기반으로 한 표준 VQA 지표입니다.
    *   **커스텀 개선:** **부분 일치(Substring Matching)** 로직을 추가했습니다. 모델의 답변이 정답 문장의 일부(예: "12"와 "총 12개입니다")인 경우 만점(1.0)을 부여합니다.
    *   **해석:** ScreenQA의 주지표로, 1.0은 완벽한 정답을 의미합니다.
*   **지표: F1 Score**
    *   **설명:** 예측값과 정답 사이의 단어 단위 겹침을 측정합니다.

## 4. Widget Captioning
**데이터셋 설명:** 강조된 UI 요소의 기능(예: "뒤로가기 버튼", "검색창")을 설명하는 문구를 생성해야 합니다.

*   **지표: FUNCTIONAL_MATCH (커스텀)**
    *   **설명:** 표현 방식이 다르더라도 핵심 기능 키워드가 포함되었는지 판단하는 유연한 지표입니다.
    *   **로직:** 부분 일치 및 주요 단어 겹침 비율(50% 이상)을 사용하여 의미적 정답 여부를 확인합니다.
*   **지표: CIDEr / METEOR / ROUGE-L**
    *   **설명:** 표준 이미지 캡셔닝 지표입니다. CIDEr는 용어의 희귀성과 합의도를, METEOR는 동의어를, ROUGE-L은 구조적 유사성을 중점적으로 봅니다.
