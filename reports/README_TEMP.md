# UI Understanding Benchmark Reports

이 저장소는 **UI Understanding Benchmark Framework**를 통해 측정된 다양한 LVLM(Large Visual Language Models)의 성능 평가 결과를 투명하게 공유하고 이력을 관리하기 위한 공간입니다.

## 📌 주요 목적
*   **성능 추적**: 모델 버전 업데이트 및 프롬프트 최적화에 따른 성능 변화를 기록합니다.
*   **결과 공유**: 팀 내외 관계자들에게 최신 벤치마크 결과를 시각화하여 보고합니다.
*   **의사 결정 지원**: 데이터셋별 강점과 약점을 파악하여 향후 모델 고도화 방향을 설정합니다.

---

## 🚀 최신 벤치마크 요약 (Latest Results)
*마지막 업데이트: 2026-02-02*

| 모델명 | ScreenQA (ANLS) | SROIE (F1) | Widget Cap. (FM) | ScreenSpot (SR) |
| :--- | :---: | :---: | :---: | :---: |
| **Gemma 3 (4B)** | *Pending* | *Pending* | *Pending* | *Pending* |
| **Qwen2-VL (7B)** | 0.8200 | 0.9615 | 0.7992 | 0.0519 |

---

## 📊 평가 데이터셋 및 지표 안내

본 벤치마크는 실제 UI 환경에서의 모델 능력을 측정하기 위해 다음 4가지 핵심 영역을 평가합니다.

1.  **ScreenQA (Screen Question Answering)**
    *   **내용**: 모바일 화면에 대한 자연어 질의응답
    *   **핵심 지표**: **ANLS** (편집 거리 기반 유사도)
2.  **SROIE (Receipt Information Extraction)**
    *   **내용**: 영수증 이미지 내 텍스트 OCR 및 구조화 정보 추출
    *   **핵심 지표**: **F1 Score**, **WER** (단어 오류율)
3.  **Widget Captioning (Function Prediction)**
    *   **내용**: UI 요소(버튼, 입력창 등)의 기능적 역할 설명
    *   **핵심 지표**: **Functional Match** (의미적 일치도)
4.  **ScreenSpot (UI Grounding)**
    *   **내용**: 텍스트 지시문에 해당하는 UI 요소의 좌표 찾기
    *   **핵심 지표**: **Success Rate** (바운딩 박스 내 포함 여부)

> 상세한 지표 정의는 [Evaluation Metrics](./assets/eval_metrics.md) 문서를 참고하세요. (해당 레포의 경로에 맞게 수정 필요)

---

## 📂 리포트 이력 (Report History)

상세 보고서는 `/reports` 디렉토리 내에서 날짜별/모델별로 확인할 수 있습니다.

*   *(추가 예정)*

---

## 🛠 관련 리소스
*   **Benchmark Framework**: [UI_Understanding_BM](https://github.ecodesamsung.com/jhyuk88-choi/UI_Understanding_BM) (실제 실험 및 추론 코드가 포함된 메인 저장소)
    *   *참고: Samsung Ecode 정책상 현재 collaboration 멤버 추가 불가. 추후 협업 공유 레포 방안 마련 필요*
*   **Issue Tracker**: 결과에 대한 이상치 보고나 분석 요청은 Issue 탭을 이용해 주세요.

---

## 📝 리포트 작성 가이드
새로운 실험 결과를 추가할 때는 다음 형식을 권장합니다.
1.  실험 환경 (모델 파라미터, 프롬프트 버전 등) 명시
2.  데이터셋별 정량적 지표 기록
3.  오답 샘플 분석 및 정성적 평가 포함
4.  이전 실험 결과와의 비교 분석
