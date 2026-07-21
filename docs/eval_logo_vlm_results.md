# Logo Recognition 제로샷 평가 결과 — Qwen2.5-VL vs LLaVA-OneVision (PRD §5.1 ②)

- 일자: 2026-07-21
- 데이터: `logos/` — NASDAQ-100 로고 1,992장 / 101개사 (전부 positive, negative 셋 없음 → 로고 인지(①)와 OOD 기권율은 미측정)
- 환경: RTX 5080 16GB, PyTorch cu128, **4bit(NF4) 양자화 — 두 모델 동일 조건**, greedy decoding
- 방법: VQA "후보 기업 3개를 가능성 순으로" → 자유 텍스트 응답을 엔티티 링킹(`scripts/entity_linking.py`)으로 티커 정규화 후 채점
- 재현: `python scripts/eval_logo_vlm.py --model qwen|llava --data logos --out results/*.jsonl` → `python scripts/eval_metrics.py results/qwen.jsonl results/llava.jsonl`

## 종합 비교

| 지표 | Qwen2.5-VL-7B-Instruct | LLaVA-OneVision-7B-OV |
|---|---|---|
| Top-1 Accuracy | **94.3%** | 83.2% |
| Top-3 Accuracy | **94.7%** | 83.7% |
| Macro F1 | **0.943** | 0.859 |
| Macro Precision | 0.970 | 0.952 |
| Macro Recall | **0.933** | 0.822 |
| 엔티티 미검출률* | **2.8%** | 14.3% |
| 지연시간 (중앙값) | **470ms/장** | 1,405ms/장 |
| 지연시간 (p95) | 801ms | 2,104ms |

\* 응답에서 어떤 기업도 링킹되지 않은 비율. LLaVA는 "I don't know"류 응답이 많아 미검출이 Top-1 손실의 대부분을 차지 (미검출 제외 시 두 모델 정밀도는 비슷 — LLaVA Macro P 0.952).

## 관찰

- **Qwen이 4개 축(정확도·지연·강건성·후보 다양성) 전부 우위.** 특히 3배 빠르면서 Top-1이 11%p 높다.
- LLaVA 실패 모드는 오분류보다 **무응답/비인식** — PANW(19/20), BKNG(17/20), MELI·XEL(16/20) 등 중견 브랜드에서 기업명을 아예 못 냄.
- Qwen 취약 클래스는 저인지도 신생 기업에 집중: NBIS(Nebius→Apple 오인 8건, F1 0.18), WDAY(→Walmart 9건), CEG(→Exelon 5건).
- 공통으로 NBIS, MELI, BKNG, ALAB이 어려움 → ViT 파인튜닝 경로(YOLO+ViT)가 커버해야 할 영역.

## PRD §5 비교표 반영 제안

| 모듈 | 비교 후보 | 제안 |
|---|---|---|
| Logo Recognition (VLM 폴백) | Qwen2.5-VL vs LLaVA-OneVision-7B | **Qwen2.5-VL-7B-Instruct 채택** — 정확도/속도/강건성 전부 우위 |

남은 비교: YOLO+ViT(2단계) 학습 후 동일 지표로 3자 비교 + negative 셋 확보 시 로고 인지(①)/OOD 기권율 측정.
