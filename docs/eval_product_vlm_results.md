# 제품 이미지 → 기업 식별 제로샷 평가 (진행 중)

- 일자: 2026-07-21
- 데이터: `products/` — NASDAQ-100 제품 이미지 수집분에서 **사람 포함 458장을 YOLO11n(person, conf≥0.45)으로 제외**한 2,572장 / 101개사
- 방법: 로고 평가와 동일 하네스, 프롬프트만 제품용("이 제품을 만들거나 파는 기업 후보 3개") — PRD §3.3 VLM 폴백 시나리오
- 환경: 로고 평가와 동일 (RTX 5080, 4bit NF4)

## 결과

| 지표 | Qwen2.5-VL-7B | LLaVA-OneVision-7B |
|---|---|---|
| Top-1 Accuracy | 68.7% | (실행 중) |
| Top-3 Accuracy | 71.5% | |
| Macro F1 | 0.720 | |
| Macro Precision | 0.884 | |
| Macro Recall | 0.678 | |
| 엔티티 미검출률 | 18.0% | |
| 지연시간 (중앙값) | 513ms/장 | |

## 관찰 (Qwen)

- 로고(94.3%) 대비 크게 하락 — 과제 자체가 어려움. 제품에 브랜드 단서가 없는 업종에서 실패 집중:
  전력(AEP·CEG·XEL 무응답 다수), 인프라(FER), 경매(CPRT), 에너지(FANG→Baker Hughes 오인 21건), B2B 장비(LRCX·ADI).
- 소비재/테크(AAPL, TSLA, SBUX, NFLX 등)는 로고 수준으로 정확.
- 표본 주의: 사람 필터링으로 CTAS 3장, EA·TTWO 8장, NFLX 11장 등 일부 클래스 표본이 작음.
