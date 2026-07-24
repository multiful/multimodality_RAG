# [13] Adaptive Table Complexity Router — 전체 파이프라인 연동 결과

## 성능 지표

| 지표 | v2+penalty(전부 Docling, 현재 최선) | **v5(Adaptive Router)** |
|---|---|---|
| Recall | 90.0% (9/10) | **90.0% (9/10)** |
| Precision(근사) | 83.3% | **84.2%** |
| F1(근사) | 86.5% | **87.0%** |
| 표 단계 소요 | 11.422초(전부 Docling) | **12.437초**(SIMPLE 2개 pdfplumber + COMPLEX 12개 Docling) |
| 총 처리시간 | 1097.51초 | **992.74초** |

### 구간별 지연
- 페이지분류+텍스트+이미지설명(Qwen2.5-VL): 390.6초 (변경 없음, 기존 재사용)
- 표 구조화(Adaptive Router): 12.437초
- 엔티티 추출(Qwen2.5-VL, repetition_penalty=1.3): 589.74초

### Hit
- LG CNS
- 064400
- LG전자
- LG화학
- LG유플러스
- 네이버클라우드
- NH농협은행
- 국민연금공단
- 교보증권

### Miss
- 우리은행
