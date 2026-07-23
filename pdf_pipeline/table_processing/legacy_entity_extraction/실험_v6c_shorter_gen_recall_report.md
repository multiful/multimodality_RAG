# [14] Table-aware Entity Extraction 분기 — 결과

## 성능 지표

| 지표 | v2+penalty(전부 LLM) | v5(Adaptive Router) | **v6(Table-aware 분기)** |
|---|---|---|---|
| Recall | 90.0%(9/10) | 90.0%(9/10) | **90.0%(9/10)** |
| Precision(근사) | 83.3% | 84.2% | **97.0%** |
| F1(근사) | 86.5% | 87.0% | **93.4%** |
| 표 단계 소요 | 11.4초 | 12.4초 | **12.164초** |
| 엔티티추출 LLM 호출 수 | 6/6페이지 | 6/6페이지 | **6/6페이지(0개 생략)** |
| 엔티티추출 단계 소요 | 695.5초 | 589.7초 | **60.068초** |
| 총 처리시간 | 1097.51초 | 992.74초 | **462.8초** |

자동 도출 문서 앵커: ['064400', 'LG CNS']

### 표 타입 분류 결과
- page1 table1: finance
- page1 table2: contract_or_general
- page1 table3: contract_or_general
- page6 table1: contract_or_general
- page3 table1: contract_or_general
- page3 table2: contract_or_general
- page4 table1: finance
- page4 table2: finance
- page4 table3: finance
- page4 table4: finance
- page4 table5: finance
- page4 table6: finance
- page5 table1: contract_or_general
- page2 table1: finance

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
