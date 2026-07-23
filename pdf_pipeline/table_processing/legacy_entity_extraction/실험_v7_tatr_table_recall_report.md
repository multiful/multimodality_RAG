# [18] v7: 표 단계를 TATR(adaptive_padding+300dpi)+pdfplumber로 교체 — 결과

## 성능 지표

| 지표 | v6c(Docling, grounding filter 적용 후) | **v7(TATR)** |
|---|---|---|
| Recall | 90.0%(9/10) | **100.0%(10/10)** |
| Precision(근사, grounding filter 전) | 97.0% | **96.9%** |
| F1(근사, grounding filter 전) | 93.4% | **98.4%** |
| 표 단계 소요 | 12.16초(Docling) | **4.113초(TATR)** |
| 엔티티추출 LLM 호출 수 | 6/6페이지 | **6/6페이지(0개 생략)** |
| 엔티티추출 단계 소요 | 60.1초 | **98.572초** |
| 총 처리시간 | 462.8초 | **493.25초** |

자동 도출 문서 앵커: ['064400', 'LG CNS']

### 표 타입 분류 결과
- page1 table1: finance
- page1 table2: contract_or_general
- page1 table3: contract_or_general
- page6 table1: contract_or_general
- page2 table1: finance
- page3 table1: contract_or_general
- page3 table2: contract_or_general
- page4 table1: finance
- page4 table2: finance
- page4 table3: finance
- page4 table4: finance
- page4 table5: finance
- page4 table6: finance
- page5 table1: contract_or_general

### Hit
- LG CNS
- 064400
- LG전자
- LG화학
- LG유플러스
- 네이버클라우드
- 우리은행
- NH농협은행
- 국민연금공단
- 교보증권

### Miss
