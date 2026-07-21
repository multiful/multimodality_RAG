# 멀티모달 금융 정보 RAG 시스템 — PRD (v0.2)

## 1. 개요

사용자가 **로고 이미지, 제품 이미지, 텍스트(또는 이들의 조합)** 를 입력하면 관련 기업을 식별하고, 해당 기업의
**재무제표 / 관련 기사 / 현재 주가**를 검색해 제공하는 멀티모달 RAG 시스템.

- 1차 스코프: **NASDAQ 100** 기업 (`nasdaq_logo/` 데이터셋 기준 로고 이미지 확보)
- 저장소: **Supabase** (Postgres + pgvector)

## 2. 파이프라인

```
User Input (Text / Image / Text+Image)
        │
        ▼
  입력 라우팅 (필드 존재 여부 기반 — 규칙 기반, 모델 불필요)
        │
   ┌────┴─────────────────────────┐
   │                               │
 Text                        Image / Hybrid
   │                               │
Advanced RAG              SVG면 래스터 변환
                                   │
                                   ▼
                  YOLO (단일 클래스 "로고" 탐지, negative 포함 학습)
                                   │
                    박스 검출 & Confidence ≥ 0.90 ?
                    │Yes                          │No (로고 없음/제품 등)
                    ▼                               ▼
          크롭 → ViT 회사 분류            Qwen2.5-VL 기업 판정(VQA 방식)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                (Hybrid인 경우) 사용자 텍스트 쿼리와 병합
                                    │
                                    ▼
                        엔티티 링킹 (별칭/티커 매칭)
                                    │
                                    ▼
                  Supabase 검색 (재무제표/뉴스)
              + 주가는 시세 API Tool-calling 즉시 호출
                                    │
                                    ▼
                          최종 답변 생성
```

## 3. 모듈별 요구사항

### 3.1 입력 라우팅
- 텍스트/이미지 필드 존재 여부로 Text / Image / Hybrid 분기
- **규칙 기반 처리** — 별도 분류 모델(BERT 등) 불필요. 이미 알고 있는 입력 상태(필드 존재 여부)를 판단하는 데 ML 모델을 쓰는 건 불필요한 오버헤드

### 3.2 텍스트 처리 — Advanced RAG
- 기존 경험 기반으로 Advanced RAG 기법 적용 (쿼리 재작성, 멀티홉 등 확장 가능)

### 3.3 이미지 처리 — YOLO(탐지) → ViT(분류) 2단계 + VLM 폴백
- **전처리**: `nasdaq_logo/`에 SVG 파일이 섞여 있음 — 모든 모델이 픽셀 이미지 입력 필요하므로 SVG → 래스터(PNG 등) 변환 단계 필요
- **1단계 (YOLO)**: "로고" 단일 클래스 탐지기로 학습(로고 없는 negative 이미지 포함 필수). 위치 탐지와 "로고 존재 여부 판별"을 겸함 — 별도의 로고 판별 분류기 불필요
  - 박스 검출 & confidence ≥ **0.90** → 크롭 영역을 ViT로 전달
  - 박스 미검출/confidence 미만 → 비로고(제품 등)로 간주, Qwen2.5-VL 폴백
- **2단계 (ViT)**: Roboflow로 학습한 ViT가 크롭된 로고 영역을 보고 회사 분류 (기업당 학습 이미지 수가 적어 배경/negative 데이터 품질이 정확도에 중요)
- **폴백 (Qwen2.5-VL)**: YOLO가 로고를 못 찾은 경우(제품 이미지 등) VQA 방식으로 기업 판정 ("이 이미지의 기업은?") — 별도 파인튜닝 없이 제로샷 사용
- 추출된 기업명은 사용자 텍스트 쿼리와 병합(텍스트 없으면 기업명만 사용)
- 검증(2026-07-21): `NVIDIA1.jpg` → Qwen2.5-VL-7B-Instruct 제로샷 → "NVIDIA" 정확히 응답 (M4 Pro, MPS, bfloat16)

### 3.4 엔티티 링킹
- NASDAQ 100 기업을 엔티티로 정의, 별칭/다국어 표기(예: "애플"/"Apple Inc."/"AAPL") 매핑 테이블 구성
- 쿼리에서 추출된 기업명을 엔티티 링킹을 통해 DB의 정규 엔티티(티커)로 매칭

### 3.5 검색 및 데이터 소스 (Supabase)
- 재무제표/뉴스: Supabase에서 검색 (재무제표는 구조화 테이블, 뉴스는 pgvector 시맨틱 검색 — 미해결 항목 참고)
- **현재 주가: 크롤링/DB 적재 없이 시세 API를 Tool-calling으로 즉시 호출** (결정됨) — 벡터 검색·캐시 대상 아님

## 4. Open Questions (결정 필요)

| # | 질문 | 상태 |
|---|------|------|
| 1 | 재무제표를 벡터 임베딩할지, 구조화 테이블로 저장할지 | 미결정 — 검토의견 참고 |
| 2 | 뉴스 크롤링 시 원본 사이트 ToS/저작권 이슈 | 확인 필요 |
| 3 | NASDAQ 100 구성 종목 변경 시 엔티티 테이블 갱신 주기 | 미정 |
| 4 | YOLO negative(로고 없는) 이미지 데이터셋 확보 방법 | 확인 필요 |

## 5. 모델/기법 비교 계획

| 모듈 | 비교 후보 | 최종 선택 | 비고 |
|---|---|---|---|
| Logo Recognition | YOLO+ViT(2단계) vs Qwen2.5-VL vs LLaVA-OneVision-7B (제로샷) | ○○ | 후자 둘은 로컬 다운로드 완료 |
| Query Analyzer | BERT vs GPT-5 mini급 vs Qwen3 | ○○ | 모델 정식 명칭 재확인 필요 |
| Dense Embedding | BGE-M3 vs GPT Embedding | ○○ | 한/영 혼용 코퍼스 고려 |
| Retrieval | Dense vs Hybrid | Hybrid | |
| Re-ranker | 미적용 vs Cross Encoder | Cross Encoder | |
| Generator | GPT vs Qwen3-7B LoRA (vs 후보 추가 가능) | ○○ | |

평가 축(예정): **정확도 · 응답속도(Latency) · 비용 · 도메인 특화(파인튜닝 가능성)**
— 각 후보를 이 4축으로 채점 후 "최종 선택" 근거를 정량화할 것.

## 6. 고도화(Stretch) 아이디어

- 재무제표는 Supabase 관계형 테이블 + Text-to-SQL, 뉴스/서술형 텍스트만 pgvector로 이원화
- 로고/제품 판별용 평가셋 구축 (accuracy, confusion matrix 등) 및 YOLO/ViT threshold 캘리브레이션 — OOD(스코프 밖 이미지) 기권 여부 포함
- 쿼리-이미지 엔티티 충돌 처리 로직 (예: 이미지=Apple, 텍스트=Tesla)
- 최종 답변 생성 전 근거 문서 인용/그라운딩 체크
- 시세/뉴스 API 응답 캐싱 (TTL) 및 레이트리밋 대응
- "투자 조언 아님" 고지 등 컴플라이언스 문구

## 7. 스코프 밖

- NASDAQ 100 외 종목
- 실시간 스트리밍(웹소켓) 시세
