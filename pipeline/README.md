# 리서치 PDF 이미지 파이프라인 (고도화판)

한국 증권사 리서치 PDF에서 **차트·이미지 크롭을 분류·정형화·저장**하는 이미지 파이프라인.
설계 스펙은 [MASTER_PROMPT_IMAGE_PIPELINE.md](../MASTER_PROMPT_IMAGE_PIPELINE.md).
1차 실측 대상은 **하나증권 산업분석 리포트 20건**.

## 무엇이 "고도화"인가 (기본형 대비 델타)

| 고도화 포인트 | 내용 | 효과(측정 지표) |
|---|---|---|
| **VLM 캐시 (L1)** | `content_hash + prompt_ver + model` 키로 판정 결과 캐시 | **재실행 시 VLM 0회** — 절감률 |
| **pHash 중복제거** | dHash 해밍거리 0=판정복사, 1~6=유사표시 후 재판정 | 완전중복 복사 수 / 유사 표시 |
| **confidence 게이트** | VLM confidence<0.6 → `review_queue=true` | 저신뢰 자동 선별율 |
| **table 인계** | 표는 VLM 안 태우고 `handoff_tables.jsonl` 로만 넘김 | 인계 무결성(표 VLM 0건) |
| **규칙필터 고도화** | 크기·종횡비 + 면적 하한, chart는 크기무관 통과 | 규칙 탈락률 |
| **prompt_ver** | 프롬프트 버전 관리 → 올리면 캐시 전량 미스 재판정 | 회귀 비교 |
| **eval_image** | 위 지표 + 판정분포·속도 p50/p95 + (라벨 시)정확도 → HTML 대시보드 | 한 장으로 상태 확인 |

## 실행 순서 (더블클릭 bat)

```
run_collect.bat   → 하나증권 산업분석 20건 수집 (data\raw\industry\)
run_parse.bat     → MinerU 파싱 (data\parsed\{doc_id}\)         ※ GPU, 문서당 수십 초
run_images.bat    → 이미지 고도화 파이프라인 (규칙+캐시+pHash+VLM) → image_cards.jsonl
run_images.bat    → (한 번 더) 캐시 적중 확인 (VLM 0회 로그)
run_eval.bat      → 고도화 지표 산출 + eval\eval_report.html 자동 오픈
run_review.bat    → 표본 검수 뷰어 → 라벨 CSV 저장 → eval\image_labels.csv → run_eval 재실행 시 정확도 채워짐
```

CLI 직접 실행 예:
```
run_images.bat --category industry            # 산업분석만
run_images.bat --doc industry_hana_01         # 특정 문서만
run_images.bat --no-cache                     # 캐시 무시(전량 재판정)
run_images.bat --rules-only                   # VLM 없이 규칙필터만
run_images.bat --route clf                    # 그림 분류기로 junk 선컷 후 VLM (S3 게이트)
```

## 실행 환경 (연산은 로컬 GPU에서 무료)

- **MinerU 3.4 + torch** : PDF 레이아웃 파싱. `pdfex\demo_venv` 를 재사용 (`_env.bat` 가 자동 탐색)
- **Ollama + `qwen3-vl:8b`** : VLM 이미지 판정 (`http://localhost:11434`)
- **저장** : 기본은 로컬 JSONL (`data\images\image_cards.jsonl`)이 원본.
  `.env` 에 `SUPABASE_URL/SUPABASE_SERVICE_KEY` 를 넣으면 `image_cards` 테이블에도 upsert (선택).

## 구조

```
파이프라인 고도화\
├── collect_hana_industry.ps1     하나증권 산업분석 수집기 (섹터/증권사 파싱 버그 수정판)
├── pipeline\
│   ├── common.py                 설정·해시·L1캐시·dHash·Ollama·(선택)Supabase
│   ├── s1_parse.py               MinerU 배치 파싱 (resume)
│   ├── s2_image_pipeline.py  ★   게이트: 규칙→캐시/pHash→(그림분류기)→VLM(conf)→저장+table인계
│   ├── figure_classifier.py      S3 그림분류기 게이트 (DocumentFigureClassifier-v2.5, EfficientNet-B0, --route clf)
│   ├── eval_image.py             고도화 지표 산출 + HTML 대시보드
│   └── review_viewer.py          표본 검수 뷰어 (라벨→CSV)
├── data\
│   ├── raw\industry\*.pdf + metadata.csv   원본 20건
│   ├── parsed\{doc_id}\          MinerU 출력
│   ├── images\{useful,discarded}\ + image_cards.jsonl
│   ├── cache\vlm\                L1 VLM 캐시
│   └── handoff\handoff_tables.jsonl   표 인계 목록
├── eval\eval_report.html / review.html / image_labels.csv
└── run_*.bat, requirements.txt, .env.example, .gitignore
```

## 재실행 안전성 (resume)

- 결정적 `image_id = {doc_id}_p{page}_{block_type}{idx}` — 같은 입력 = 같은 ID
- 완료분(useful/discarded)은 스킵, 어디서 죽어도 다시 돌리면 이어서 진행
- 개별 건 실패는 기록 후 계속, **Ollama 5연속 실패 시에만 배치 중단**
- discard 이미지는 삭제하지 않고 보관 (복구·튜닝 근거)

---

## 개발 로그 (문제해결 과정)

> 이 파이프라인이 어떤 문제를 만나 어떻게 바뀌었는지 기록. 새 항목은 **맨 위에** 추가.
> 형식: `문제 → 조사/발견 → 결정`. 날짜는 절대표기(YYYY-MM-DD).

### 2026-07-22 · #10 엔티티 추출 3변형 A/B — ChartQA의 한글 붕괴와 개선
- **문제**: 베이스라인 / 고도화(+분류기) / 고도화+ChartQA 를 **차트 엔티티(기업·티커·지표) P/R·시간**으로 비교(골든셋은 LLM이 이미지 직접 판독). 보고서: [docs/엔티티_평가_보고서.md](../docs/엔티티_평가_보고서.md), 하네스 `eval_entities.py`.
- **조사/발견** (표본 12장, 골든 56개):
  | 변형 | 시간 | P | R | F1 |
  |---|---:|---:|---:|---:|
  | V1 베이스라인 | 326s | 92.7% | **91.1%** | **91.9%** |
  | V2 +분류기 | **222s** | **100%** | 62.5% | 76.9% |
  | V3 +ChartQA | 28s | 7.4% | 3.6% | **4.8%** |
  | V3′ 하이브리드(원시) | 250s | 58.3% | 62.5% | 60.3% |
  | V4 개선(필터) | 250s | **100%** | 62.5% | 76.9% |
  - **V3 붕괴**: DePlot이 한글 범례를 gibberish(`현물가`→`<0x..>น`)로 오독 → 엔티티 전멸.
  - **V3′ 정밀도 하락**: gibberish·축라벨(`1Q21`,`2026F`)이 합집합 오염 → P 58%.
  - **V2 recall 격차**: 분류기가 엔티티-리치 "junk"(주요 고객사 로고 16개)를 컷 → recall 62.5%.
- **결정/개선**: ① **DePlot은 수치 전용**(엔티티 부적합) ② `clean_deplot_entities()`로 gibberish·축라벨 필터 → **V4에서 P 58%→100%, F1 60→77** ③ **엔티티 추출을 저장 게이트와 분리** 권장(분류기=저장, DePlot=수치, VLM=엔티티). junk 엔티티 recall 회수는 향후 저비용 OCR 과제로 기록.

### 2026-07-22 · #9 CUDA 셋업 + 강한 차트모델 비교 (TinyChart→ChartGemma)
- **문제**: DePlot이 다계열·한글에 약하고 venv가 `torch=CPU`라 속도도 불리. 더 강한 ChartQA 모델을 CUDA로 재평가.
- **조사/발견**:
  - **CUDA torch 셋업**: 격리 venv(`cuda_venv`)에 `torch 2.11.0+cu128` → RTX 5080(sm_120) 검증 OK. **DePlot GPU 2~3s/장**(CPU 25s → ~10배). 즉 DePlot은 GPU에선 Qwen3-VL(~27s)보다 훨씬 빠름.
  - **TinyChart 블로커**: config `transformers_version=4.37.2` 핀 + `auto_map`이 phi-2 기본코드만 가리켜 **비전 아키텍처가 HF에 없음**(github repo 필수) + 12.8GB. Windows/Blackwell 불확실 → **가볍게 가기로 하고 보류.**
  - **ChartGemma(PaliGemma-3B)**: 표준 로드, CUDA 즉시. **한글 범례 읽음**(`현물가/고정가/프리미엄`) ↔ DePlot gibberish 대비 우수. 단 **수치 QA 부정확**(최고값 y~23을 x좌표 26.3으로, 현물가 최종 ~31을 1로). QA 특화라 표 추출엔 부적합.
- **결정**: **단일 승자 없음 → 하이브리드 확정.** 모델별 강점: Qwen3-VL(한글라벨+요약, 수치X) · DePlot(정확 수치·구조화, 단일계열, 한글X) · ChartGemma(한글읽기+빠름, 수치 부정확). 최적 = **분류기(트리아지) + DePlot(단일계열 수치 골격) + Qwen3-VL/MinerU(한글 라벨·범례·요약)**. 강한 단일모델(TinyChart 등)은 필요성 대비 셋업비용 커서 후순위.

  | 모델 | 한글 라벨 | 차트 수치 | 구조화 | 속도(GPU) |
  |---|---|---|---|---|
  | Qwen3-VL(베이스) | ✅ | ❌ 축눈금만 | ❌ | ~27s |
  | DePlot(ChartQA) | ❌ | ✅ 정확(단일) | ✅ 표 | ~2-3s |
  | ChartGemma | ✅ | ❌ 부정확 | QA | ~0.5-1s |

### 2026-07-22 · #8 ChartQA(DePlot) 3변형 A/B — 상호보완 발견
- **문제**: 베이스라인(Qwen3-VL) / +분류기 / **+ChartQA(DePlot 차트→표)** 3변형 성능 비교. 영어학습 DePlot이 한국 리포트 차트에서 통하나?
- **조사/발견**: 표본 12건. 실제 곡선과 대조 —
  - **단일계열 차트**: DePlot이 **실제 데이터 값을 정확히 추출**(예: 2.23→20.33, 곡선과 일치). 반면 **Qwen3-VL ocr_text는 축 눈금만 나열, 데이터값 0** → 베이스라인의 차트 수치이해가 얕았음.
  - **다계열 차트**: DePlot은 범례가 gibberish(비한글) + 계열 누락·축라벨 오염. Qwen3-VL은 한글 범례 정확하나 수치 구조 없음.
  - 속도: 이 venv `torch=CPU`라 DePlot ~25s/장(GPU면 2~4s) — 인프라 이슈.
- **결정**: 승자독식 아님 → **상호보완**. 최적형 = **분류기(트리아지) + DePlot(수치 골격) + Qwen3-VL/MinerU(한글 라벨·범례·제목)** 하이브리드. DePlot은 VLM 대체가 아니라 **차트 수치추출 보강**으로 자리매김(단일계열에 강, 다계열은 후처리 필요). 벤치 하네스 `bench_variants.py`, 뷰어 `eval/bench_variants.html`.

### 2026-07-22 · #7 그림 분류기 A/B — VLM의 FP를 발견하다
- **문제**: `--route clf`(그림분류기 게이트)가 정말 쓸모 있나? junk 리치 문서 `industry_hana_17`로 full vs clf A/B.
- **조사/발견**: 분류기가 로고/사진 24장(17%)을 VLM 없이 선컷. 처음엔 "VLM은 그중 73%를 useful이라 함 → 분류기 오폐기?"로 보였으나, **이미지를 직접 열어 확인**하니 정반대였다 — 고객사 로고 그리드·제품사진·건물 조감도였고 **분류기가 정답, VLM이 로고/사진을 useful로 오판(FP)** 하고 있었다. 우리가 처음부터 "제일 해로운 오류"로 지목한 그 FP.
- **결정**: 분류기는 **비용(17% VLM 절감) + 품질(VLM FP 차단)** 이중 이득 → `--route clf` 채택 방향. 단 `full_page_image`는 인포그래픽(실수치 포함) 사례가 있어 **자동 junk에서 제외**하고 VLM에 위임. 안전 junk = 로고·사진·아이콘 등 수치 없는 것만.

### 2026-07-22 · #6 그림 분류기 도입 (Figure Classifier)
- **문제**: 싼 트리아지를 VLM으로 하면 이미지 인코딩 비용을 그대로 낸다(~3s). 더 싼 게이트가 필요.
- **조사/발견**: HuggingFace `docling-project/DocumentFigureClassifier-v2.5`(EfficientNet-B0, 4.08M, MIT, 실제 PDF그림 학습). 우리 크롭에 스모크 테스트 → 차트 94/94 VLM 일치(conf~1.0), 49ms/장.
- **결정**: `figure_classifier.py`로 S3 게이트 신설, `s2 --route clf`로 통합. **보수적 임계**(고신뢰 junk만 컷, 애매하면 VLM)로 FN 회피. 라벨은 review 루프로 축적 → 후일 전용 분류기 학습(플라이휠).

### 2026-07-22 · #5 "MinerU가 갭 라우팅인가?" 검증
- **문제**: 현재가 "MinerU가 1차 OCR → 못한 부분만 VLM" 흐름인가?
- **조사/발견**: 파싱 산출물 확인 — MinerU는 **본문·캡션은 OCR하지만 차트 *내부* 수치는 안 읽음**(chart 블록 `content` 빈값). 우리 s2는 chart/image 크롭을 **통째로 VLM에 재OCR**. 즉 블록타입 라우팅만 있고 "갭만 넘기기"는 아님. 그리고 이미지 판독은 텍스트 Qwen3-8B가 아니라 **비전 Qwen3-VL-8B**.
- **결정**: 갭 라우팅 설계 수립(MinerU 텍스트 재사용 + VLM은 차트내부·유용성만 + 요약/엔티티는 텍스트LLM). "지표 없는 확장 금지" 원칙에 따라 **실측 후 채택**하기로.

### 2026-07-22 · #4 자원 낭비 문제 제기 (사진에 OCR)
- **문제**: 투자에 무의미한 사진/로고까지 VLM OCR을 돌리면 낭비 아닌가?
- **조사/발견**: 현 s2는 분류·OCR·요약을 **한 번의 VLM 호출에 묶어** 유용성 판단 *전에* OCR을 이미 생성. 규칙필터는 작은 것만 공짜로 거르고, chart로 오탐된 로고는 그냥 통과. → 구조적 낭비 존재.
- **결정**: "싼 판정 먼저 → 유용할 때만 OCR" 방향 확정. junk 비율 실측을 선행조건으로.

### 2026-07-22 · #3 이미지 파이프라인 고도화 구현
- **결정**: 기본형(규칙+VLM) 위에 **캐시(content_hash+prompt_ver+model), pHash 중복제거, confidence→review_queue, table 인계, prompt_ver**를 얹어 `s2_image_pipeline.py` 작성. 재실행 시 VLM 0회(캐시적중 100%) 실측 검증.

### 2026-07-22 · #2 하나증권 수집 — 브로커 파싱 버그
- **문제**: 산업분석에서 하나증권이 0건 수집됨.
- **조사/발견**: 네이버 목록 첫 칸이 "분류(섹터)"라, 순진한 파싱이 **섹터명을 증권사로 오인**. 원본 수집기의 버그.
- **결정**: **제목 칸 다음 칸**을 증권사로 읽도록 위치 기반 파싱으로 수정(`collect_hana_industry.ps1`). 기존 오라벨 9건도 복구. → 하나증권 산업분석 20건 확보.

<!-- 템플릿 (복사해서 맨 위에 추가)
### YYYY-MM-DD · #N 제목
- **문제**: 
- **조사/발견**: 
- **결정**: 
-->

