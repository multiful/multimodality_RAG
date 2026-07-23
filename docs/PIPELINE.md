# 이미지 파이프라인 아키텍처

## 1. 배경·범위

- 전체 시스템: 네이버 금융 리서치 PDF → MinerU 파싱 → 파트별 처리(이미지/테이블/텍스트) →
  Supabase 색인 → RAG 질의응답. 이 문서는 **이미지 파트**(`pipeline/s2_onestop_mineru.py`)만 다룬다.
- 처리 대상: MinerU 블록 타입 `chart`와 `image`만. `table`은 판정하지 않고 인계 표시만 한다
  (텍스트/테이블 파트 소관 — 남의 테이블에 직접 쓰지 않음).
- **왜 CPU 전제인가**: 배포 서버에 상시 GPU를 두지 않는다는 결정에 맞춰 설계했다. 검증 결과
  레이아웃탐지·OCR·MinerU 경량 VLM까지는 CPU로도 실용적인 속도가 나온다(§4 벤치마크 참고).
  서빙(RAG 질의응답) 자체도 저장된 카드를 읽기만 하므로 GPU가 필요 없다 — GPU가 꼭 필요한 부분은
  없고, 원한다면 배치 인제스트만 별도 GPU 머신에서 돌리고 결과만 옮기는 것도 가능하다.

## 2. 파이프라인 4단계

```
PDF 입력
  │
  ▼
[0] MinerU 파싱 (pipeline 백엔드)  — 파싱본 없으면 자동 실행
    레이아웃탐지 + PP-OCR(캡션/각주) + 표/수식인식
    → type(chart/image/table) · bbox · score, 크롭 이미지
  │
  ├─ table 블록 → 판정 없이 인계 표시(status=handoff), 텍스트/테이블 파트 소관
  │
  └─ chart/image 블록
      │
      ▼
    [1] 규칙필터        폭·높이<100px / 종횡비>8:1 / 면적<15,000px²
                        (chart는 크기 무관 통과 — 스파크라인 보호)
      │
      ▼
    [2] MinerU 내장 OCR  PytorchPaddleOCR(PP-OCRv6 det + PP-OCRv5 rec, korean)
                        크롭 내부 텍스트를 줄 단위로 정밀 추출. ~0.5초/장, CPU에서도 GPU와
                        속도 차이 거의 없음(경량 CNN이라 병목이 아님)
      │                 판정: chart→useful 자동 / image→OCR 텍스트≥20자면 useful
      │
      ├─ [3] 그림 분류기 (--with-classifier, 기본 on)
      │      DocumentFigureClassifier-v2.5 (EfficientNet-B0, 4M) — 26종 세부라벨
      │      (line_chart/bar_chart/pie_chart/photo/logo/icon 등). ~10ms/장으로 사실상 무료.
      │      MinerU 탐지 결과와 상호검증 신호로만 병기 — 판정 자체를 바꾸지 않고,
      │      불일치 시 review_queue 후보로만 표시(사람 검수 범위를 크게 줄이는 용도)
      │
      └─ [4] 차트 분석 (--with-chart-analysis, 기본 off — 비용 큼, useful chart만)
             4a. MinerU 내장 VLM(MinerU2.5-Pro-2605-1.2B) "Image Analysis"
                 → 근사 데이터 표 추출 (예: `| Date | 현물가 | 고정가 | ... |`)
                 15.5초/장 (CPU)
             4b. 4a의 표를 텍스트전용 LLM(qwen3:8b, Ollama)에 다시 넣어 애널리스트 문체
                 서술형 해석으로 변환 (비전 인코딩 없음 → 3.5초/장, 훨씬 빠름)
      │
      ▼
    저장: onestop_cards.jsonl (전 건 기록) + useful/discarded 크롭 복사
```

## 3. 설계에서 확정된 것 vs 시행착오

### 3.1 MinerU 내장 VLM은 "OCR"이 아니라 "해석 1회 호출"이다

`mineru_vl_utils`의 `content_extract(type="chart")`는 프롬프트가 **"Image Analysis:"** 고정
트리거 하나뿐이다. "글자를 읽는 단계"와 "내용을 해석하는 단계"가 나뉘어 있지 않고, 한 번의
추론으로 축·범례를 읽으면서 표로 재구성까지 한다. 정밀한 문자 인식이 필요하면(줄 단위 신뢰도
포함) 같은 MinerU 패키지 안의 **다른 컴포넌트인 PP-OCR**(`mineru.model.ocr.pytorch_paddle`,
단계 [2])을 써야 한다 — 이름이 비슷해도 완전히 다른 두 모델이다.

### 3.2 MinerU2.5-Pro는 자유 지시를 따르지 않는다 (실측 확인)

"차트를 서술형으로 해석해줘" 같은 커스텀 한국어 프롬프트를 `prompts` 파라미터로 넘겨봤지만,
모델은 지시를 무시하고 학습된 고정 패턴(`<|box_start|>`, `<|txt_contd_src|>` 같은 MinerU 내부
전용 특수토큰 포함)을 그대로 반복 생성했다. **이 모델의 `prompts`는 커스터마이즈하지 말 것** —
좁게 파인튜닝된 추출 전용 모델이라 자유 지시를 이해하지 못한다. 서술형 해석이 필요하면
반드시 별도의 범용 LLM(텍스트든 비전이든)에 맡겨야 한다(§2의 [4b]).

### 3.3 `max_new_tokens` 무제한 폭주 버그

`MinerUSamplingParams`의 `max_new_tokens` 기본값이 `None`(무제한)이다. 다항목(국가별
수입비중 등) 차트 일부에서 EOS 토큰 없이 CPU로 수십 분씩 생성이 멈추지 않는 사례를 실측으로
발견했다. **`sampling_params`로 `max_new_tokens` 상한을 반드시 걸 것**(코드 기본값 1024) —
상한 적용 후 장당 편차가 6~34초 → 14~17초로 정상화됨을 확인했다.

### 3.4 서술형 해석(4b)의 신뢰도 주의

4b는 원본 이미지를 다시 보지 않고 4a의 표만 텍스트로 받아 작문한다. 표에 없는 구체적 수치를
그럴듯하게 채워 넣는 사례가 관찰됐다(예: 표에 없는 개별 기업 금액을 서술문에 추가). 따라서
`narrative`는 참고용 요약이고, **근거는 항상 `chart_table`(4a) 쪽을 우선한다.**

## 4. 벤치마크 (industry_15, 31p, chart/image 106장 기준)

### 4.1 단계별 속도 (CPU)

| 단계 | 모델 | 속도 |
|---|---|---|
| MinerU 파싱 | PDF-Extract-Kit-1.0 | 문서당 ~152초 |
| [2] OCR | PP-OCR | ~0.5초/장 (신뢰도 평균 0.986) |
| [3] 분류기 | DocumentFigureClassifier-v2.5 | ~10ms/장 |
| [4a] 차트표추출 | MinerU2.5-Pro-2605-1.2B | 15.5초/장 (104/106 성공) |
| [4b] 서술형해석 | qwen3:8b | 3.5초/장 (103/103 성공) |

### 4.2 그림 분류기 vs MinerU 탐지 — 차트/비차트 분리 정확도

VLM(Qwen3-VL) 판정을 심판으로 대조: **106/106 (100%) 일치**. 두 방식 중 하나만 있어도
차트 분리는 충분하다는 근거 — 분류기는 주로 "세부 라벨"(line/bar/pie 구분, 사진·로고 식별)에
부가가치가 있다.

### 4.3 엔티티 추출 정확도 — 검색용 embed_text 품질 (골든셋 9장/38엔티티)

Qwen3-VL 같은 비전 LLM 판정 없이 어디까지 갈 수 있는지 비교:

| 구성 | 시간 | Precision | Recall | F1 |
|---|---|---|---|---|
| VLM(Qwen3-VL) 전량 판정 | 249.7s | 100.0% | 92.1% | 95.9% |
| **원스톱(MinerU만, 이 파이프라인 기본 경로)** | **4.5s** | 85.7% | 68.4% | 76.1% |

품질 상한(95.9%)이 필요하면 여전히 비전 LLM 판정이 필요하지만, 정보 손실을 어느 정도
감수할 수 있으면(F1 76.1%) 이 파이프라인만으로 **약 55배 빠르게** 처리할 수 있다.

## 5. 카드 스키마 (`onestop_cards.jsonl`, 1행 = 1블록)

**안정 필드** (모든 실행에서 항상 채움, 다른 파이프라인은 이것만 보면 됨):

| 필드 | 설명 |
|---|---|
| `image_id` | `{doc_id}_p{page}_{block_type}{순번}` — 결정적 생성 |
| `doc_id`, `page`, `block_type` | chart/image/table |
| `bbox`, `det_score` | MinerU 탐지 결과 |
| `caption`, `footnote` | 파싱 시 MinerU가 이미 OCR한 캡션/각주 |
| `status` | useful / discarded_rule / discarded_ocr / handoff / skipped |
| `filter_stage` | 탈락·통과 사유 문자열 |
| `embed_text` | 검색/RAG용 — caption+footnote+ocr(500자)(+narrative) |
| `crop` | 크롭 파일 경로 |

**선택 필드** (해당 단계를 켰을 때만):

| 단계 | 필드 |
|---|---|
| [2] OCR | `ocr {text, n_boxes, mean_conf, seconds}`, `ocr_lines[]` |
| [3] 분류기 | `clf_label`, `clf_confidence`, `clf_route`, `clf_agree`(MinerU탐지와 일치 여부, bool) |
| [4] 차트분석 | `chart_table`, `chart_table_sec`, `narrative`, `narrative_sec` |

## 6. CLI

```
python pipeline/s2_onestop_mineru.py --doc {doc_id} [옵션]

--lang               MinerU OCR 언어 (기본 korean)
--with-classifier    [3] 그림 분류기 (기본 on) / --no-classifier로 끄기
--with-chart-analysis [4] MinerU VLM 표추출+서술형해석 (기본 off)
--chart-max-new-tokens  [4a] 생성 토큰 상한 (기본 1024, §3.3 참고)
--narrative-model    [4b] 텍스트전용 LLM (기본 config의 LLM_MODEL)
--force              완료분도 전 단계 재계산
--export-txt         완료 후 사람이 읽는 .txt 리포트 생성
```

resume이 기본 동작이다 — 이미 채워진 필드는 `--force` 없이 재계산하지 않는다. 예를 들어
기본 경로로 먼저 돌린 뒤 나중에 `--with-chart-analysis`를 추가해서 재실행하면, 이미 끝난
[1][2][3]은 스킵하고 [4]만 새로 계산한다.

## 7. 알려진 한계 / TODO

- `table` 블록은 이 스크립트 내 `status=handoff`로만 표시되고, 팀 공용 인계 파일
  (`data/handoff/handoff_tables.jsonl`)에는 아직 쓰지 않는다 — 텍스트/테이블 파트와 연결할 때
  필요하면 추가해야 한다.
- [4] 차트분석은 `block_type=chart`에만 적용된다. 사진·로고 등 `image` 타입은 서술형 해석
  대상이 아니다(애초에 수치 해석이 의미 없는 컨텐츠).
- 4b(서술형 해석)의 할루시네이션 위험은 §3.4 참고 — 정확도가 중요한 용도에는 `chart_table`을
  우선 신뢰할 것.
- 검증은 단일 문서(industry_15, 31p) 기준. 다른 카테고리(사진·다이어그램이 많은 리포트 등)에서는
  재검증 필요.
