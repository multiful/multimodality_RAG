# MASTER PROMPT — 리서치 PDF 이미지 파이프라인 고도화 구현

> 사용법: 새 폴더에서 VS Code Claude Code를 열고, 이 파일을 프로젝트 루트에
> **`CLAUDE.md`** 로 저장하면 모든 세션이 이 스펙을 자동으로 참조한다.
> 첫 세션은 플랜 모드(Shift+Tab)로 시작해서 "이 문서의 구현 순서 1번부터 진행해줘"라고 하면 됨.

---

당신은 한국 증권사 리서치 PDF에서 추출된 이미지들을 분류·정형화·저장하는
**이미지 파이프라인**을 구현하는 시니어 파이썬 엔지니어다. 아래 스펙은 실측 검증을
거친 확정 설계이므로 임의로 변경하지 말고, 변경이 필요하면 이유를 설명하고 승인을 받아라.

## 1. 프로젝트 컨텍스트 (검증된 사실)

- 전체 시스템: 네이버 금융 리서치 PDF 600개(6카테고리×100) → MinerU 파싱 →
  파트별 처리(이미지/테이블/텍스트) → Supabase 색인 → 로컬 LLM RAG 질의응답
- **내 담당은 이미지 파트뿐이다.** 테이블·텍스트 처리는 팀원 파트이므로 구현하지 않는다
- 상류(MinerU) 출력은 이미 검증됨. `{doc_id}\auto\` 아래:
  - `*_middle.json`: `pdf_info[].para_blocks[]`에 `{type, bbox}` — type은
    `chart | image | table | text | title | ...`이고 chart/image/table은 하위 `blocks[]`에
    `chart_body/chart_caption/image_body/...` 서브블록을 가짐
  - `*_content_list.json`: `{type, page_idx, img_path, img_caption[]}` — 크롭 파일 경로 포함
  - `images\`: 크롭된 jpg 파일들 (해시 파일명)
- VLM: **Qwen3-VL-8B via Ollama** (`http://localhost:11434`, model `qwen3-vl:8b`),
  GPU 12GB, 장당 5~10초. Ollama API: `POST /api/chat`, messages에 `images:[base64]`, `stream:false`
- 저장소: **Supabase** (Postgres + pgvector + Storage). 접속 정보는 `.env`
  (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) — 절대 하드코딩 금지, `.gitignore`에 `.env` 필수
- 실행 환경: Windows, Python 3.10+, venv. 사용자는 더블클릭 bat 실행을 선호한다

## 2. 범위 (엄격)

- **처리 대상: MinerU 블록 타입 `chart`와 `image`만.**
- `table` 크롭은 처리하지 않고 인계 목록(`data\handoff\handoff_tables.jsonl`)에만 기록
- 임베딩·색인·검색은 텍스트 파트 소관 — 나는 검색용 텍스트(`embed_text`)를 만들어 저장하는 데까지만
- 팀 규약: **남의 테이블(`text_chunks`, `financial_tables` 등)에 쓰기 금지.**
  내 소유 테이블은 `image_cards`와 `pipeline_status`의 `stage='image'` 행뿐

## 3. 파이프라인 아키텍처 — 4단계 게이트 + 캐싱

```
입력: {PARSED_DIR}\{doc_id}\auto\ 의 chart·image 크롭   (PARSED_DIR은 config로 주입)
  │
  ├─ [A] 규칙 필터 (즉시 탈락 → discarded\rule\, 사유 기록)
  │      · 너비 or 높이 < 100px / 종횡비 >8:1 or <1:8 / 면적 < 15,000px²
  │      · 단 type=chart는 크기 무관 통과 (스파크라인 보호)
  ├─ [B] 캐시·중복 게이트
  │      · sha256(이미지 바이트)+prompt_ver+model 로 VLM 캐시 조회 (L1 로컬 → L2 Supabase)
  │      · pHash(64bit) 계산·저장. 해밍거리 0 이면 기존 판정 복사(dedup_of 링크),
  │        거리 1~6은 "유사"로만 표시하고 VLM 재실행 (시계열 차트 오복사 방지)
  ├─ [C] VLM 판정 (Qwen3-VL-8B, temperature 0.1, JSON 강제)
  │      · 캡션 블록·리포트 제목을 프롬프트에 동봉 / 장변 1280px 초과 시 리사이즈(캐시)
  │      · JSON 파싱 실패 1회 재시도 → 재실패 시 에러 기록 후 다음 건 진행
  │      · confidence < 0.6 → review_queue=true
  └─ [D] 저장
         · useful=true  → data\images\useful\{doc_id}\{image_id}.jpg + Storage 업로드 + upsert
         · useful=false → data\images\discarded\vlm\{doc_id}\ (보관만)
         · VLM이 table_image로 판정 → useful 처리 + 인계 목록에도 추가 (이중 등록)
         · 전 건(탈락 포함) image_cards에 기록 — filter_stage 컬럼에 통과/탈락 사유
```

**캐싱 원칙**: 키는 항상 `내용 해시 + 버전`. 파일명·시간 기준 캐싱 금지.
L1 = `data\cache\{stage}\{key[:2]}\{key}.json`, L2 = image_cards의 (content_hash, prompt_ver) 행.
버전을 올리면 자동 전량 미스. 적중 통계를 실행 끝에 로그·pipeline_status.detail에 기록.

## 4. VLM 프롬프트 (v2 — 이 문구로 시작, 개선 시 prompt_ver 올려 별도 저장)

```
이 이미지는 한국 증권사 리서치 리포트에서 추출된 그림입니다.
[캡션: {caption}]  [리포트: {doc_title} / {category}]
아래 JSON만 출력하세요.
{
 "type": "line_chart|bar_chart|pie_chart|radar_chart|candle_chart|mixed_chart|table_image|diagram|photo|logo|decoration|other",
 "useful": true|false,
 "confidence": 0.0~1.0,
 "title": "이미지 제목",
 "ocr_text": "축 라벨·범례·수치 포함 모든 텍스트",
 "summary": "핵심 내용 1~2문장 (한국어)",
 "entities": ["언급된 기업명·티커·지표명"],
 "chart_data": { "x_axis":"...", "y_axis":"...", "unit":"...",
                 "series":[{"name":"...","points":[["23/6",70000]]}] }  // 차트일 때만, 불확실하면 null
}
```

판정 철학: **로고를 유용으로 넣는 오류(FP)가 차트를 놓치는 것(FN)보다 해롭다.**
애매하면 useful=false 쪽으로 (탈락분은 보관되므로 복구 가능).

## 5. 데이터 스키마

```sql
create extension if not exists vector;
create table if not exists image_cards (
  image_id      text primary key,      -- {doc_id}_p{page}_{type}{idx}  ※ 결정적 생성, 랜덤·시간 금지
  doc_id        text not null,
  page          int, block_type text,  -- chart|image
  bbox          float8[], caption text,
  content_hash  text,                  -- sha256 (캐시 L2 키)
  phash         text, dedup_of text references image_cards(image_id),
  vlm_type text, vlm_useful boolean, confidence numeric,
  ocr_text text, summary text, entities text[],
  chart_data    jsonb,
  embed_text    text,                  -- title+summary+entities+ocr(500자) — 텍스트 파트가 읽어감
  review_queue  boolean default false, reviewed boolean default false,
  prompt_ver    text, filter_stage text,
  storage_path  text, local_path text,
  updated_at    timestamptz default now()
);
create index if not exists ic_hash on image_cards (content_hash, prompt_ver);
create index if not exists ic_queue on image_cards (review_queue) where review_queue;

create table if not exists pipeline_status (
  doc_id text, stage text, status text,   -- pending|running|done|error
  ver text, detail jsonb, updated_at timestamptz default now(),
  primary key (doc_id, stage)
);
```

인계 파일 `data\handoff\handoff_tables.jsonl` (1줄=1건):
`{"image_id","doc_id","page","bbox","crop_path","caption","source":"mineru_table"|"vlm_reclass","handoff_ver":"1"}`

## 6. 만들 파일 (전부 이 레포 안에서)

| 파일 | 내용 |
|---|---|
| `common.py` | config 로드(.env·경로), sb() supabase 클라이언트, upsert(500행 배치+재시도3), content_hash, cache_get/put, ollama_chat(JSON 강제), 로깅 |
| `s2_image_pipeline.py` | 메인. CLI: `--category --doc --retry-errors --rejudge-queue --no-cache --prompt-ver v2 --dry-run`. resume 기본(완료분 스킵). 시작/종료 시 pipeline_status upsert |
| `review_viewer.py` | review_queue+표본을 카드형 HTML로 생성 (✓유용/✗불필요/유형수정 버튼 → 선택 결과 JSON 다운로드). self-contained HTML, 이미지 base64 embed |
| `apply_review.py` | 뷰어에서 내보낸 JSON → image_cards 반영(reviewed=true) + `eval\image_labels.csv` 누적 |
| `eval_image.py` | image_labels.csv vs 판정 대조 → accuracy/precision/recall/혼동행렬 출력 + prompt_ver 간 비교 |
| `clean_cache.py` | `--older-than 30d` 캐시 정리 |
| `run_images.bat` / `run_review.bat` / `run_eval.bat` | venv 자동 생성·활성화 + 실행 (더블클릭용, chcp 65001) |
| `requirements.txt`, `.env.example`, `.gitignore`, `README.md` | 표준 세팅 |

## 7. 품질 기준 (Definition of Done)

- [ ] 결정적 image_id — 같은 입력에 항상 같은 ID (재실행 안전)
- [ ] 어떤 단계에서 죽어도 재실행하면 이어서 진행 (resume)
- [ ] 개별 건 실패가 배치를 멈추지 않음 (기록 후 계속), 단 Ollama 5연속 실패 시 중단
- [ ] 캐시 적중 시 VLM 미호출 확인 (로그로 검증 가능)
- [ ] 판정 정확도: 표본 50장 수동 라벨 기준 accuracy ≥ 90%, 로고 FP ≤ 5%
- [ ] 처리 속도: 장당 p50 ≤ 7초 (GPU)
- [ ] table 크롭이 한 건도 VLM을 타지 않고 전량 인계 목록에 존재
- [ ] pipeline_status에 문서별 image stage 상태·통계 기록

## 8. 구현 순서 (이 순서로 진행, 각 단계 끝날 때마다 실행 가능한 상태 유지)

1. 프로젝트 스캐폴드: requirements, .env.example, .gitignore, common.py (Supabase 연결 테스트 포함)
2. Supabase 스키마 적용 스크립트(`setup_db.py`) + 연결 스모크 테스트
3. s2 골격: 크롭 로딩 + [A] 규칙 필터 + image_cards 기록 (VLM 없이 dry-run 동작 확인)
4. [B] 캐시·pHash 게이트 + handoff_tables 분리
5. [C] Ollama VLM 판정 + 프롬프트 v2 + 에러 처리 (소량 문서로 실측)
6. [D] 저장·Storage 업로드 + bat 파일들
7. review_viewer + apply_review + eval_image (검수 루프 완성)
8. 전체 배치 실측 → 통계 확인 → DoD 체크리스트 검증

## 9. 금지사항

- 남의 테이블 쓰기, 파일명/시간 기반 캐시 키, 비결정적 ID, .env 하드코딩·커밋
- 테이블·텍스트 파트 로직 구현 (인계 파일 생성까지만)
- 스펙 임의 변경 (필요 시 근거 제시 후 승인 요청)
- discard 이미지 삭제 (반드시 보관 — 복구·튜닝 근거)

## 10. 경로 설정 (config, 새 폴더 기준)

```
PARSED_DIR   = 기존 pdfex 산출물 경로 (예: C:\Users\wodlf\OneDrive\Desktop\pdfex\demo_out\mineru_raw
               또는 배치 후 data\parsed) — .env 또는 config.yaml로 주입, 코드에 박지 말 것
OUTPUT_ROOT  = 이 레포의 data\   (images\, cache\, handoff\, logs\)
```

시작 시 PARSED_DIR 존재·구조를 검증하고, 없으면 명확한 안내 메시지를 출력하라.
