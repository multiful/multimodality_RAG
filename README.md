# 이미지 파이프라인 — 증권사 리서치 PDF → 차트/이미지 카드

네이버 금융 증권사 리서치 PDF에서 MinerU가 분리한 `chart`/`image` 크롭을 분류·해석해
RAG 색인용 카드(JSONL)로 만드는 파이프라인. `table` 블록은 판정 없이 인계 표시만 하고
텍스트/테이블 파트로 넘긴다.

**핵심 특징 — MinerU 컴포넌트만으로 완결**: 객체탐지·OCR·차트해석까지 전부 MinerU 생태계
안에서 처리하고, 전량 **CPU에서 실행 가능**하다(배포 서버에 GPU가 없다는 전제로 설계·검증됨).
자세한 아키텍처·설계 근거·벤치마크는 [docs/PIPELINE.md](docs/PIPELINE.md) 참고.

## Setup

```bash
pip install -r requirements.txt
```

- **MinerU**: `pip install -U "mineru[pipeline,vlm]"` — 레이아웃탐지·OCR(PP-OCR)·경량 VLM
  (MinerU2.5-Pro-2605-1.2B)이 이 한 패키지에 다 들어있다. 최초 실행 시 모델을 자동 다운로드한다.
- **Ollama** (선택, `--with-chart-analysis` 사용 시에만 필요): 서술형 해석 단계가 텍스트전용
  LLM을 로컬 Ollama로 호출한다. `ollama pull qwen3:8b` 필요.
- `.env`에 Supabase 접속정보(`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) — 없으면 로컬 JSONL만
  쓰고 자동으로 DB 쓰기를 건너뛴다.

## Quickstart

```bash
# 기본: 객체탐지 + OCR + 그림분류기 (LLM 호출 0회, 문서당 1~2분)
python pipeline/s2_onestop_mineru.py --doc industry_15

# + MinerU VLM 표추출 + 서술형 해석까지 (chart당 ~19초 추가)
python pipeline/s2_onestop_mineru.py --doc industry_15 --with-chart-analysis --export-txt
```

파싱본이 없으면 `data/raw/metadata.csv` 기준으로 MinerU CLI 파싱부터 자동 실행한다(원스톱).

## 출력

```
data/onestop/{doc_id}/
  onestop_cards.jsonl   # 카드 전체 (스키마: docs/PIPELINE.md 참고)
  useful/, discarded/   # 판정별 크롭 복사본
  summary.json          # 실행 통계
  vlm_chart_analysis.txt  # --export-txt 시, 사람이 읽는 리포트
```

다른 파이프라인(텍스트/테이블/RAG 색인)은 `onestop_cards.jsonl`의 `embed_text` 필드만
읽으면 되고, 나머지 필드(`ocr`, `chart_table`, `narrative` 등)는 필요할 때만 참조한다.
