# Legacy: LLM 기반 엔티티 추출 파이프라인 (아카이브)

이 폴더는 표 처리 쪽에 공존하던 **두 세대** 중 지금은 쓰지 않는 쪽을 모아둔 것이다.

- **현재 프로덕션**: `table_processing/run_table_metadata_pipeline.py` — canonical field 규칙
  매칭 + Redis 캐싱, LLM/RAG 미사용(`[19]` 사용자 요청). 여기에 `[25]`에서 OpenAI Structured
  Output(정성적 메타데이터 보완)을 추가로 배선했다.
- **이 폴더(레거시)**: `run_full_pipeline_v2.py` ~ `v7_tatr_table.py`, `run_full_pipeline_yolo26.py`,
  `postprocess_grounding_filter.py`, `run_construct_entity_validation.py`,
  `run_kwave_entity_validation.py` — 로컬 Qwen2.5-VL로 문서 전체에서 회사/기관 엔티티를 뽑아
  `ground_truth_064400.json`의 `entity_recall_target_set` 대비 Recall/Precision/F1을 재는 별도
  트랙. `[25]`에서 도입한 구조화 출력의 `entities`/`entities_mentioned` 필드가 사실상 같은
  목적(문서에서 엔티티 뽑기)을 더 단순한 방식(OpenAI Structured Output, 표/텍스트 라우팅 끝에
  자동 배선)으로 대체한다고 판단해 여기로 옮겼다.

**옮기며 고친 것**: 파일이 `table_processing/`에서 한 단계 더 안쪽(`legacy_entity_extraction/`)
으로 들어오면서 `ROOT = Path(__file__).resolve().parent.parent.parent`(프로젝트 루트 계산)와
일부 파일의 `sys.path.insert(0, str(Path(__file__).resolve().parent))`(형제 모듈,
예: `table_type_router.py` import용)가 깨졌던 것을 각각 `.parent` 한 단계씩 더 추가해 수정.
결과 JSON/리포트 md 파일도 스크립트와 함께 이동시켜서 그대로 재실행 가능하다(단, 재실행 자체를
권장하는 건 아니고 — 히스토리 보존 목적).

**삭제하지 않은 이유**: Recall/Precision/F1 A/B 히스토리(v2→v7까지 병목 진단, 할루시네이션
발견·수정 과정 등)가 `실험_v*_recall_report.md`에 남아있고, 나중에 구조화 출력의 entities 필드
품질을 검증할 때 비교 기준으로 참고할 수 있어서 보존.
