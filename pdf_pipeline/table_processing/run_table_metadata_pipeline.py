"""[19] Table Metadata Pipeline — 사용자 피드백 전면 반영판(2차: 텍스트 품질 3건 동적 정제 추가).

YOLO Table Crop -> Adaptive Router(SIMPLE/COMPLEX) -> Table Type Classify(finance는 스킵, DB에 있음)
    -> Row Parser(컬럼 수에 따라 TATR grid 또는 word-clustering 동적 선택, label+cells)
    -> 텍스트 정규화(한글 과잉 띄어쓰기/값 타입별 정제/글리프 매핑 실패 플래그 — 전부 동적 판단)
    -> Canonical Field Mapping(alias -> 표준필드) -> Structured Table JSON -> Redis Cache

RAG/Retriever/LLM은 이 파이프라인에 전혀 들어가지 않는다(사용자 요청: 순수 규칙 기반).
다른 팀원 파트(이미지/텍스트 메타데이터, 검색)와는 아직 연결하지 않고, **표 파트 단독으로
정상 동작하는지만** 이번에 검증한다 — 캐시 레코드의 "source" 필드는 처음부터 "table"로 못박아서,
나중에 이미지/텍스트 추출기가 같은 스키마로 "image"/"text" 레코드를 추가할 수 있게 설계해둔다.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402
from table_type_router import classify_table, is_pure_financial_line_item  # noqa: E402
from row_parser import parse_table_adaptive, parse_simple_table_from_words  # noqa: E402
from canonical_field_schema import match_canonical_field, detect_wide_form  # noqa: E402
from text_normalization import fix_hangul_spacing, clean_value_by_type, detect_cid_artifact, is_over_spaced  # noqa: E402
from metadata_cache import get_client, cache_metadata, get_all_metadata  # noqa: E402
from value_enrichment import extract_unit_and_value, compute_trend, evaluate_derived_signals  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [25] structured_output이 pdf_pipeline/에 있음
from structured_output import extract_table_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "result_table_metadata_pipeline.json"

TATR_DPI = 300
TATR_TOP_PAD_PT = 35 / (150 / 72)
TATR_SIDE_PAD_PT = 12 / (150 / 72)

# [25] 표 라우팅 끝에 OpenAI Structured Output(gpt-4o-mini)으로 정성적 메타데이터(엔티티/논조/
# 특이사항 등, structured_output.TableMetadata 참고)를 추가할지 여부. 유료 API 호출이라 기본 False
# — 켜려면 True로 바꾸거나 build_records(add_structured_metadata=True)로 직접 호출.
ADD_STRUCTURED_METADATA = False


def _normalize_row(row: dict, table_over_spaced: bool) -> dict:
    """행 단위로 한글 과잉 띄어쓰기만 우선 정리(값 타입별 정제는 canonical field가 정해진 뒤
    적용 — 필드마다 기대 타입이 달라서 매칭 이전엔 적용할 기준이 없음).

    table_over_spaced: 이 표의 raw_text(전체, 토큰 수가 충분해 판단이 신뢰도 높음) 기준으로
    이미 내린 판단 — "LG전 자"처럼 셀 하나만 보면 토큰이 2개뿐이라 자체 판단이 불안정한 경우를
    표 단위 판단으로 보강(force=True로 적용, 폰트 렌더링 특성은 표/페이지 전체에 공통이므로 안전)."""
    return {
        "label": fix_hangul_spacing(row["label"], force=table_over_spaced),
        "cells": [fix_hangul_spacing(c, force=table_over_spaced) for c in row["cells"]],
        "row_top_pt": row.get("row_top_pt"),
    }


def _finalize_value(raw_value: str, cf) -> dict:
    """canonical field 매칭 결과에 따라 값 타입 기반 정제 + cid 아티팩트 탐지 + 단위/숫자값
    분리를 적용. cf가 None(매칭 안 됨)이면 정제 없이 원문만 유지(단위 분리는 필드 기대 타입을
    알아야 의미가 있어서 매칭 안 된 값엔 적용 안 함)."""
    value = raw_value
    numeric_value, unit = None, None
    if cf:
        value = clean_value_by_type(value, cf.value_type)
        if cf.value_type in ("numeric_amount", "percent"):
            parsed = extract_unit_and_value(value, default_unit=cf.unit)
            numeric_value, unit = parsed["numeric_value"], parsed["unit"]
    quality = "unmapped_glyph" if detect_cid_artifact(value) else None
    return {"value": value, "data_quality": quality, "numeric_value": numeric_value, "unit": unit}


def build_records(pdf_id: str, add_structured_metadata: bool = False, openai_client=None,
                   page_boxes: dict = None, yolo_model=None, structured_metadata_workers: int = 8,
                   sector: str = None):
    """page_boxes: [26] page_classification 등 앞단에서 이미 같은 PDF에 YOLO를 돌린 결과가 있으면
    `{page_number(1-based): [(cls_name, fitz.Rect), ...]}` 형태로 넘겨 표 라우터가 YOLO를 다시
    호출하지 않게 함(`page_classification.page_classifier.classify_pdf()`가 반환하는 cached_boxes를
    모아서 그대로 전달 가능 — text_processing.text_extraction.process_pdf(page_boxes=...)와 동일 관례).

    [29] add_structured_metadata=True일 때 표마다 순차로 API를 부르면 표가 많은 문서(K-Wave 104개
    표에서 실측 +115초/finance 표 55개만으로)에서 병목이 커서, 표 파싱(로컬 연산)을 모두 끝낸 뒤
    구조화 출력 호출만 `concurrent.futures.ThreadPoolExecutor`로 한꺼번에 병렬 디스패치한다."""
    import fitz
    routed = detect_and_route(RouterThresholds(), yolo_model=yolo_model, page_boxes=page_boxes)
    print(f"[router] 표 {len(routed)}개(SIMPLE {sum(1 for r in routed if r['complexity']=='simple')} / "
          f"COMPLEX {sum(1 for r in routed if r['complexity']=='complex')})", flush=True)

    if add_structured_metadata and openai_client is None:
        import os
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # [37] 사용자 지적("페이지 늘어날수록 조정할 부분" — TATR이 표 개수만큼 반복되는 병목) 대응.
    # 주의: 처음에 더미 이미지(800x500 흰 배경)로 측정했을 땐 MPS가 2.79배 빠르다고 나왔는데,
    # 실제 LGCNS 표 12개(300dpi 실제 크롭, 훨씬 큼)로 다시 재보니 CPU 322ms/표 -> MPS 292ms/표로
    # 겨우 9% 개선 — 더미 이미지 벤치마크가 실제 워크로드를 대표 못 한 것으로 확인(작은 이미지는
    # MPS 커널 디스패치/디바이스 전송 오버헤드 대비 이득이 작음). 배치 처리도 시도했으나 CPU에서
    # 오히려 손해(표마다 크기가 달라 가장 큰 쪽에 패딩하는 오버헤드가 배치 이득을 상쇄) — 기각.
    # 9%는 크진 않지만 손해는 없어 유지. 표 처리가 텍스트 처리와 동시에(병렬) 돌아가는 구조가
    # 되면 BGE-m3-ko와 MPS 자원을 경합할 수 있음(sector_classifier 조사에서 실측한 문제, [24]
    # 참고) — 지금은 순차 실행 구조라 이 위험은 없음.
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()
    tatr_device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(tatr_device)
    doc_fitz = fitz.open(str(PDF_PATH))

    import pdfplumber
    pdf_pp = pdfplumber.open(str(PDF_PATH))

    records = []
    n_finance_rows_filtered = 0  # 표 전체 스킵이 아니라 "행" 단위로 걸러낸 순수 재무항목 개수([22])
    n_cid_flagged = 0
    pending_structured_calls = []  # [29] (page, table_idx, ttype, table_text, mapped_t, unmapped_labels_t)
    for r in routed:
        # [22] 사용자 요청 반영: classify_table()로 표 전체를 finance/contract 이진 판정해 통째로
        # 스킵하던 방식은 폐기 — "실적테이블"처럼 매출액/영업이익 같은 순수 재무항목과 세그먼트
        # 정보(음원/음반, 공연 등)가 한 표에 섞여 있으면 세그먼트 정보까지 같이 날아가는 문제가
        # 실측(K-Wave PDF)에서 발견됨. 이제는 표 전체는 항상 파싱하고, "행" 단위로만 순수 재무항목을
        # 걸러낸다(is_pure_financial_line_item) — ttype은 로깅/참고용으로만 남겨둠.
        ttype = classify_table(r["raw_text"])
        page_pp = pdf_pp.pages[r["page"] - 1]
        # [34] 성능 수정: detect_and_route()가 라우팅 단계에서 이미 페이지별 median_line_height를
        # 계산해 각 표 entry에 "median_line_height_pt"로 담아 돌려주는데, 여기서 그걸 안 쓰고
        # page_median_line_height(page_pp)(내부적으로 extract_words()로 페이지 전체를 다시
        # 스캔, 프로파일링 기준 페이지당 ~230ms)를 또 호출해 완전히 같은 값을 중복 계산하고
        # 있었다 — 이미 계산된 값을 그대로 재사용하도록 수정(페이지당 계산 횟수 2회->1회).
        median_lh = r["median_line_height_pt"]

        x1, y1, x2, y2 = r["bbox_px"]
        SCALE = 150 / 72
        bbox_pt = (x1 / SCALE, y1 / SCALE, x2 / SCALE, y2 / SCALE)

        if r["complexity"] == "simple":
            # 기존: pdfplumber extract_table()(quick_rows_data)을 그대로 신뢰 -> 촘촘한 표에서
            # 셀 텍스트가 뒤섞이는 버그 발견. word 좌표 기반 클러스터링으로 교체(동적 판단, 하드코딩 아님)
            parsed_rows = parse_simple_table_from_words(page_pp, bbox_pt, median_lh)
            method = "word-clustering(SIMPLE)"
        else:
            parsed_rows = parse_table_adaptive(model, processor, doc_fitz, page_pp, r["page"], bbox_pt,
                                                TATR_DPI, TATR_TOP_PAD_PT, TATR_SIDE_PAD_PT, median_lh)
            method = "TATR-grid 또는 word-clustering(컬럼 수 기반 동적 선택)"

        table_over_spaced = is_over_spaced(r["raw_text"])
        parsed_rows = [_normalize_row(row, table_over_spaced) for row in parsed_rows]

        n_before_filter = len(parsed_rows)
        parsed_rows = [row for row in parsed_rows if not is_pure_financial_line_item(row["label"])]
        n_finance_rows_filtered += n_before_filter - len(parsed_rows)
        if not parsed_rows:
            print(f"  page{r['page']} table{r['table_idx']}: {ttype} — 전 행이 순수 재무항목이라 "
                  f"필터됨(표 전체가 재무제표, DB에 이미 있음)", flush=True)
            continue

        n_mapped = 0
        table_records = []  # [25] 이 표만의 레코드 — 표 끝에서 structured metadata 호출 시 요약용
        header_fields, data_rows = detect_wide_form(parsed_rows)

        if header_fields:
            for row_idx, row in enumerate(data_rows):
                row_cells = [row["label"]] + row["cells"]
                for col_idx, cf in enumerate(header_fields):
                    if col_idx >= len(row_cells) or not row_cells[col_idx]:
                        continue
                    finalized = _finalize_value(row_cells[col_idx], cf)
                    if cf:
                        n_mapped += 1
                    if finalized["data_quality"]:
                        n_cid_flagged += 1
                    rec = {
                        "source": "table", "pdf_id": pdf_id,
                        "canonical_field": cf.key if cf else None,
                        "canonical_category": cf.category if cf else None,
                        "raw_label": finalized["value"], "cells": [],
                        "numeric_value": finalized["numeric_value"], "unit": finalized["unit"],
                        "data_quality": finalized["data_quality"],
                        "page": r["page"], "table_idx": r["table_idx"],
                        "row_record_idx": row_idx, "table_type": ttype,
                    }
                    records.append(rec)
                    table_records.append(rec)
        else:
            for row in parsed_rows:
                cf = match_canonical_field(row["label"])
                finalized_cells = [_finalize_value(c, cf) for c in row["cells"]]
                if cf:
                    n_mapped += 1
                if any(fc["data_quality"] for fc in finalized_cells):
                    n_cid_flagged += 1
                # narrow-form은 한 라벨에 여러 시점 값이 나열되는 구조라(예: 수주잔고 2024/2025/
                # 2026F) cells가 여러 개면 추세를 계산해볼 수 있음(이 표본 PDF는 재무제표류 다년치
                # 표를 스킵해서 실제로 cells>=2인 canonical 매칭 사례로 검증은 못 했음 — 함수
                # 자체는 합성 데이터로 단위테스트 완료, 아래 트렌드 필드는 다른 PDF에서 재검증 필요)
                trend = compute_trend(finalized_cells) if len(finalized_cells) >= 2 else None
                rec = {
                    "source": "table", "pdf_id": pdf_id,
                    "canonical_field": cf.key if cf else None,
                    "canonical_category": cf.category if cf else None,
                    "raw_label": row["label"], "cells": [fc["value"] for fc in finalized_cells],
                    "numeric_values": [fc["numeric_value"] for fc in finalized_cells],
                    "unit": next((fc["unit"] for fc in finalized_cells if fc["unit"]), None),
                    "trend": trend,
                    "data_quality": next((fc["data_quality"] for fc in finalized_cells if fc["data_quality"]), None),
                    "page": r["page"], "table_idx": r["table_idx"], "table_type": ttype,
                }
                records.append(rec)
                table_records.append(rec)

        if add_structured_metadata and table_records:
            # [29] 여기서 바로 호출하지 않고 나중에 병렬로 한꺼번에 디스패치 — 표 파싱(로컬 연산)이
            # 다 끝난 뒤에 API 호출만 몰아서 concurrent.futures로 돌리기 위함
            table_text = r.get("markdown") or r["raw_text"]
            mapped_t = [x for x in table_records if x["canonical_field"]]
            unmapped_labels_t = [x["raw_label"] for x in table_records if not x["canonical_field"]]
            pending_structured_calls.append((r["page"], r["table_idx"], ttype, table_text, mapped_t, unmapped_labels_t))

        form = "wide-form(헤더=필드명)" if header_fields else "narrow-form(행라벨=필드명)"
        filtered_note = f", 재무항목 {n_before_filter - len(parsed_rows)}행 필터" if n_before_filter != len(parsed_rows) else ""
        print(f"  page{r['page']} table{r['table_idx']}: {ttype}, {method}, {form}, "
              f"{len(parsed_rows)}행 파싱{filtered_note}, {n_mapped}개 canonical field 매칭", flush=True)

    pdf_pp.close()
    doc_fitz.close()

    if pending_structured_calls:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=structured_metadata_workers) as executor:
            future_to_meta = {
                executor.submit(extract_table_metadata, table_text, mapped_t, unmapped_labels_t,
                                openai_client, "gpt-4o-mini", sector):
                    (page, table_idx, ttype)
                for page, table_idx, ttype, table_text, mapped_t, unmapped_labels_t in pending_structured_calls
            }
            for future in as_completed(future_to_meta):
                page, table_idx, ttype = future_to_meta[future]
                meta = future.result()
                records.append({
                    "source": "table", "record_type": "table_metadata", "pdf_id": pdf_id,
                    "page": page, "table_idx": table_idx, "table_type": ttype,
                    **meta,
                })

    return records, n_finance_rows_filtered, n_cid_flagged


def main():
    pdf_id = PDF_PATH.stem
    records, n_finance_rows_filtered, n_cid_flagged = build_records(
        pdf_id, add_structured_metadata=ADD_STRUCTURED_METADATA)

    client = get_client(use_fake=True)  # 로컬 Redis 서버 없어 fakeredis로 검증(운영 전환 시 use_fake=False)
    cache_metadata(client, pdf_id, records)
    cached = get_all_metadata(client, pdf_id)

    # record_type="table_metadata"([25])는 표 단위 정성적 요약이라 row 단위 hit_rate 집계에서 제외
    row_records = [r for r in records if r.get("record_type") != "table_metadata"]
    table_meta_records = [r for r in records if r.get("record_type") == "table_metadata"]
    mapped = [r for r in row_records if r.get("canonical_field")]
    unmapped = [r for r in row_records if not r.get("canonical_field")]

    from canonical_field_schema import DERIVED_SIGNALS
    signals = evaluate_derived_signals(mapped, DERIVED_SIGNALS)

    result = {
        "pdf_id": pdf_id,
        "n_finance_rows_filtered": n_finance_rows_filtered,
        "n_rows_extracted": len(row_records),
        "n_table_metadata_records": len(table_meta_records),
        "n_rows_mapped_to_canonical_field": len(mapped),
        "hit_rate": round(len(mapped) / len(row_records), 4) if row_records else 0.0,
        "n_records_flagged_data_quality": n_cid_flagged,
        "canonical_field_counts": {},
        "derived_signals_triggered": signals,
        "records": records,
        "redis_cache_verification": {
            "cached_record_count": len(cached),
            "matches_source_record_count": len(cached) == len(records),
        },
    }
    from collections import Counter
    result["canonical_field_counts"] = dict(Counter(r["canonical_field"] for r in mapped))

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 결과 ===")
    print(f"총 추출 행: {len(records)}개, canonical field 매칭: {len(mapped)}개(hit rate {result['hit_rate']:.1%})")
    print(f"순수 재무항목이라 필터된 행(표 단위 아님, 행 단위): {n_finance_rows_filtered}개")
    print(f"데이터 품질 플래그(글리프 매핑 실패 등): {n_cid_flagged}건")
    print(f"Canonical field별 매칭 수: {result['canonical_field_counts']}")
    print(f"발동된 Derived Signal: {signals if signals else '없음(이 표본 PDF엔 다년치 트렌드 데이터가 canonical 매칭되지 않음 — 재무제표 스킵 설계상 예상된 결과)'}")
    print(f"Redis 캐시 검증: 저장 {len(records)}개 -> 조회 {len(cached)}개 "
          f"({'일치' if result['redis_cache_verification']['matches_source_record_count'] else '불일치!'})")
    print(f"\n--- Canonical field 매칭 레코드(LLM에 줄 구조화 데이터) ---")
    for r in mapped:
        flag = f" [!{r['data_quality']}]" if r["data_quality"] else ""
        print(f"  page{r['page']} table{r['table_idx']}: {r['canonical_field']} = {r['raw_label']!r}{flag}")
    print(f"\n--- 매칭 안 된(unmapped) 행 샘플(최대 10개) ---")
    for r in unmapped[:10]:
        print(f"  page{r['page']} table{r['table_idx']}: label='{r['raw_label']}' cells={r['cells']}")
    print(f"\n[result] saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
