"""Recall / Precision / F1 / Latency를 한 번에 집계해서 recall_report.md를 갱신한다.

- Recall, Precision: extracted_entities.json(페이지별 원본 추출) vs ground_truth_064400.json
- Latency: memory_store.json["timing"](분류+텍스트+표+이미지) + entity_extract_timing.json(엔티티 추출)
- 랭킹지표(Recall@K/MRR/nDCG)는 이번 엔티티 추출 baseline에는 적용하지 않음 — 결과가 confidence/order
  없는 集合이라 랭킹 개념이 성립하지 않음. 이후 검색·리랭킹 단계(BM25+Dense+CrossEncoder, docs/PRD_pdf_pipeline.md)
  에서 필요해지면 그때 추가.
"""

import json
import re
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_064400.json"
ENTITIES_PATH = OUT_DIR / "extracted_entities.json"
MEMORY_PATH = OUT_DIR / "memory_store.json"
ENTITY_TIMING_PATH = OUT_DIR / "entity_extract_timing.json"
REPORT_PATH = OUT_DIR / "recall_report.md"

# 육안 검수로 확인한, 엔티티가 아닌데 뽑힌 항목들(baseline 규모가 작아 수작업 큐레이션).
# 더 큰 문서셋으로 갈수록 이 블록리스트 방식 대신 실제 기업 DB/API 대조로 바꿔야 함.
KNOWN_NON_ENTITIES = {"대외고객", "기타특수관계자", "기타 특수관계자", "researchcenter", "research center"}


def norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("주식회사", "")


def split_candidates(raw: str):
    parts = re.split(r"[\n,]", raw)
    out = []
    for p in parts:
        p = re.sub(r"\([^)]*\)", "", p).strip(" -·")
        if p:
            out.append(p)
    return out


def main():
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    per_page = json.loads(ENTITIES_PATH.read_text(encoding="utf-8"))
    target_set = gt["entity_recall_target_set"]
    aliases = gt.get("aliases", {})

    combined_raw = "\n".join(per_page.values())
    combined_norm = norm(combined_raw)

    # ---- Recall ----
    hits, misses = [], []
    for ent in target_set:
        candidates = [ent] + aliases.get(ent, [])
        if any(norm(c) in combined_norm for c in candidates):
            hits.append(ent)
        else:
            misses.append(ent)
    recall = len(hits) / len(target_set)

    # ---- Precision ----
    # 1) 모든 페이지 출력에서 개별 후보 추출 후 정규화 dedup
    all_candidates = []
    for raw in per_page.values():
        all_candidates += split_candidates(raw)
    unique_candidates = {}
    for c in all_candidates:
        key = norm(c)
        if key and key not in unique_candidates:
            unique_candidates[key] = c

    tp_items, fp_items, imprecise_items = [], [], []
    for key, original in unique_candidates.items():
        matched_gt = None
        for ent in target_set:
            candidates = [ent] + aliases.get(ent, [])
            if any(norm(c) == key for c in candidates):
                matched_gt = ent
                break
        if matched_gt:
            tp_items.append((original, matched_gt))
        elif key in KNOWN_NON_ENTITIES:
            fp_items.append(original)
        else:
            # 정확한 target 엔티티는 아니지만 실재하는 기업명으로 보이는 경우
            # (예: "네이버" — 네이버클라우드의 모기업). precision 계산엔 TP로 넣되 별도 표기.
            imprecise_items.append(original)

    precision_strict = len(tp_items) / len(unique_candidates) if unique_candidates else 0.0
    precision_lenient = (len(tp_items) + len(imprecise_items)) / len(unique_candidates) if unique_candidates else 0.0

    def f1(p, r):
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    f1_strict = f1(precision_strict, recall)
    f1_lenient = f1(precision_lenient, recall)

    # ---- Latency ----
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    timing = memory.get("timing", {})
    entity_timing = json.loads(ENTITY_TIMING_PATH.read_text(encoding="utf-8")) if ENTITY_TIMING_PATH.exists() else None

    page_rows = []
    if timing:
        et_by_page = {p["page"]: p for p in entity_timing["pages"]} if entity_timing else {}
        for p in timing["pages"]:
            ee = et_by_page.get(p["page"], {}).get("entity_extract_s", None)
            page_rows.append(
                f"| {p['page']} | {p['classify_s']} | {p['text_extract_s']} | {p['table_extract_s']} | "
                f"{p['image_vlm_s']} | {ee if ee is not None else '-'} | "
                f"{round(p['page_total_s'] + (ee or 0), 3)} |"
            )

    total_s = None
    if timing and entity_timing:
        total_s = round(
            timing["model_load_s"] + sum(p["page_total_s"] for p in timing["pages"])
            + entity_timing["model_load_s"] + entity_timing["total_entity_extract_s"],
            2,
        )

    # ---- 리포트 작성 ----
    lines = [
        "# Baseline 평가 리포트 (Recall / Precision / F1 / Latency)",
        "",
        "- 대상 문서: `20260721_company_279243000.pdf` (교보증권 LG CNS(064400) 기업분석, 2026-07-21)",
        "- 파이프라인: 페이지 분류 → 페이지별 텍스트/표/이미지 추출(run_baseline.py) → "
        "페이지별 엔티티 추출(extract_entities_and_eval.py) → 채점(evaluate.py)",
        "",
        "## 지표 요약",
        "",
        "| 지표 | 값 | 비고 |",
        "|---|---|---|",
        f"| Recall | **{recall:.1%}** ({len(hits)}/{len(target_set)}) | LG/엘지 등 별칭 정규화 반영 |",
        f"| Precision (strict) | **{precision_strict:.1%}** ({len(tp_items)}/{len(unique_candidates)}) | "
        "정답 엔티티와 정확히 매칭된 것만 TP |",
        f"| Precision (lenient) | {precision_lenient:.1%} ({len(tp_items)+len(imprecise_items)}/{len(unique_candidates)}) | "
        "실재하지만 목표 엔티티와 다른 회사(예: '네이버')도 TP로 인정 |",
        f"| F1 (strict) | **{f1_strict:.1%}** | |",
        f"| F1 (lenient) | {f1_lenient:.1%} | |",
        f"| 총 처리 시간 | {total_s if total_s is not None else '(측정 안 됨)'}s | 모델 로딩 2회(별도 프로세스) 포함, 6페이지 |",
        "",
        "**랭킹지표(Recall@K/MRR/nDCG) 미적용 사유**: 지금 엔티티 추출 결과는 confidence나 순서가 없는 "
        "集合이라 랭킹 개념이 성립하지 않음. 검색·리랭킹 단계(BM25+Dense+CrossEncoder)가 붙으면 그때 필요.",
        "",
        "## Precision 상세",
        "",
        "### TP (정답과 정확히 매칭)",
        *[f"- {orig} → {gt_name}" for orig, gt_name in tp_items],
        "",
        "### 애매(실재 기업이나 목표 엔티티와 불일치, lenient에서만 TP)",
        *([f"- {x}" for x in imprecise_items] or ["- (없음)"]),
        "",
        "### FP (엔티티 아님)",
        *([f"- {x}" for x in fp_items] or ["- (없음)"]),
        "",
        "## Recall 상세",
        "",
        "### Hit", *[f"- {h}" for h in hits], "",
        "### Miss", *[f"- {m}" for m in misses], "",
        "## Latency 상세 (페이지별, 초)",
        "",
        "| page | classify | text | table | image_vlm | entity_extract | page_total |",
        "|---|---|---|---|---|---|---|",
        *page_rows,
        "",
        f"- 모델 로딩(1회차, run_baseline.py): {timing.get('model_load_s', '-')}s",
        f"- 모델 로딩(2회차, extract_entities_and_eval.py): {entity_timing['model_load_s'] if entity_timing else '-'}s "
        "(별도 프로세스라 2번 로딩됨 — 하나의 서비스 프로세스로 합치면 이 시간은 1회로 절약 가능, 고도화 포인트)",
        "",
        "## 페이지별 추출 원본",
    ]
    for pg, ents in per_page.items():
        lines += [f"### page {pg}", "```", ents or "(내용 없음)", "```"]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Recall: {recall:.1%} | Precision(strict): {precision_strict:.1%} "
          f"| Precision(lenient): {precision_lenient:.1%} | F1(strict): {f1_strict:.1%}")
    print(f"Total latency: {total_s}s")
    print(f"[report] saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
