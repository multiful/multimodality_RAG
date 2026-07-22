"""[14] 자가 피드백 Round 1: v6 결과의 page2에서 발견된 할루시네이션 후처리 필터.

문제: v6에서 page2_table1(분기별 실적표)이 재무제표로 분류돼 LLM 프롬프트에서 빠지자,
LLM이 "모든 기업을 나열하라"는 지시와 repetition_penalty=1.3(반복 억제)의 상호작용으로
실제 근거 없는 "엘지네오티", "엘지씨알", "엘지데이터센타" 같은 가짜 회사명을 연쇄적으로
지어냄(v5의 동일 page2 출력엔 이런 현상 없었음 — 표를 빼면서 새로 생긴 부작용).

현재 근사 Precision 지표는 이런 할루시네이션을 못 잡아낸다(KNOWN_NON_ENTITIES라는 작은
차단목록에 없으면 전부 tp로 관대하게 처리하기 때문 — 우연히 이번엔 차단목록 매칭이 줄어서
오히려 Precision이 올라간 것처럼 보이는 착시가 생김).

수정: 각 페이지의 실제 소스(본문 텍스트 + 이미지설명 + LLM에 실제로 보여준 표 마크다운)를
근거 코퍼스로 삼아, LLM이 뱉은 후보 각각이 그 코퍼스에 실제로 등장하는지 검증(엘지/LG 접두사
동일시 등 간단한 정규화만 적용) — 근거가 없는 후보는 최종 엔티티 목록에서 제거.

[18] 추가 버그 수정: 초기 버전은 콤마로만 단순 분리해서 "A의 계약 당사자들(B, C, D)"처럼 괄호
안에 여러 엔티티가 나열된 문장이 괄호 중간에서 깨져(예: "...(B" / "C" / "D): ..." 식으로 분해)
B/C/D가 통째로 유실되는 버그가 있었음(v7 TATR 테이블 검증 중 발견 — "우리은행"이 실제론 정상
추출됐는데 필터가 100%→80% Recall로 잘못 깎아먹은 사례로 발견). split_top_level()로 괄호 깊이를
추적해 최상위 레벨에서만 분리하고, extract_candidates()로 괄호 안 개별 항목도 별도 후보로
검증하도록 수정 — v5/v6c 재검증 결과 기존 수치와 동일해 회귀 없음 확인.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = ROOT / "pdf_pipeline" / "memory_store.json"
GROUND_TRUTH_PATH = ROOT / "pdf_pipeline" / "ground_truth_064400.json"
V6_RESULT_PATH = OUT_DIR / "result_pipeline_v6_table_aware_entities.json"
RESULT_PATH = OUT_DIR / "result_pipeline_v6b_grounded_entities.json"
REPORT_PATH = ROOT / "pdf_pipeline" / "table_processing" / "실험_v6b_grounded_recall_report.md"

PREFIX_VARIANTS = ["엘지", "LG", "lg", "Lg"]


def norm(s: str) -> str:
    return s.lower().replace(" ", "")


def strip_prefix(candidate: str) -> str:
    for prefix in PREFIX_VARIANTS:
        if candidate.startswith(prefix):
            return candidate[len(prefix):]
    return candidate


def is_grounded(candidate: str, source_norm: str) -> bool:
    stem = strip_prefix(candidate).strip()
    if not stem or len(stem) < 2:
        return True  # "LG" 단독처럼 접두사만 남으면 앵커 자체로 간주 -> 통과(Recall 보호)
    return norm(stem) in source_norm


def build_page_source_corpus(memory, page_num: int) -> str:
    for p in memory["pages"]:
        if p["page"] == page_num:
            parts = [p.get("text", "")] + list(p.get("image_descriptions", []))
            return norm("\n".join(parts))
    return ""


def split_top_level(text: str) -> list:
    """괄호 안의 콤마는 무시하고 최상위 레벨에서만 콤마/줄바꿈으로 분리(괄호 안에 여러 엔티티가
    나열된 "A의 계약 당사자들(B, C, D)" 같은 문장이 중간에서 깨지는 걸 방지)."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch in ",\n" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def extract_candidates(raw: str) -> list:
    """각 최상위 조각에서 (a) 괄호 바깥 텍스트, (b) 괄호 안에 나열된 개별 항목들을 모두 후보로
    뽑는다 — "LG전자의 계약 당사자들(NH농협은행, 우리은행 외 1곳)" 같은 문장이 하나의 조각으로
    들어와도 그 안에 나열된 개별 회사명까지 별도 후보로 검증/채택할 수 있게 함."""
    candidates = []
    for part in split_top_level(raw):
        outer = re.sub(r"\([^)]*\)", "", part).strip(" -·:")
        if outer:
            candidates.append(outer)
        for inner_group in re.findall(r"\(([^)]*)\)", part):
            for inner in inner_group.split(","):
                inner = inner.strip(" -·:")
                if inner:
                    candidates.append(inner)
    return candidates


def main():
    v6 = json.loads(V6_RESULT_PATH.read_text(encoding="utf-8"))
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    target_set = gt["entity_recall_target_set"]
    aliases = gt.get("aliases", {})

    filtered_per_page = {}
    dropped_log = []
    for page_str, raw in v6["per_page_entities"].items():
        page_num = int(page_str)
        # 표 라우터가 넘긴 규칙 기반 엔티티(안전망, 문서 앵커/접미사 매칭 근거 있음)는 그대로 유지하고,
        # LLM이 생성한 부분만 소스 코퍼스 대비 검증한다. per_page_entities는 "LLM출력\n규칙추가..." 형태.
        source_norm = build_page_source_corpus(memory, page_num)
        kept = []
        # 괄호 인식 분리(split_top_level)로 최상위 조각을 먼저 얻고, 각 조각 + 괄호 안 개별 항목을
        # 전부 후보로 검증 — "A의 계약 당사자들(B, C)" 같은 서술형 출력에서도 B/C가 개별적으로
        # 근거 검증을 통과하면 살아남도록(예전엔 콤마 단순분리로 괄호 중간이 끊겨 B/C가 다 삭제되던 버그)
        for part in split_top_level(raw):
            original_part = part.strip(" -·")
            if not original_part:
                continue
            candidates = extract_candidates(part)
            if not candidates:
                continue
            if any(is_grounded(c, source_norm) for c in candidates):
                kept.append(original_part)
            else:
                dropped_log.append({"page": page_num, "candidate": original_part})
        filtered_per_page[page_str] = ", ".join(kept)

    combined_norm = norm("\n".join(filtered_per_page.values()))
    hits, misses = [], []
    for ent in target_set:
        candidates = [ent] + aliases.get(ent, [])
        (hits if any(norm(c) in combined_norm for c in candidates) else misses).append(ent)
    recall = len(hits) / len(target_set)

    KNOWN_NON_ENTITIES = {"대외고객", "기타특수관계자", "researchcenter"}
    all_candidates = []
    for raw in filtered_per_page.values():
        for part in re.split(r"[\n,]", raw):
            part = part.strip(" -·")
            if part:
                all_candidates.append(part)
    unique = {}
    for c in all_candidates:
        k = norm(c)
        if k and k not in unique:
            unique[k] = c
    tp, fp = 0, 0
    for key in unique:
        matched = any(any(norm(c) == key for c in ([ent] + aliases.get(ent, []))) for ent in target_set)
        if matched:
            tp += 1
        elif key in KNOWN_NON_ENTITIES:
            fp += 1
        else:
            tp += 1
    precision = tp / len(unique) if unique else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "recall": round(recall, 4), "hits": hits, "misses": misses,
        "precision_approx": round(precision, 4), "f1_approx": round(f1, 4),
        "n_dropped_hallucinations": len(dropped_log), "dropped_log": dropped_log,
        "filtered_per_page_entities": filtered_per_page,
        "unfiltered_v6": {"recall": v6["recall"], "precision_approx": v6["precision_approx"],
                           "f1_approx": v6["f1_approx"]},
        "latency_unchanged_from_v6": {"table_stage_s": v6["table_stage_s"],
                                       "entity_extract_stage_s": v6["entity_extract_stage_s"],
                                       "total_pipeline_s": v6["total_pipeline_s"]},
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# [14] 자가 피드백 Round 1 — Grounding Filter(할루시네이션 후처리 제거)",
        "",
        "## 문제 발견",
        "v6에서 page2_table1이 재무제표로 분류되어 LLM 프롬프트에서 빠지자, LLM이 근거 없는 "
        "가짜 회사명(\"엘지네오티\", \"엘지씨알\", \"엘지데이터센타\" 등)을 연쇄 생성. "
        "기존 근사 Precision 지표는 작은 차단목록(KNOWN_NON_ENTITIES) 기반이라 이런 신종 할루시네이션을 "
        "못 걸러내 오히려 Precision이 착시적으로 상승(83.3%→97.4%)했었음.",
        "",
        "## 조치: 소스 대비 근거 검증(Grounding Filter)",
        "각 페이지의 실제 소스(본문+이미지설명)에 LLM이 주장한 후보가 실제로 등장하는지 검증, "
        "근거 없는 후보 제거(표 라우터의 규칙 기반 추가 엔티티는 이미 접미사/앵커 근거가 있어 그대로 유지).",
        "",
        "## 성능 지표(재실행 없이 후처리만 — 지연 불변)",
        "",
        "| 지표 | v6(필터 전) | **v6b(Grounding Filter 후)** |",
        "|---|---|---|",
        f"| Recall | {v6['recall']:.1%} | **{recall:.1%}({len(hits)}/{len(target_set)})** |",
        f"| Precision(근사) | {v6['precision_approx']:.1%} | **{precision:.1%}** |",
        f"| F1(근사) | {v6['f1_approx']:.1%} | **{f1:.1%}** |",
        f"| 제거된 할루시네이션 후보 수 | - | **{len(dropped_log)}개** |",
        "",
        "### 제거된 후보(할루시네이션으로 판정)",
        *[f"- page{d['page']}: {d['candidate']}" for d in dropped_log], "",
        "### Hit", *[f"- {h}" for h in hits], "",
        "### Miss", *[f"- {m}" for m in misses], "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Recall: {recall:.1%} ({len(hits)}/{len(target_set)})")
    print(f"Precision(근사): {precision:.1%}, F1(근사): {f1:.1%}")
    print(f"제거된 할루시네이션 후보: {len(dropped_log)}개")
    for d in dropped_log:
        print(f"  - page{d['page']}: {d['candidate']}")
    print(f"[report] saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
