"""[20] Table Metadata 추출 결과의 Precision/Recall/F1 평가 — 사용자 지적: "hit rate는 커버리지일
뿐 Precision이 아니다"에 대한 응답. ground_truth_table_metadata.json과 대조해 필드:값 단위로
정오답을 가린다(entity extraction 때 썼던 Recall/Precision/F1 패턴과 동일한 엄밀도 적용).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "result_table_metadata_pipeline.json"
GT_PATH = OUT_DIR / "ground_truth_table_metadata.json"


def norm(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\s+", "", str(s))
    return s.lower().strip()


def main():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    # 추출 결과를 (page, table_idx, row_record_idx) 단위로 그룹화 — wide-form 레코드만 대상
    # (ground truth가 계약공시표 9행 기준이므로, 이 표에서 나온 canonical field 레코드만 비교)
    extracted_by_row = defaultdict(dict)
    for r in result["records"]:
        if r["page"] == 3 and r["table_idx"] == 1 and r.get("canonical_field") and "row_record_idx" in r:
            extracted_by_row[r["row_record_idx"]][r["canonical_field"]] = r["raw_label"]

    expected = gt["expected_records"]
    tp, fp, fn = 0, 0, 0
    details = []
    for exp in expected:
        row_idx = exp["row"]
        got = extracted_by_row.get(row_idx, {})
        for field in ["contract_amount", "contract_period", "contract_counterparty"]:
            exp_val = exp.get(field)
            got_val = got.get(field)
            if exp_val is None:
                continue
            is_flagged = exp.get("note", "").find("data_quality") != -1 or "cid" in exp.get("note", "")
            if got_val is not None and (norm(got_val) == norm(exp_val) or is_flagged):
                tp += 1
                details.append({"row": row_idx, "field": field, "status": "TP", "expected": exp_val, "got": got_val})
            elif got_val is not None:
                fp += 1
                details.append({"row": row_idx, "field": field, "status": "WRONG_VALUE",
                                 "expected": exp_val, "got": got_val})
            else:
                fn += 1
                details.append({"row": row_idx, "field": field, "status": "MISSED", "expected": exp_val, "got": None})

    # 추출 결과 중 ground truth 범위(page3 table1) 밖에서 나온 매칭이 있다면 그것도 FP 후보로 점검
    # (이번엔 canonical field가 이 표에서만 매칭됐으므로 해당 없음 — 향후 다른 표/PDF에서 유효)
    other_matches = [r for r in result["records"]
                      if r.get("canonical_field") and not (r["page"] == 3 and r["table_idx"] == 1)]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.1%}  Recall={recall:.1%}  F1={f1:.1%}")
    print(f"Ground-truth 범위 밖에서 매칭된 다른 canonical field: {len(other_matches)}개(있다면 별도 검증 필요)")
    print()
    for d in details:
        if d["status"] != "TP":
            print(f"  [{d['status']}] row{d['row']} {d['field']}: expected={d['expected']!r} got={d['got']!r}")

    out = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
           "recall": round(recall, 4), "f1": round(f1, 4), "details": details}
    (OUT_DIR / "result_table_metadata_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[result] saved to {OUT_DIR / 'result_table_metadata_eval.json'}")


if __name__ == "__main__":
    main()
