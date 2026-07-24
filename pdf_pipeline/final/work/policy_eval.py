# -*- coding: utf-8 -*-
"""[재일] 라우팅 정책 개선안 평가 — results_fusion_routing_ab.json의 질의별 실측 점수를 그대로
재조합해 "분류 결과에 따라 다른 검색기를 쓰는 정책"의 성능을 계산한다(새 API 호출 없음, 같은
실측치를 쓰므로 비교가 공정하다).

현행: abstract -> MQE/HyDE, keyword_specific -> hybrid(fusion="rrf")  [index_text.py:387]
개선안: keyword 경로의 퓨전만 rrf -> 선형가중(0.7/0.3)으로 교체하거나 dense-only로 교체.
지연은 질의별 실측 지연으로 계산하되, keyword 경로는 분류 LLM 호출 비용을 그대로 부담한다고
보고 route 실측 지연에서 rrf 검색시간을 빼 분류 오버헤드를 추정해 더한다."""
import json
from pathlib import Path
PP = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
R = json.loads((PP/"final"/"results_fusion_routing_ab.json").read_text(encoding="utf-8"))
G = json.loads((PP/"final"/"golden_set_construct_routing.json").read_text(encoding="utf-8"))
OUT = PP/"final"/"results_policy_eval.json"
pq, cls = R["per_query"], R["classifier"]
qtype = {q["id"]: q["type"] for q in G["queries"]}

POLICIES = {
    "현행 route(abs=MQE/HyDE, kw=rrf)":      ("route(gpt-4o)",   "route(gpt-4o)"),
    "개선A(abs=route, kw=linear0.7/0.3)":    ("route(gpt-4o)",   "linear_0.7/0.3"),
    "개선B(abs=route, kw=dense_only)":       ("route(gpt-4o)",   "dense_only"),
    "무라우팅 고정 linear0.7/0.3":            ("linear_0.7/0.3",  "linear_0.7/0.3"),
    "무라우팅 고정 dense_only":               ("dense_only",      "dense_only"),
}

def agg(rows):
    n = len(rows)
    return {"ndcg": round(sum(r["ndcg"] for r in rows)/n, 3),
            "mrr": round(sum(r["mrr"] for r in rows)/n, 3),
            "recall": round(sum(r["hit"] for r in rows)/n, 3)}

res = {}
for pname, (abs_m, kw_m) in POLICIES.items():
    by_type, all_rows, lat = {}, [], []
    for qid, t in qtype.items():
        is_abs = "abstract" in (cls[qid] or "")
        m = abs_m if is_abs else kw_m
        s = dict(pq[qid][m])
        if not is_abs and m != "route(gpt-4o)":
            # 분류 오버헤드 추정: route 실측지연 - 그 경로가 실제로 쓴 rrf 검색지연
            overhead = max(0.0, pq[qid]["route(gpt-4o)"]["latency_s"] - pq[qid]["rrf_k60"]["latency_s"])
            s["latency_s"] = round(s["latency_s"] + overhead, 3)
        by_type.setdefault(t, []).append(s); all_rows.append(s); lat.append(s["latency_s"])
    row = {t: agg(v) for t, v in by_type.items()}
    row["ALL"] = agg(all_rows); row["latency_s_mean"] = round(sum(lat)/len(lat), 3)
    res[pname] = row

print(f"{'정책':36} {'키워드형':>18} {'하이브형':>18} {'추상형':>18} {'ALL':>18}  지연")
for pname, row in res.items():
    cells = []
    for t in ["키워드형", "하이브리드형", "추상형", "ALL"]:
        r = row.get(t, {})
        cells.append(f"{r.get('ndcg',0):.3f}/{r.get('recall',0):.2f}".rjust(18))
    print(f"{pname:36} " + " ".join(cells) + f"  {row['latency_s_mean']:.2f}s")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n(수치는 ndcg@8/recall@8) -> {OUT.name} 저장")
