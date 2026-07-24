# -*- coding: utf-8 -*-
"""eval_entities: 3변형 엔티티 추출 A/B — 골든셋 대조 Precision/Recall + 총 처리시간.

  V1 베이스라인          : 전 크롭 → Qwen3-VL (entities)
  V2 고도화(+분류기)      : 분류기 junk 선컷 → 나머지 Qwen3-VL
  V3 고도화+ChartQA      : junk 선컷 → chart는 DePlot(표 헤더→엔티티), 비차트는 Qwen3-VL
  V3' 개선(하이브리드)    : junk 선컷 → chart는 DePlot ∪ Qwen3-VL, 비차트는 Qwen3-VL

절차:
  --run     크롭당 분류기·VLM·DePlot를 각 1회 실측 → eval/entity_bench_raw.json
  (사람/LLM) 이미지를 직접 보고 eval/golden_entities.json 작성  ← 골든셋
  --metrics 골든셋 대조 → 변형별 micro/macro P·R + 총시간 표 + eval/entity_bench_results.json

시간 산정: 같은 이미지에 대한 동일 모델 호출은 변형 간 공유되므로 1회씩만 실측하고
변형 정의에 따라 합산한다(캐시 미사용, 전부 라이브 측정)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import figure_classifier as fc
import chartqa_deplot as dp
from s2_image_pipeline import PROMPT_V2, normalize_vlm
from bench_variants import collect_sample

CFG = common.CONFIG
logger = common.get_logger("eval_entities")
RAW = CFG["EVAL_DIR"] / "entity_bench_raw.json"
GOLD = CFG["EVAL_DIR"] / "golden_entities.json"
OUT = CFG["EVAL_DIR"] / "entity_bench_results.json"


# ---------------------------------------------------------------- 유틸

def _numberish(s: str) -> bool:
    t = s.strip().replace(",", "").replace("%", "").replace("$", "").replace("(", "") \
        .replace(")", "").replace("-", "").replace(".", "").replace("~", "")
    return t == "" or t.isdigit()


import re as _re
_AXIS_RE = _re.compile(r"^\d+[QqHh]\d+$|^\d{1,2}[.\-]\d{1,2}$|^20\d\d[Ff]?$|^\d{4}$")


def deplot_entities(table: str, cap: int = 12) -> list[str]:
    """DePlot 표에서 비수치 셀(범례·계열명 후보)을 엔티티로 수집 (필터 없음, V3 원시)."""
    ents: list[str] = []
    for ln in table.splitlines():
        for c in ln.split("|"):
            c = c.strip()
            if not c or c.upper() == "TITLE" or _numberish(c):
                continue
            if len(c) >= 2 and c not in ents:
                ents.append(c)
    return ents[:cap]


def _is_gibberish(s: str) -> bool:
    """DePlot의 한글 오독 산출물(바이트토큰·비한글/비라틴 문자 위주) 판별."""
    if "<0x" in s or "▁" in s:
        return True
    good = sum(1 for ch in s if ("가" <= ch <= "힣") or ch.isascii() and ch.isalnum())
    return good < max(2, len(s) * 0.5)


def clean_deplot_entities(table: str, cap: int = 12) -> list[str]:
    """개선판: 축라벨(1Q21·2026F 등)과 gibberish(한글 오독)를 제거한 계열명만."""
    ents: list[str] = []
    for c in deplot_entities(table, cap=99):
        if _AXIS_RE.match(c) or _is_gibberish(c):
            continue
        if c not in ents:
            ents.append(c)
    return ents[:cap]


def norm(s: str) -> str:
    return "".join(str(s).split()).casefold()


def match(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na == nb or (len(na) >= 2 and na in nb) or (len(nb) >= 2 and nb in na)


# ---------------------------------------------------------------- --run

def run() -> None:
    common.ensure_dirs()
    if not common.ollama_alive():
        logger.info("Ollama 미실행 — 중단")
        return
    logger.info("DePlot 로딩…")
    dp.available()
    sample = collect_sample()
    logger.info(f"표본 {len(sample)}건")

    rows = []
    for iid, bt, crop, cap in sample:
        t = time.time()
        clf = fc.classify(crop)
        clf_dt = time.time() - t

        t = time.time()
        res = common.ollama_chat(
            CFG["VLM_MODEL"],
            PROMPT_V2.format(caption=cap or "없음", doc_title="", category="industry"),
            images=[str(crop)], num_ctx=CFG["VLM_NUM_CTX"],
            img_max_edge=CFG["VLM_MAX_EDGE"], think=CFG["VLM_THINK"])
        vlm_dt = time.time() - t
        vlm = normalize_vlm(res, bt) if (res and not res.get("_parse_error")) else \
            {"entities": [], "useful": bt == "chart", "type": bt}

        de = {"table": "", "rows": 0, "seconds": 0.0}
        if bt == "chart":
            de = dp.extract_table(crop)

        rows.append({
            "iid": iid, "bt": bt, "crop": str(crop), "cap": cap,
            "clf_label": clf["label"] if clf else "?",
            "route": clf["route"] if clf else "other",
            "clf_dt": round(clf_dt, 3),
            "vlm_dt": round(vlm_dt, 2), "vlm_entities": vlm.get("entities", []),
            "vlm_type": vlm.get("type"), "vlm_useful": vlm.get("useful"),
            "de_dt": de["seconds"], "de_table": de["table"],
            "de_entities": deplot_entities(de["table"]),
        })
        logger.info(f"  {iid} [{bt}/{rows[-1]['route']}] clf {clf_dt*1000:.0f}ms · "
                    f"vlm {vlm_dt:.0f}s · deplot {de['seconds']:.0f}s")

    RAW.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"원시결과 저장: {RAW}")
    logger.info(f"다음: {GOLD} 에 골든셋 작성 후 --metrics 실행")


# ---------------------------------------------------------------- --metrics

def variant_preds(r: dict, variant: str) -> list[str] | None:
    """변형별 이미지 1장의 예측 엔티티. None = 이 변형에선 처리 안 함(junk 컷)."""
    junk = r["route"] == "junk"
    chart = r["bt"] == "chart"
    if variant == "V1":
        return r["vlm_entities"]
    if junk:
        return None
    if variant == "V2":
        return r["vlm_entities"]
    if variant == "V3":
        return r["de_entities"] if chart else r["vlm_entities"]
    if variant == "V3H":  # 하이브리드(원시 union — DePlot gibberish 미필터)
        return (r["vlm_entities"] + [e for e in r["de_entities"]
                                     if not any(match(e, v) for v in r["vlm_entities"])]) \
            if chart else r["vlm_entities"]
    if variant == "V4":  # 개선판: 축라벨·gibberish 필터한 DePlot ∪ VLM
        if not chart:
            return r["vlm_entities"]
        clean = clean_deplot_entities(r.get("de_table", ""))
        return r["vlm_entities"] + [e for e in clean
                                    if not any(match(e, v) for v in r["vlm_entities"])]
    raise ValueError(variant)


def variant_time(rows: list[dict], variant: str) -> float:
    tot = 0.0
    for r in rows:
        junk = r["route"] == "junk"
        chart = r["bt"] == "chart"
        if variant == "V1":
            tot += r["vlm_dt"]
            continue
        tot += r["clf_dt"]
        if junk:
            continue
        if variant == "V2":
            tot += r["vlm_dt"]
        elif variant == "V3":
            tot += r["de_dt"] if chart else r["vlm_dt"]
        elif variant in ("V3H", "V4"):
            tot += (r["de_dt"] + r["vlm_dt"]) if chart else r["vlm_dt"]
    return tot


def metrics() -> None:
    if not RAW.exists() or not GOLD.exists():
        logger.info(f"필요 파일 없음: {RAW.name} / {GOLD.name}")
        return
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    gold = json.loads(GOLD.read_text(encoding="utf-8"))

    names = {"V1": "V1 베이스라인", "V2": "V2 고도화(+분류기)",
             "V3": "V3 고도화+ChartQA", "V3H": "V3' 하이브리드(원시)",
             "V4": "V4 개선(필터 하이브리드)"}
    results = {}
    for v in names:
        tp_pred = n_pred = tp_gold = n_gold = 0
        per_img = []
        for r in rows:
            g = gold.get(r["iid"], {}).get("entities", [])
            preds = variant_preds(r, v)
            p_list = preds or []
            m_pred = sum(1 for p in p_list if any(match(p, x) for x in g))
            m_gold = sum(1 for x in g if any(match(x, p) for p in p_list))
            tp_pred += m_pred; n_pred += len(p_list)
            tp_gold += m_gold; n_gold += len(g)
            per_img.append({"iid": r["iid"], "processed": preds is not None,
                            "n_pred": len(p_list), "n_gold": len(g),
                            "matched_pred": m_pred, "matched_gold": m_gold})
        prec = tp_pred / n_pred if n_pred else 0.0
        rec = tp_gold / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results[v] = {"name": names[v], "precision": round(prec, 4),
                      "recall": round(rec, 4), "f1": round(f1, 4),
                      "n_pred": n_pred, "n_gold": n_gold,
                      "time_s": round(variant_time(rows, v), 1), "per_image": per_img}

    p = logger.info
    p("=" * 68)
    p(f"{'변형':24s}{'시간(s)':>9s}{'Precision':>11s}{'Recall':>9s}{'F1':>7s}")
    for v in names:
        r = results[v]
        p(f"{r['name']:24s}{r['time_s']:>9.1f}{r['precision']*100:>10.1f}%"
          f"{r['recall']*100:>8.1f}%{r['f1']*100:>6.1f}%")
    p(f"(골든 엔티티 {results['V1']['n_gold']}개 / 표본 {len(rows)}장)")
    p("=" * 68)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"결과 저장: {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description="엔티티 A/B: --run 후 골든셋 작성, --metrics")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.metrics:
        metrics()
    if not a.run and not a.metrics:
        print("사용법: --run (수집) → golden_entities.json 작성 → --metrics")


if __name__ == "__main__":
    main()
