# -*- coding: utf-8 -*-
"""bench_variants: 3변형 A/B 벤치.

  V1 베이스라인            = 모든 chart/image 크롭 → Qwen3-VL (OCR+요약+유용성)
  V2 +분류기(고도화)       = 분류기로 junk 선컷 → 나머지 Qwen3-VL
  V3 +분류기+ChartQA       = 분류기 선컷 → chart는 DePlot(수치표), 비차트는 Qwen3-VL

크롭당 분류기/VLM/DePlot를 한 번씩만 계산해 세 변형 지표를 조립한다.
차트별로 [이미지 | Qwen3-VL ocr_text | DePlot 표]를 HTML로 나란히 → 한글 처리 육안 비교.
콘솔 요약 + eval/bench_variants.html 생성."""
from __future__ import annotations

import base64, html as html_mod, time
from pathlib import Path

import common
import figure_classifier as fc
import chartqa_deplot as dp
from s2_image_pipeline import find_crop, get_caption, rule_filter, PROMPT_V2, normalize_vlm

CFG = common.CONFIG
logger = common.get_logger("bench_variants")

N_CHART = 8   # 비교할 차트 표본 수
N_JUNK = 4    # 분류기 junk(로고/사진) 표본 수


def collect_sample():
    """clean 차트(industry_15 등) + junk(hana_17 로고/사진) 표본 수집."""
    docs = dict(common.find_parsed_docs())
    charts, junk = [], []
    order = ["industry_15", "industry_hana_17", "industry_14", "industry_hana_13"]
    for did in order + [d for d in docs if d not in order]:
        mdir = docs.get(did)
        if not mdir:
            continue
        content = common.load_content_list(mdir)
        cnt = {}
        for it in content:
            bt = it.get("type")
            if bt not in ("chart", "image"):
                continue
            page = int(it.get("page_idx", 0)) + 1
            k = f"{page}:{bt}"; cnt[k] = cnt.get(k, 0) + 1
            iid = f"{did}_p{page}_{bt}{cnt[k]}"
            crop = find_crop(Path(mdir), it)
            if crop is None or rule_filter(crop, bt):
                continue
            item = (iid, bt, crop, get_caption(it))
            if bt == "chart" and len(charts) < N_CHART:
                charts.append(item)
            elif bt == "image" and len(junk) < N_JUNK:
                r = fc.classify(crop)
                if r and r["route"] == "junk":
                    junk.append(item)
        if len(charts) >= N_CHART and len(junk) >= N_JUNK:
            break
    return charts + junk


def run():
    common.ensure_dirs()
    if not common.ollama_alive():
        logger.info("Ollama 미실행 — 중단"); return
    logger.info("DePlot 로딩…")
    dp.available()

    sample = collect_sample()
    logger.info(f"표본 {len(sample)}건 수집 (차트+junk)")

    rows = []
    for iid, bt, crop, cap in sample:
        clf = fc.classify(crop)
        route = clf["route"] if clf else "other"
        # Qwen3-VL (V1은 항상, V2/V3은 필요시 재사용)
        t0 = time.time()
        res = common.ollama_chat(CFG["VLM_MODEL"], PROMPT_V2.format(
            caption=cap or "없음", doc_title="", category="industry"),
            images=[str(crop)], num_ctx=CFG["VLM_NUM_CTX"],
            img_max_edge=CFG["VLM_MAX_EDGE"], think=CFG["VLM_THINK"])
        vlm_dt = time.time() - t0
        vlm = normalize_vlm(res, bt) if (res and not res.get("_parse_error")) else \
            {"useful": bt == "chart", "ocr_text": "", "summary": "(파싱실패)"}
        # DePlot (차트만)
        de = dp.extract_table(crop) if bt == "chart" else {"table": "", "rows": 0, "seconds": 0.0}
        rows.append({"iid": iid, "bt": bt, "crop": str(crop), "cap": cap,
                     "clf_label": clf["label"] if clf else "?", "route": route,
                     "vlm_useful": vlm.get("useful"), "vlm_ocr": vlm.get("ocr_text", ""),
                     "vlm_summary": vlm.get("summary", ""), "vlm_dt": vlm_dt,
                     "de_table": de["table"], "de_rows": de["rows"], "de_dt": de["seconds"]})
        logger.info(f"  {iid} [{bt}/{route}] vlm {vlm_dt:.0f}s · deplot {de['seconds']:.0f}s")

    _report(rows)


def _agg(rows):
    n = len(rows)
    charts = [r for r in rows if r["bt"] == "chart"]
    junk = [r for r in rows if r["route"] == "junk"]
    # V1: 전건 VLM
    v1_vlm = n
    v1_time = sum(r["vlm_dt"] for r in rows)
    # V2: junk 제외 VLM
    v2_vlm = n - len(junk)
    v2_time = sum(r["vlm_dt"] for r in rows if r["route"] != "junk")
    # V3: junk 제외, chart→DePlot / 비차트→VLM
    v3_vlm = sum(1 for r in rows if r["route"] != "junk" and r["bt"] != "chart")
    v3_de = sum(1 for r in rows if r["route"] != "junk" and r["bt"] == "chart")
    v3_time = (sum(r["vlm_dt"] for r in rows if r["route"] != "junk" and r["bt"] != "chart")
               + sum(r["de_dt"] for r in rows if r["route"] != "junk" and r["bt"] == "chart"))
    return {
        "n": n, "charts": len(charts), "junk": len(junk),
        "V1": {"vlm": v1_vlm, "deplot": 0, "time": v1_time},
        "V2": {"vlm": v2_vlm, "deplot": 0, "time": v2_time},
        "V3": {"vlm": v3_vlm, "deplot": v3_de, "time": v3_time},
    }


def _report(rows):
    a = _agg(rows)
    p = logger.info
    p("=" * 60)
    p(f"표본 {a['n']}건 (차트 {a['charts']}, junk {a['junk']})")
    p(f"{'변형':22s}{'VLM호출':>8s}{'DePlot':>8s}{'총시간(s)':>10s}")
    for v in ("V1", "V2", "V3"):
        d = a[v]
        name = {"V1": "V1 베이스라인", "V2": "V2 +분류기", "V3": "V3 +분류기+ChartQA"}[v]
        p(f"{name:22s}{d['vlm']:>8d}{d['deplot']:>8d}{d['time']:>10.0f}")
    p("* 주의: 이 venv torch=CPU라 DePlot ~25s/장. GPU면 ~2-4s로 급감(인프라 이슈).")
    p("=" * 60)
    _html(rows, a)


def _html(rows, a):
    esc = html_mod.escape
    def img(p):
        f = Path(p)
        if not f.is_file(): return "(없음)"
        b = base64.b64encode(f.read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{b}">'
    cards = []
    for r in rows:
        de = esc(r["de_table"][:600]) or "(차트 아님/없음)"
        cards.append(f"""<div class="card">
  <div class="imgbox">{img(r['crop'])}</div>
  <div class="meta"><b>{esc(r['iid'])}</b> · {esc(r['bt'])} · 분류기:{esc(r['clf_label'])}({esc(r['route'])})</div>
  <div class="cap">{esc(r['cap'] or '')}</div>
  <div class="two">
    <div class="col"><div class="h">V1/V2 · Qwen3-VL ocr_text ({r['vlm_dt']:.0f}s)</div>
      <pre>{esc((r['vlm_ocr'] or '')[:600])}</pre>
      <div class="sub">요약: {esc(r['vlm_summary'] or '')}</div></div>
    <div class="col"><div class="h">V3 · DePlot 표 ({r['de_dt']:.0f}s, {r['de_rows']}행)</div>
      <pre>{de}</pre></div>
  </div></div>""")
    trows = "".join(
        f"<tr><td>{n}</td><td>{a[v]['vlm']}</td><td>{a[v]['deplot']}</td><td>{a[v]['time']:.0f}s</td></tr>"
        for v, n in (("V1", "V1 베이스라인"), ("V2", "V2 +분류기"), ("V3", "V3 +분류기+ChartQA")))
    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>3변형 A/B 벤치</title><style>
:root{{color-scheme:dark}} body{{background:#14161a;color:#dfe3ea;font-family:'Malgun Gothic',sans-serif;margin:20px}}
h1{{font-size:20px}} table{{border-collapse:collapse;margin:12px 0}} th,td{{border:1px solid #2c313b;padding:6px 14px;text-align:center}}
.card{{background:#1d2027;border:1px solid #2c313b;border-radius:10px;padding:14px;margin:14px 0}}
.imgbox{{background:#fff;border-radius:6px;text-align:center;padding:6px}} .imgbox img{{max-width:100%;max-height:300px}}
.meta{{margin:8px 0 2px}} .cap{{color:#9fb2d8;font-size:12px;margin-bottom:8px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} @media(max-width:800px){{.two{{grid-template-columns:1fr}}}}
.col .h{{font-size:12px;color:#8fd0a0;margin-bottom:4px}} .col .h:last-of-type{{color:#8fb4d0}}
pre{{white-space:pre-wrap;background:#0f1114;border-radius:6px;padding:8px;max-height:260px;overflow:auto;font-size:12px;color:#cdd6e6}}
.sub{{font-size:12px;color:#9aa3b2;margin-top:4px}}
</style></head><body>
<h1>3변형 A/B 벤치 — 표본 {a['n']}건 (차트 {a['charts']}, junk {a['junk']})</h1>
<table><tr><th>변형</th><th>VLM 호출</th><th>DePlot 호출</th><th>총 시간</th></tr>{trows}</table>
<p style="color:#9aa3b2;font-size:12px">주의: 이 venv torch=CPU → DePlot ~25s/장. GPU면 ~2-4s. 아래는 차트별 Qwen3-VL vs DePlot 출력(한글 처리 비교).</p>
{''.join(cards)}
</body></html>"""
    out = CFG["EVAL_DIR"] / "bench_variants.html"
    out.write_text(doc, encoding="utf-8")
    logger.info(f"HTML: {out}")


if __name__ == "__main__":
    run()
