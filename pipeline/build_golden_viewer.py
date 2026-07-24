# -*- coding: utf-8 -*-
"""build_golden_viewer: LLM이 만든 골든셋(엔티티 정답)을 이미지와 나란히 보여주는 뷰어.

golden_entities.json + entity_bench_raw.json(이미지경로·캡션)을 합쳐, 각 이미지 카드에
[이미지 | 캡션·유형 | 정답 엔티티]를 렌더. 인쇄용 밝은테마. PDF 변환 대비."""
from __future__ import annotations
import base64, html as H, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
CFG = common.CONFIG

gold = json.loads((CFG["EVAL_DIR"] / "golden_entities.json").read_text(encoding="utf-8"))
meta = gold.get("_meta", {})
raw = {r["iid"]: r for r in json.loads((CFG["EVAL_DIR"] / "entity_bench_raw.json").read_text(encoding="utf-8"))}

def esc(s): return H.escape(str(s))

def img_tag(p):
    f = Path(p)
    if not f.is_file(): return '<div class="noimg">이미지 없음</div>'
    b = base64.b64encode(f.read_bytes()).decode()
    return f'<img src="data:image/jpeg;base64,{b}">'

order = [k for k in gold if not k.startswith("_")]
total_ent = sum(len(gold[k]["entities"]) for k in order)

cards = []
for i, iid in enumerate(order, 1):
    g = gold[iid]["entities"]
    r = raw.get(iid, {})
    cap = r.get("cap", "")
    bt = r.get("bt", "")
    route = r.get("route", "")
    clf = r.get("clf_label", "")
    crop = r.get("crop", "")
    if g:
        chips = "".join(f'<span class="chip">{esc(e)}</span>' for e in g)
        ent_html = f'<div class="chips">{chips}</div><div class="cnt">{len(g)}개</div>'
    else:
        ent_html = '<div class="empty">엔티티 없음 <span>(아이콘·조감도 등 분석 대상 아님)</span></div>'
    badge = "junk" if route == "junk" else bt
    bclass = "bjunk" if route == "junk" else ("bchart" if bt == "chart" else "bimg")
    cards.append(f"""<div class="card">
  <div class="imgbox">{img_tag(crop)}</div>
  <div class="info">
    <div class="head"><span class="num">{i:02d}</span><span class="iid">{esc(iid)}</span>
      <span class="badge {bclass}">{esc(badge)}</span></div>
    <div class="cap">{esc(cap) or '<i>캡션 없음</i>'}</div>
    <div class="lbl">정답 엔티티 (LLM 판독)</div>
    {ent_html}
  </div>
</div>""")

BLUE = "#2a78d6"
HTML = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>엔티티 정답(골든셋) — LLM 판독</title><style>
:root{{ color-scheme:light }}
*{{box-sizing:border-box}}
body{{background:#fff;color:#0b0b0b;margin:0;padding:28px 22px;
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;line-height:1.55}}
.wrap{{max-width:860px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}} .sub{{color:#7a7873;font-size:13px}}
.note{{background:#f4f7fb;border-left:3px solid {BLUE};padding:10px 14px;margin:14px 0;font-size:13px;color:#333;border-radius:0 6px 6px 0}}
.card{{display:grid;grid-template-columns:230px 1fr;gap:16px;border:1px solid rgba(11,11,11,.12);
  border-radius:12px;padding:14px;margin:12px 0;break-inside:avoid;page-break-inside:avoid}}
.imgbox{{background:#fafafa;border:1px solid rgba(11,11,11,.08);border-radius:8px;
  display:flex;align-items:center;justify-content:center;min-height:150px;overflow:hidden}}
.imgbox img{{max-width:100%;max-height:210px;display:block}}
.noimg{{color:#aaa;padding:40px}}
.head{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}}
.num{{background:{BLUE};color:#fff;border-radius:6px;font-size:12px;font-weight:700;padding:1px 7px}}
.iid{{font-size:12.5px;color:#52514e;font-family:ui-monospace,monospace}}
.badge{{font-size:11px;font-weight:700;padding:1px 8px;border-radius:10px;color:#fff}}
.bchart{{background:#008300}} .bimg{{background:#4a3aa7}} .bjunk{{background:#eb6834}}
.cap{{font-size:14px;font-weight:600;margin:2px 0 10px}}
.lbl{{font-size:11px;color:#898781;text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px}}
.chip{{background:#eaf3ff;color:#184f95;border:1px solid #cde2fb;border-radius:14px;
  padding:3px 11px;font-size:13px;font-weight:600}}
.cnt{{color:#898781;font-size:12px;margin-top:6px}}
.empty{{color:#a08b6a;font-size:13px;background:#faf6ee;border:1px dashed #e3d3b0;border-radius:8px;padding:8px 12px}}
.empty span{{color:#b7a988;font-weight:400}}
@page{{ size:A4; margin:12mm }}
@media print{{ body{{padding:0;-webkit-print-color-adjust:exact;print-color-adjust:exact}} }}
</style></head><body><div class="wrap">
<h1>엔티티 정답(골든셋) — LLM 판독</h1>
<div class="sub">엔티티 추출 A/B 평가의 정답지 · 이미지 {len(order)}장 · 정답 엔티티 {total_ent}개 · {esc(meta.get('date',''))}</div>
<div class="note"><b>정의</b>: {esc(meta.get('definition',''))}<br>
<b>작성</b>: {esc(meta.get('annotator',''))} — 아래 각 이미지를 직접 판독해 기업명·티커·지표명을 정답으로 기록했습니다. 로고 그리드는 실재 16개 기업 포함, 아이콘·조감도는 분석 엔티티가 없어 빈칸.</div>
{''.join(cards)}
<div class="sub" style="margin-top:20px">원본: eval/golden_entities.json · 생성 {esc(common.now_iso())}</div>
</div></body></html>"""

out = CFG["EVAL_DIR"] / "golden_entities.html"
out.write_text(HTML, encoding="utf-8")
print(f"골든셋 뷰어 생성: {out}")
