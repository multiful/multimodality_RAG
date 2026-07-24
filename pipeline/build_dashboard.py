# -*- coding: utf-8 -*-
"""build_dashboard: A/B 실험 산출물을 한 장의 자립형 HTML 대시보드로 정리·시각화.

entity_bench_results.json(엔티티 A/B) + 차트모델 비교/분류기 A/B 발견을 모아
SVG 막대차트·표·인덱스로 렌더. 외부 라이브러리 없음. 검증된 팔레트(슬롯1 blue/슬롯2 orange)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
CFG = common.CONFIG
OUT = CFG["EVAL_DIR"] / "ab_dashboard.html"

# ---- 데이터 로드 (엔티티 A/B) ----
res = json.loads((CFG["EVAL_DIR"] / "entity_bench_results.json").read_text(encoding="utf-8"))
ORDER = [("V1", "① 베이스라인"), ("V2", "② +분류기"), ("V3", "③ +ChartQA"),
         ("V3H", "④ 하이브리드"), ("V4", "⑤ 개선(필터)")]
V = [{"key": k, "name": nm, "t": res[k]["time_s"],
      "p": res[k]["precision"] * 100, "r": res[k]["recall"] * 100, "f1": res[k]["f1"] * 100}
     for k, nm in ORDER]

BLUE, ORANGE, MUTED, GRID, INK, INK2 = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#0b0b0b", "#52514e"


def esc(s):
    import html; return html.escape(str(s))


def bars(data, series, ymax, unit, W=660, H=300):
    """data: list of dict. series: [(label,key,color),...]. → SVG 문자열."""
    ml, mr, mt, mb = 44, 16, 28, 62
    pw, ph = W - ml - mr, H - mt - mb
    n, ns = len(data), len(series)
    gw = pw / n
    bw = min(46, (gw - 12) / ns)
    svg = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    # y 그리드 + 눈금
    ticks = 5
    for i in range(ticks + 1):
        val = ymax * i / ticks
        y = mt + ph * (1 - i / ticks)
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lbl = f"{val:.0f}{unit}" if unit != "%" else f"{val:.0f}"
        svg.append(f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end" class="tick">{lbl}</text>')
    # 막대
    for gi, d in enumerate(data):
        gx = ml + gw * gi
        grp_w = bw * ns + 2 * (ns - 1)
        x0 = gx + (gw - grp_w) / 2
        for si, (slab, skey, col) in enumerate(series):
            v = d[skey]
            bh = ph * (v / ymax)
            x = x0 + si * (bw + 2)
            y = mt + ph - bh
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" '
                       f'fill="{col}"><title>{esc(d["name"])} · {esc(slab)}: {v:.1f}{unit}</title></rect>')
            svg.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" text-anchor="middle" class="vlab">{v:.0f}{unit if unit=="%" else ""}</text>')
        # x 라벨 (2줄)
        parts = d["name"].split(" ", 1)
        svg.append(f'<text x="{gx+gw/2:.1f}" y="{mt+ph+18:.1f}" text-anchor="middle" class="xlab">{esc(parts[0])}</text>')
        if len(parts) > 1:
            svg.append(f'<text x="{gx+gw/2:.1f}" y="{mt+ph+32:.1f}" text-anchor="middle" class="xlab2">{esc(parts[1])}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def legend(series):
    items = "".join(f'<span class="lg"><i style="background:{c}"></i>{esc(l)}</span>' for l, _, c in series)
    return f'<div class="legend">{items}</div>'


# ---- 차트 구성 ----
pr_series = [("Precision", "p", BLUE), ("Recall", "r", ORANGE)]
chart_pr = legend(pr_series) + bars(V, pr_series, 100, "%")
time_series = [("처리시간", "t", BLUE)]
chart_time = bars(V, time_series, 350, "s")

# ---- 표 행 ----
def rowclass(d):
    return "best" if d["f1"] == max(x["f1"] for x in V) else ""

entity_rows = "".join(
    f'<tr class="{rowclass(d)}"><td class="l">{esc(d["name"])}</td>'
    f'<td>{d["t"]:.0f}s</td><td>{d["p"]:.1f}%</td><td>{d["r"]:.1f}%</td>'
    f'<td><b>{d["f1"]:.1f}%</b></td></tr>' for d in V)

# 차트모델 비교 (Exp #8/#9)
CM = [
    ("Qwen3-VL-8B (베이스)", "✅ 정확", "❌ 축눈금만", "❌ 평문", "~27s"),
    ("DePlot (ChartQA)", "❌ gibberish", "✅ 정확(단일계열)", "✅ 표", "2–3s"),
    ("ChartGemma", "✅ 읽음", "❌ 부정확", "QA", "0.5–1s"),
]
cm_rows = "".join(
    f'<tr><td class="l">{esc(m)}</td><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>'
    for m, a, b, c, d in CM)

HTML = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>이미지 파이프라인 A/B 실험 대시보드</title>
<style>
:root{{ --surface:#fcfcfb; --plane:#f9f9f7; --ink:{INK}; --ink2:{INK2}; --muted:{MUTED};
        --grid:{GRID}; --line:rgba(11,11,11,.10); --best:#eaf3ff; }}
@media (prefers-color-scheme: dark){{ :root:where(:not([data-theme=light])){{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --line:rgba(255,255,255,.10); --best:#12233a; }} }}
:root[data-theme=dark]{{ --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --line:rgba(255,255,255,.10); --best:#12233a; }}
@media print{{
  :root{{ --surface:#fff; --plane:#fff; --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7873;
    --grid:#e1e0d9; --line:rgba(11,11,11,.14); --best:#eaf3ff; color-scheme:light; }}
  body{{ padding:0; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  .card,.two,.tiles,table,.chart,.call,.tile,tr{{ break-inside:avoid; page-break-inside:avoid; }}
  h1,h2{{ break-after:avoid; page-break-after:avoid; }}
}}
@page{{ size:A4; margin:12mm; }}
*{{box-sizing:border-box}}
body{{background:var(--plane);color:var(--ink);margin:0;padding:32px 20px;
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;line-height:1.6}}
.wrap{{max-width:920px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:34px 0 6px;padding-top:10px;border-top:2px solid var(--line)}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:8px}}
.lead{{color:var(--ink2);font-size:14px;margin:6px 0 14px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:center}}
th{{color:var(--ink2);font-weight:600;font-size:12.5px}} td.l{{text-align:left;font-weight:600}}
tr.best{{background:var(--best)}}
.chart{{width:100%;height:auto;display:block;margin:6px 0}}
.tick{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
.vlab{{fill:var(--ink2);font-size:11px;font-weight:600}}
.xlab{{fill:var(--ink);font-size:12px;font-weight:600}} .xlab2{{fill:var(--muted);font-size:11px}}
.legend{{display:flex;gap:16px;margin:2px 0 2px}} .lg{{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--ink2)}}
.lg i{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.tile .k{{font-size:12px;color:var(--muted)}} .tile .v{{font-size:24px;font-weight:700;margin:2px 0}}
.tile .s{{font-size:12px;color:var(--ink2)}}
.call{{border-left:3px solid {BLUE};padding:6px 12px;margin:8px 0;background:var(--surface);font-size:13.5px;color:var(--ink2)}}
.call.warn{{border-left-color:{ORANGE}}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
a{{color:{BLUE};text-decoration:none}} a:hover{{text-decoration:underline}}
.idx li{{margin:3px 0;font-size:13.5px}} code{{background:var(--plane);padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>이미지 파이프라인 — A/B 실험 대시보드</h1>
<div class="sub">하나증권 산업분석 리포트 · RTX 5080/CUDA · Qwen3-VL-8B + 그림분류기 + DePlot · 2026-07-22</div>
<div class="lead">세 갈래 A/B 실험을 한 장으로 정리: <b>①엔티티 추출 5변형</b> · <b>②차트 이해 모델 비교</b> · <b>③분류기 junk 트리아지</b>.</div>

<h2>실험 A · 엔티티 추출 5변형 (핵심)</h2>
<div class="lead">차트/이미지에서 기업·티커·지표 <b>엔티티</b>를 뽑는 정확도(P/R/F1)와 처리시간. 골든셋 56개(이미지 직접 판독), 표본 12장.</div>
<div class="card">
<table><tr><th>구성</th><th>처리시간</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{entity_rows}</table>
</div>
<div class="two">
  <div class="card"><b style="font-size:13px">Precision vs Recall</b>{chart_pr}</div>
  <div class="card"><b style="font-size:13px">총 처리시간 (초, 낮을수록 빠름)</b>{chart_time}</div>
</div>
<div class="call warn">⚠️ <b>③ +ChartQA 붕괴 (F1 4.8%)</b> — DePlot이 한글 범례를 gibberish(<code>현물가→&lt;0x..&gt;</code>)로 오독. 차트 전문모델이라도 <b>한글 엔티티엔 독</b>.</div>
<div class="call">✅ <b>⑤ 개선</b> — DePlot gibberish·축라벨 필터 후 합집합 → ④ 정밀도 58%→<b>100%</b>, F1 60→<b>77</b>. 결론: <b>DePlot은 수치 전용, 엔티티는 VLM</b>.</div>
<div class="call">📌 <b>①→② recall 하락(91→62%)</b> — 분류기가 "주요 고객사" 로고 그리드(기업 16개)를 junk로 컷. <b>"유용 차트"와 "엔티티 풍부"는 다른 축</b> → 엔티티 추출을 저장 게이트와 분리해야.</div>

<h2>실험 B · 차트 이해 모델 비교</h2>
<div class="lead">차트 1장을 각 모델에 통과시켜 한글 라벨·수치·구조화·속도 비교 (GPU).</div>
<div class="card">
<table><tr><th>모델</th><th>한글 라벨</th><th>차트 수치</th><th>구조화</th><th>속도</th></tr>
{cm_rows}</table>
</div>
<div class="call">단일 승자 없음 — <b>Qwen3-VL</b>(한글·요약) · <b>DePlot</b>(수치·구조, 단일계열) · <b>ChartGemma</b>(빠름, 수치 부정확)가 상보적.</div>

<h2>실험 C · 분류기 junk 트리아지 (industry_hana_17)</h2>
<div class="lead">그림 분류기가 로고·사진을 VLM 없이 걸러낸 효과. 144장 chart/image 중.</div>
<div class="tiles">
  <div class="tile"><div class="k">VLM 호출</div><div class="v">144 → 120</div><div class="s">분류기가 24장 선컷</div></div>
  <div class="tile"><div class="k">junk 선컷 비율</div><div class="v">17%</div><div class="s">로고·제품사진 (~15ms/장)</div></div>
  <div class="tile"><div class="k">정확도(육안검증)</div><div class="v">분류기 승</div><div class="s">VLM이 로고를 useful로 오판(FP), 분류기가 잡음</div></div>
</div>

<h2>결론 · 역할별 배치</h2>
<div class="call" style="border-left-color:#008300"><b>분류기는 저장을, DePlot은 수치를, VLM은 엔티티를</b> — 각자 최적 지점에 두고 하나의 게이트로 묶지 않는다.</div>

<h2>산출물 인덱스</h2>
<div class="card idx"><ul>
<li>📊 <a href="ab_dashboard.html">이 대시보드</a> (eval/ab_dashboard.html)</li>
<li>📝 엔티티 평가 보고서 — <a href="../docs/엔티티_평가_보고서.md">docs/엔티티_평가_보고서.md</a></li>
<li>🔬 실험 B 차트모델 상세 뷰어 — <a href="bench_variants.html">eval/bench_variants.html</a></li>
<li>🗂 원시·정답·지표 JSON — <code>eval/entity_bench_raw.json</code> · <code>golden_entities.json</code> · <code>entity_bench_results.json</code></li>
<li>📒 개발 로그(#7~#10) — <a href="../README.md">README.md</a></li>
<li>▶ 재현: <code>cuda_venv\\Scripts\\python pipeline\\eval_entities.py --run|--metrics</code></li>
</ul></div>

<div class="sub" style="margin-top:24px">생성: {esc(common.now_iso())} · 자립형 HTML(외부 의존 없음)</div>
</div></body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"대시보드 생성: {OUT}")
