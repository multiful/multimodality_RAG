# -*- coding: utf-8 -*-
"""review_viewer: image_cards.jsonl 에서 review_queue + 무작위 표본을 카드형 HTML로 만든다.

브라우저에서 각 카드의 [유용]/[불필요] 를 클릭하면 하단 '라벨 CSV 저장' 버튼으로
eval/image_labels.csv (image_id, vlm_type, vlm_useful, label_useful, label_type) 를 내려받는다.
그 파일을 eval/ 에 두고 eval_image.py 를 다시 실행하면 accuracy/precision/recall(M4)이 채워진다.

우선순위: review_queue=true(저신뢰) 카드를 먼저, 나머지는 무작위로 채운다. 이미지 base64 임베드 → 자립형."""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import json
import random
from pathlib import Path

import common

CFG = common.CONFIG
logger = common.get_logger("review_viewer")


def _img_tag(card: dict) -> str:
    p = common.PROJECT_ROOT / card.get("file", "")
    if not p.is_file():
        return '<div class="noimg">이미지 없음</div>'
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{b64}" loading="lazy">'


def build(n: int, seed: int = 42) -> Path:
    common.ensure_dirs()
    cards = list(common.jsonl_index(CFG["IMAGE_CARDS_JSONL"], "image_id").values())
    judged = [c for c in cards if c.get("vlm")]
    if not judged:
        logger.info("판정된 카드가 없습니다. 먼저 s2_image_pipeline.py 를 실행하세요.")
        return CFG["EVAL_DIR"] / "review.html"

    rq = [c for c in judged if c.get("review_queue")]
    rest = [c for c in judged if not c.get("review_queue")]
    rnd = random.Random(seed)
    rnd.shuffle(rest)
    sample = (rq + rest)[:n]

    esc = html_mod.escape
    blocks = []
    for c in sample:
        vlm = c.get("vlm") or {}
        chips = [f'<span class="chip">{esc(c.get("block_type",""))}</span>',
                 f'<span class="chip">{esc(str(vlm.get("type","")))}</span>',
                 f'<span class="chip">VLM:useful={esc(str(vlm.get("useful")))}</span>',
                 f'<span class="chip">conf={esc(str(round(float(vlm.get("confidence",0)),2)))}</span>']
        if c.get("review_queue"):
            chips.append('<span class="chip warn">review</span>')
        if c.get("similar_of"):
            chips.append(f'<span class="chip">유사~{esc(str(c.get("similar_ham")))}</span>')
        ocr = esc((vlm.get("ocr_text") or "")[:1200])
        blocks.append(f"""<div class="card" data-id="{esc(c['image_id'])}" data-vtype="{esc(str(vlm.get('type','')))}" data-vuseful="{esc(str(vlm.get('useful')))}">
  <div class="imgbox">{_img_tag(c)}</div>
  <div class="meta"><b>{esc(c['image_id'])}</b> · p{c.get('page')}</div>
  <div class="chips">{''.join(chips)}</div>
  <div class="cap">{esc(c.get('caption') or '')}</div>
  <div class="sum">{esc(vlm.get('summary') or '')}</div>
  <div class="btns"><button class="bU" onclick="mark(this,'1')">유용</button>
    <button class="bX" onclick="mark(this,'0')">불필요</button>
    <span class="lab"></span></div>
  <details><summary>OCR</summary><pre>{ocr}</pre></details>
</div>""")

    js = """
<script>
function mark(btn,val){
  var card=btn.closest('.card'); card.dataset.label=val;
  card.querySelector('.lab').textContent = (val==='1'?'✓ 유용':'✗ 불필요');
  card.style.outline = '2px solid '+(val==='1'?'#2f9e5f':'#c0504d');
}
function saveCsv(){
  var rows=[['image_id','vlm_type','vlm_useful','label_useful','label_type']];
  var done=0;
  document.querySelectorAll('.card').forEach(function(c){
    if(c.dataset.label!==undefined){
      rows.push([c.dataset.id,c.dataset.vtype,c.dataset.vuseful,c.dataset.label,'']);
      done++;
    }
  });
  if(done===0){alert('라벨을 하나 이상 표시하세요.');return;}
  var csv='\\ufeff'+rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\\r\\n');
  var blob=new Blob([csv],{type:'text/csv'}); var a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='image_labels.csv'; a.click();
  document.getElementById('cnt').textContent=done+'건 라벨 저장됨 → eval/image_labels.csv 로 옮기세요';
}
</script>"""

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>이미지 판정 검수 ({len(sample)}장)</title><style>
:root{{color-scheme:dark}}
body{{background:#14161a;color:#dfe3ea;font-family:'Malgun Gothic',sans-serif;margin:20px}}
h1{{font-size:20px}} .bar{{position:sticky;top:0;background:#14161a;padding:10px 0;z-index:5;border-bottom:1px solid #2c313b;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}}
.card{{background:#1d2027;border:1px solid #2c313b;border-radius:10px;padding:12px}}
.imgbox{{background:#fff;border-radius:6px;text-align:center;margin-bottom:8px}}
.imgbox img{{max-width:100%;max-height:280px}} .noimg{{color:#888;padding:40px 0}}
.meta{{font-size:13px}} .chips{{margin:6px 0}}
.chip{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;margin:2px 4px 2px 0;background:#2c3a55;color:#fff}}
.chip.warn{{background:#7a4a12}} .cap{{font-size:12px;color:#9fb2d8}} .sum{{font-size:13px;margin:6px 0}}
.btns{{margin:8px 0}} .btns button{{border:0;border-radius:6px;padding:6px 14px;margin-right:6px;cursor:pointer;font-weight:600}}
.bU{{background:#2f9e5f;color:#fff}} .bX{{background:#c0504d;color:#fff}} .lab{{font-size:13px;color:#cdd6e6}}
details{{font-size:12px}} pre{{white-space:pre-wrap;color:#aab;max-height:200px;overflow:auto}}
#save{{background:#3f6fd1;color:#fff;border:0;border-radius:6px;padding:8px 18px;cursor:pointer;font-weight:700}}
#cnt{{margin-left:12px;color:#9aa3b2;font-size:13px}}
</style></head><body>
<h1>이미지 판정 검수 — 표본 {len(sample)}장 (review_queue 우선)</h1>
<div class="bar"><button id="save" onclick="saveCsv()">라벨 CSV 저장</button><span id="cnt"></span>
<span style="float:right;color:#8b93a3;font-size:12px">각 카드에서 유용/불필요를 누른 뒤 저장 → eval/image_labels.csv</span></div>
<div class="grid">{''.join(blocks)}</div>{js}</body></html>"""

    out = CFG["EVAL_DIR"] / "review.html"
    out.write_text(doc, encoding="utf-8")
    logger.info(f"검수 뷰어 생성: {out} (표본 {len(sample)}장, review_queue {len(rq)}장 우선)")
    logger.info("브라우저로 열어 라벨 입력 → 'CSV 저장' → eval/image_labels.csv 로 저장 → eval_image.py 재실행")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="검수 뷰어 HTML 생성 (라벨→CSV 다운로드)")
    ap.add_argument("-n", "--num", type=int, default=50, help="표본 장수 (기본 50)")
    args = ap.parse_args()
    build(args.num)


if __name__ == "__main__":
    main()
