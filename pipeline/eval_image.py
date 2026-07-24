# -*- coding: utf-8 -*-
"""eval_image: image_cards.jsonl + timings.jsonl + 캐시디렉터리에서 '고도화 지표'를 산출한다.

측정 항목
  · 파싱/크롭 커버리지, filter_stage 단계별 분포
  · [고도화] VLM 절감률 = (캐시적중 + 완전중복복사) / 판정대상  ← 캐싱·pHash중복제거 효과
  · [고도화] review_queue 비율 + confidence 분포                ← 저신뢰 자동 선별
  · [고도화] table 인계 무결성 (table 크롭이 VLM을 한 건도 안 탔는지)
  · 판정 분포(useful/discarded, vlm_type별), 처리속도 p50/p95 (파싱·VLM)
  · (eval/image_labels.csv 있으면) accuracy/precision/recall + 로고 FP율 (M4·M5)

콘솔 요약 + 자립형 HTML 대시보드(eval/eval_report.html) 생성."""
from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import common

CFG = common.CONFIG
logger = common.get_logger("eval_image")


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def load_timings(script_prefix: str) -> list[float]:
    out = []
    for r in common.load_jsonl(CFG["TIMINGS_JSONL"]):
        if r.get("script", "").startswith(script_prefix):
            try:
                out.append(float(r["seconds"]))
            except (TypeError, ValueError, KeyError):
                pass
    return out


def count_cache_files() -> int:
    d = CFG["CACHE_DIR"] / "vlm"
    return sum(1 for _ in d.rglob("*.json")) if d.exists() else 0


def load_labels() -> dict[str, dict]:
    """eval/image_labels.csv → {image_id: {label_useful(bool), label_type}}. 없으면 {}."""
    path = CFG["EVAL_DIR"] / "image_labels.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            iid = (row.get("image_id") or "").strip()
            lu = (row.get("label_useful") or "").strip().lower()
            if not iid or lu == "":
                continue
            out[iid] = {"label_useful": lu in ("1", "true", "y", "yes", "useful", "o", "유용"),
                        "label_type": (row.get("label_type") or "").strip()}
    return out


def compute(cards: list[dict]) -> dict:
    n = len(cards)
    by_block = Counter(c.get("block_type") for c in cards)
    by_status = Counter(c.get("status") for c in cards)
    stage = Counter((c.get("filter_stage") or "?").split(":")[0] for c in cards)

    judged = [c for c in cards if c.get("vlm")]           # VLM 결과가 붙은 카드
    src = Counter(c.get("cache_source") for c in judged)  # vlm|cache|dedup
    vlm_calls = src.get("vlm", 0)
    cache_hit = src.get("cache", 0)
    dedup = src.get("dedup", 0)
    judge_target = vlm_calls + cache_hit + dedup
    saved = cache_hit + dedup
    saving_rate = (saved / judge_target) if judge_target else 0.0

    similar = sum(1 for c in cards if c.get("similar_of"))
    review = [c for c in cards if c.get("review_queue")]
    confs = [float(c["vlm"]["confidence"]) for c in judged
             if isinstance(c.get("vlm", {}).get("confidence"), (int, float))]

    vtypes = Counter(c["vlm"].get("type") for c in judged)
    useful_n = sum(1 for c in cards if c.get("status") == "useful")

    # table 인계 무결성: block_type=table 크롭이 VLM 판정(cache_source)이 없어야 함
    tables = [c for c in cards if c.get("block_type") == "table"]
    table_vlmd = sum(1 for c in tables if c.get("vlm"))
    handoff_n = len(common.load_jsonl(CFG["HANDOFF_TABLES_JSONL"]))

    parse_t = load_timings("s1_parse")
    vlm_t = load_timings("s2_vlm")

    result = {
        "n_cards": n, "by_block": dict(by_block), "by_status": dict(by_status),
        "stage": dict(stage),
        "vlm_calls": vlm_calls, "cache_hit": cache_hit, "dedup": dedup,
        "judge_target": judge_target, "saved": saved, "saving_rate": saving_rate,
        "similar": similar, "review_n": len(review),
        "review_rate": (len(review) / judge_target) if judge_target else 0.0,
        "conf_mean": statistics.mean(confs) if confs else None,
        "conf_p10": _pct(confs, 0.10), "conf_lt60": sum(1 for x in confs if x < 0.6),
        "vtypes": dict(vtypes.most_common()),
        "useful_n": useful_n,
        "useful_rate": (useful_n / judge_target) if judge_target else 0.0,
        "tables_n": len(tables), "table_vlmd": table_vlmd, "handoff_n": handoff_n,
        "parse_p50": _pct(parse_t, 0.5), "parse_p95": _pct(parse_t, 0.95),
        "parse_n": len(parse_t),
        "vlm_p50": _pct(vlm_t, 0.5), "vlm_p95": _pct(vlm_t, 0.95),
        "vlm_mean": statistics.mean(vlm_t) if vlm_t else None, "vlm_n": len(vlm_t),
        "cache_files": count_cache_files(),
    }

    # ---- 정확도 (라벨 있을 때만) ----
    labels = load_labels()
    if labels:
        tp = fp = tn = fn = 0
        logo_fp = 0
        matched = 0
        for c in judged:
            lab = labels.get(c["image_id"])
            if not lab:
                continue
            matched += 1
            pred = bool(c["vlm"].get("useful"))
            gold = bool(lab["label_useful"])
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
                if c["vlm"].get("type") in ("logo", "decoration"):
                    logo_fp += 1
            elif not pred and gold:
                fn += 1
            else:
                tn += 1
        acc = (tp + tn) / matched if matched else None
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        result["labeled"] = {"matched": matched, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                             "accuracy": acc, "precision": prec, "recall": rec,
                             "logo_fp": logo_fp,
                             "logo_fp_rate": (logo_fp / matched) if matched else None}
    else:
        result["labeled"] = None
    return result


# ---------------------------------------------------------------- 콘솔 출력

def print_console(r: dict) -> None:
    p = logger.info
    p("================ 이미지 파이프라인 고도화 지표 ================")
    p(f"이미지 카드 총 {r['n_cards']}건  |  블록: {r['by_block']}")
    p(f"상태 분포: {r['by_status']}")
    p(f"단계(filter_stage): {r['stage']}")
    p("---- [고도화] VLM 절감 (캐싱 + pHash 중복제거) ----")
    p(f"판정대상 {r['judge_target']} = VLM호출 {r['vlm_calls']} + 캐시적중 {r['cache_hit']} + 완전중복복사 {r['dedup']}")
    p(f"★ VLM 절감률: {r['saving_rate']*100:.1f}%  (재실행 시 캐시로 100%까지 상승)")
    p(f"유사(1~6bit) 표시: {r['similar']}  |  캐시 파일 수: {r['cache_files']}")
    p("---- [고도화] 저신뢰 자동 선별 ----")
    cm = f"{r['conf_mean']:.2f}" if r['conf_mean'] is not None else "-"
    p(f"review_queue: {r['review_n']}건 ({r['review_rate']*100:.1f}%)  |  평균 confidence: {cm}  |  conf<0.6: {r['conf_lt60']}")
    p("---- [고도화] table 인계 무결성 ----")
    ok = "✓ 통과" if r['table_vlmd'] == 0 else f"✗ 위반({r['table_vlmd']}건 VLM 탐)"
    p(f"table 크롭 {r['tables_n']}건, VLM 판정 {r['table_vlmd']}건 → {ok}  |  handoff 기록 {r['handoff_n']}건")
    p("---- 판정 분포 ----")
    p(f"useful {r['useful_n']} ({r['useful_rate']*100:.1f}%)  |  유형: {r['vtypes']}")
    p("---- 처리 속도 ----")
    vp50 = f"{r['vlm_p50']:.1f}s" if r['vlm_p50'] is not None else "-"
    vp95 = f"{r['vlm_p95']:.1f}s" if r['vlm_p95'] is not None else "-"
    pp50 = f"{r['parse_p50']:.1f}s" if r['parse_p50'] is not None else "-"
    p(f"VLM 장당 p50 {vp50} / p95 {vp95} (n={r['vlm_n']})  |  파싱 문서당 p50 {pp50} (n={r['parse_n']})")
    if r["labeled"]:
        L = r["labeled"]
        p("---- 정확도 (수동 라벨 대조) ----")
        acc = f"{L['accuracy']*100:.1f}%" if L['accuracy'] is not None else "-"
        prec = f"{L['precision']*100:.1f}%" if L['precision'] is not None else "-"
        rec = f"{L['recall']*100:.1f}%" if L['recall'] is not None else "-"
        p(f"표본 {L['matched']}장 | accuracy {acc} | precision {prec} | recall {rec} | 로고FP {L['logo_fp']}")
    else:
        p("---- 정확도: eval/image_labels.csv 없음 → review_viewer.py 로 라벨 생성 후 재실행 ----")
    p("=============================================================")


# ---------------------------------------------------------------- HTML 대시보드

def _tile(label: str, value: str, sub: str = "", ok: bool | None = None) -> str:
    color = "#2c313b" if ok is None else ("#1f6f43" if ok else "#8a3b3b")
    e = html_mod.escape
    return (f'<div class="tile" style="border-left:4px solid {color}">'
            f'<div class="tl">{e(label)}</div><div class="tv">{e(value)}</div>'
            f'<div class="ts">{e(sub)}</div></div>')


def _bar(dist: dict, total: int) -> str:
    e = html_mod.escape
    rows = []
    for k, v in dist.items():
        pctv = (v / total * 100) if total else 0
        rows.append(f'<div class="brow"><span class="bk">{e(str(k))}</span>'
                    f'<span class="bbar"><span style="width:{pctv:.0f}%"></span></span>'
                    f'<span class="bv">{v} ({pctv:.0f}%)</span></div>')
    return "".join(rows)


def write_html(r: dict) -> Path:
    e = html_mod.escape
    saving = f"{r['saving_rate']*100:.1f}%"
    tiles = [
        _tile("이미지 카드", str(r["n_cards"]), f"블록 {r['by_block']}"),
        _tile("VLM 호출", str(r["vlm_calls"]), f"판정대상 {r['judge_target']}"),
        _tile("★ VLM 절감률", saving, f"캐시 {r['cache_hit']} + 중복 {r['dedup']}",
              ok=r["saving_rate"] > 0),
        _tile("완전중복 복사", str(r["dedup"]), f"유사표시 {r['similar']}"),
        _tile("review_queue", str(r["review_n"]), f"{r['review_rate']*100:.1f}% · conf<0.6 {r['conf_lt60']}"),
        _tile("useful", str(r["useful_n"]), f"{r['useful_rate']*100:.1f}%"),
        _tile("table 인계 무결성", "통과" if r["table_vlmd"] == 0 else "위반",
              f"table {r['tables_n']} · VLM탄것 {r['table_vlmd']} · handoff {r['handoff_n']}",
              ok=r["table_vlmd"] == 0),
        _tile("VLM 속도 p50",
              f"{r['vlm_p50']:.1f}s" if r["vlm_p50"] is not None else "-",
              f"p95 {r['vlm_p95']:.1f}s" if r["vlm_p95"] is not None else "n=0",
              ok=(r["vlm_p50"] is not None and r["vlm_p50"] <= 7)),
        _tile("파싱 속도 p50",
              f"{r['parse_p50']:.1f}s" if r["parse_p50"] is not None else "-",
              f"문서 {r['parse_n']}건"),
    ]
    if r["labeled"]:
        L = r["labeled"]
        acc = L["accuracy"]
        tiles.insert(0, _tile("정확도(accuracy)",
                              f"{acc*100:.1f}%" if acc is not None else "-",
                              f"표본 {L['matched']}장 · 로고FP {L['logo_fp']}",
                              ok=(acc is not None and acc >= 0.9)))

    labeled_html = ""
    if r["labeled"]:
        L = r["labeled"]
        labeled_html = f"""<div class="panel"><h2>정확도 (수동 라벨 {L['matched']}장 대조 · M4/M5)</h2>
        <table class="cm"><tr><th></th><th>예측 useful</th><th>예측 버림</th></tr>
        <tr><th>실제 useful</th><td class="tp">{L['tp']}</td><td class="fn">{L['fn']}</td></tr>
        <tr><th>실제 버림</th><td class="fp">{L['fp']}</td><td class="tn">{L['tn']}</td></tr></table>
        <p>precision {(L['precision'] or 0)*100:.1f}% · recall {(L['recall'] or 0)*100:.1f}% · 로고 FP율 {(L['logo_fp_rate'] or 0)*100:.1f}%</p></div>"""
    else:
        labeled_html = ('<div class="panel"><h2>정확도</h2><p class="muted">'
                        'eval/image_labels.csv 가 없습니다. <b>review_viewer.py</b> 로 표본 검수 HTML을 열어 '
                        '라벨(useful/버림)을 입력·저장한 뒤 eval_image.py 를 다시 실행하면 accuracy/precision/recall 이 채워집니다.</p></div>')

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>이미지 파이프라인 고도화 지표</title><style>
:root{{color-scheme:dark}}
body{{background:#14161a;color:#dfe3ea;font-family:'Malgun Gothic',system-ui,sans-serif;margin:24px;line-height:1.5}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:0 0 10px;color:#cdd6e6}}
.sub{{color:#8b93a3;font-size:13px;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin-bottom:24px}}
.tile{{background:#1d2027;border-radius:10px;padding:14px 16px}}
.tl{{font-size:12px;color:#9aa3b2}} .tv{{font-size:26px;font-weight:700;margin:4px 0}}
.ts{{font-size:12px;color:#8b93a3}}
.panel{{background:#1d2027;border-radius:10px;padding:18px;margin-bottom:16px}}
.brow{{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px}}
.bk{{width:130px;color:#b9c2d4}} .bbar{{flex:1;background:#2c313b;border-radius:6px;height:14px;overflow:hidden}}
.bbar>span{{display:block;height:100%;background:#3f6fd1}} .bv{{width:110px;text-align:right;color:#9aa3b2}}
table.cm{{border-collapse:collapse;margin:6px 0}} table.cm th,table.cm td{{border:1px solid #2c313b;padding:8px 16px;text-align:center}}
.tp{{background:#1f5f3f}} .tn{{background:#33414f}} .fp{{background:#7a2f2f}} .fn{{background:#7a5a1f}}
.muted{{color:#8b93a3}} b{{color:#dfe3ea}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:820px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>이미지 파이프라인 — 고도화 지표 대시보드</h1>
<div class="sub">생성 {e(common.now_iso())} · prompt_ver {e(CFG['PROMPT_VER'])} · model {e(CFG['VLM_MODEL'])} · 대상 하나증권 산업분석 20건</div>
<div class="grid">{''.join(tiles)}</div>
<div class="two">
  <div class="panel"><h2>단계별 분포 (filter_stage)</h2>{_bar(r['stage'], r['n_cards'])}</div>
  <div class="panel"><h2>VLM 유형 분포</h2>{_bar(r['vtypes'], sum(r['vtypes'].values()) or 1)}</div>
</div>
<div class="panel"><h2>[고도화] VLM 절감 상세</h2>
  <p>판정대상 <b>{r['judge_target']}</b>장 중 실제 VLM 호출 <b>{r['vlm_calls']}</b>회 —
  캐시적중 <b>{r['cache_hit']}</b> + 완전중복복사 <b>{r['dedup']}</b> = <b>{r['saved']}</b>장 절감 (<b>{saving}</b>).
  캐시 키 = content_hash + prompt_ver + model 이므로 <b>재실행 시 동일 이미지는 VLM을 타지 않는다</b>(적중률 100%로 수렴).
  pHash(dHash) 해밍거리 0 은 판정 복사, 1~6 은 '유사' 표시만 하고 재판정(시계열 차트 오복사 방지).</p></div>
{labeled_html}
<div class="panel"><h2>상태 분포</h2>{_bar(r['by_status'], r['n_cards'])}</div>
</body></html>"""
    out = CFG["EVAL_DIR"] / "eval_report.html"
    CFG["EVAL_DIR"].mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="이미지 파이프라인 고도화 지표 산출 + HTML")
    ap.add_argument("--json", action="store_true", help="지표를 JSON으로도 출력")
    args = ap.parse_args()
    common.ensure_dirs()
    cards = list(common.jsonl_index(CFG["IMAGE_CARDS_JSONL"], "image_id").values())
    if not cards:
        logger.info("image_cards.jsonl 이 비어 있습니다. 먼저 s2_image_pipeline.py 를 실행하세요.")
        return
    r = compute(cards)
    print_console(r)
    out = write_html(r)
    logger.info(f"HTML 대시보드: {out}")
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
