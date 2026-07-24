# -*- coding: utf-8 -*-
"""A/B: industry_hana_17 로 full 라우팅 vs clf 라우팅 비교.

효율적 설계 — 분류기는 chart/image 전 크롭에 즉시 적용(ms), VLM은 '분류기가 junk로
스킵한' 크롭에만 돌려 정확도(진짜 유용한 걸 잘못 버렸나=FN)를 교차검증한다.
이로써 A/B 핵심(절감량 + 손실여부 + 시간)을 풀 VLM 없이 몇 분 안에 얻는다."""
from __future__ import annotations

import sys, time, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common, figure_classifier as fc
from s2_image_pipeline import find_crop, get_caption, rule_filter, PROMPT_V2, normalize_vlm
from pathlib import Path

CFG = common.CONFIG
DOC = "industry_hana_17"
FN_SAMPLE = 15   # junk 스킵분 중 VLM 교차검증 표본 상한

def main():
    docs = dict(common.find_parsed_docs())
    mdir = docs.get(DOC)
    if not mdir:
        print(f"{DOC} 파싱본 없음"); return
    content = common.load_content_list(mdir)

    items = []  # (image_id, btype, crop, caption)
    counters = {}
    for it in content:
        bt = it.get("type")
        if bt not in ("chart", "image"):
            continue
        page = int(it.get("page_idx", 0)) + 1
        k = f"{page}:{bt}"; counters[k] = counters.get(k, 0) + 1
        iid = f"{DOC}_p{page}_{bt}{counters[k]}"
        crop = find_crop(Path(mdir), it)
        if crop is None:
            continue
        # 규칙필터로 먼저 걸리는 건 두 라우팅 공통 탈락 → 제외
        if rule_filter(crop, bt):
            continue
        items.append((iid, bt, crop, get_caption(it)))

    print(f"===== A/B 대상: {DOC} =====")
    print(f"규칙필터 통과 chart/image 크롭: {len(items)}")

    # ---- 분류기 전수 라우팅 (ms) ----
    t0 = time.time()
    routed = []  # (iid, btype, crop, caption, clf)
    for iid, bt, crop, cap in items:
        clf = fc.classify(crop)
        routed.append((iid, bt, crop, cap, clf))
    clf_dt = time.time() - t0

    import collections
    by_route = collections.Counter(r[4]["route"] for r in routed if r[4])
    by_route_bt = collections.Counter((r[1], r[4]["route"]) for r in routed if r[4])
    junk = [r for r in routed if r[4] and r[4]["route"] == "junk"]

    print(f"분류기 소요: {clf_dt:.1f}s ({clf_dt/max(1,len(items))*1000:.0f}ms/장)")
    print(f"라우팅: {dict(by_route)}")
    print(f"블록타입×라우팅: {dict(by_route_bt)}")
    print(f"junk 선컷(=VLM 스킵) 후보: {len(junk)}")

    # ---- FN 교차검증: junk 스킵분을 VLM에 물어봄 ----
    rnd = random.Random(0)
    check = junk if len(junk) <= FN_SAMPLE else rnd.sample(junk, FN_SAMPLE)
    fn = 0; vlm_times = []
    print(f"\n----- junk 스킵분 VLM 교차검증 (표본 {len(check)}/{len(junk)}) -----")
    for iid, bt, crop, cap, clf in check:
        prompt = PROMPT_V2.format(caption=cap or "없음", doc_title="", category="industry")
        t = time.time()
        res = common.ollama_chat(CFG["VLM_MODEL"], prompt, images=[str(crop)],
                                 num_ctx=CFG["VLM_NUM_CTX"], img_max_edge=CFG["VLM_MAX_EDGE"],
                                 think=CFG["VLM_THINK"])
        dt = time.time() - t; vlm_times.append(dt)
        useful = None
        if res and not res.get("_parse_error"):
            useful = normalize_vlm(res, bt).get("useful")
        flag = "FN!(VLM=유용인데 버림)" if useful else "OK(VLM도 버림)"
        if useful:
            fn += 1
        print(f"  {clf['label']:12s} conf {clf['confidence']:.2f} | VLM useful={useful} {flag} | {dt:.1f}s")

    avg_vlm = sum(vlm_times)/len(vlm_times) if vlm_times else 0
    n = len(items)
    n_junk = len(junk)
    full_calls = n
    clf_calls = n - n_junk
    print("\n================ A/B 결과 ================")
    print(f"{'':22s}{'full':>10s}{'clf':>12s}")
    print(f"{'VLM 호출 수':22s}{full_calls:>10d}{clf_calls:>12d}")
    print(f"{'분류기 선컷(스킵)':22s}{0:>10d}{n_junk:>12d}")
    print(f"{'VLM 호출 절감':22s}{'-':>10s}{f'{n_junk} ({n_junk/n*100:.0f}%)':>12s}")
    if avg_vlm:
        print(f"{'예상 VLM시간(장{avg_vlm:.0f}s)':22s}{full_calls*avg_vlm/60:>9.1f}m{clf_calls*avg_vlm/60:>11.1f}m")
        print(f"{'시간 절감':22s}{'-':>10s}{f'{n_junk*avg_vlm/60:.1f}m':>12s}")
    print(f"\n정확도: junk 스킵 {len(check)}건 중 FN(진짜 유용 오폐기) = {fn}건")
    if len(check):
        print(f"  → 오폐기율 {fn/len(check)*100:.0f}% (0%면 분류기 스킵이 안전)")
    print("==========================================")

if __name__ == "__main__":
    main()
