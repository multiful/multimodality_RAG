# -*- coding: utf-8 -*-
"""s2_image_pipeline (고도화판): MinerU chart/image 크롭을 4단계 게이트로 처리한다.

  [A] 규칙 필터   : 크기·종횡비·면적 (chart는 크기 무관 통과)
  [B] 캐시·중복   : content_hash+prompt_ver+model 로 VLM 캐시 조회 (L1 로컬),
                    pHash(dHash) 해밍거리로 완전중복(=0)은 판정 복사, 1~6은 '유사'만 표시하고 VLM 재실행
  [C] VLM 판정    : Qwen3-VL, JSON 강제, confidence<0.6 → review_queue, 파싱실패 1회 재시도
  [D] 저장        : useful→data/images/useful, 그 외→discarded/vlm (보관), table→handoff(무판정)

기본형(pdfex s2) 대비 '고도화된 부분' = 캐시·pHash중복제거·confidence게이트·table인계·prompt_ver.
전 건(탈락 포함) image_cards.jsonl 에 filter_stage 사유와 함께 기록 → eval_image.py 가 지표 산출.
Supabase 미설정 시 로컬 JSONL 이 원본 (upsert는 자동 생략)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import common

CFG = common.CONFIG
logger = common.get_logger("s2_image_pipeline")

# VLM 대상 = chart, image.  table 은 판정하지 않고 인계 목록에만 기록(스펙 §2)
VLM_TYPES = ("chart", "image")

PROMPT_V2 = """이 이미지는 한국 증권사 리서치 리포트에서 추출된 그림입니다.
[캡션: {caption}]  [리포트: {doc_title} / {category}]
아래 JSON만 출력하세요. 다른 말은 하지 마세요.
{{
 "type": "line_chart|bar_chart|pie_chart|radar_chart|candle_chart|mixed_chart|table_image|diagram|photo|logo|decoration|other 중 하나",
 "useful": true 또는 false,
 "confidence": 0.0~1.0,
 "title": "이미지 제목",
 "ocr_text": "축 라벨·범례·수치 포함 모든 텍스트",
 "summary": "핵심 내용 1~2문장 (한국어)",
 "entities": ["언급된 기업명·티커·지표명"]
}}
판정 철학: 로고를 유용으로 넣는 오류(FP)가 차트를 놓치는 것(FN)보다 해롭다. 애매하면 useful=false."""

# type → useful 재분류에 쓰는 '유용 유형' 집합
_USEFUL_TYPES = {"line_chart", "bar_chart", "pie_chart", "radar_chart",
                 "candle_chart", "mixed_chart", "table_image", "diagram"}
_TABLE_IMAGE_TYPES = {"table_image"}


# ---------------------------------------------------------------- 경로/크롭 유틸

def rel_to_root(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(common.PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def find_crop(mdir: Path, item: dict) -> Path | None:
    rel = item.get("img_path") or item.get("image_path") or ""
    if not isinstance(rel, str) or not rel:
        return None
    p = Path(mdir) / rel
    if p.exists():
        return p
    for cand in Path(mdir).rglob(Path(rel).name):
        return cand
    return None


def get_caption(item: dict) -> str:
    parts: list[str] = []
    for key in ("chart_caption", "image_caption", "table_caption"):
        v = item.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
    return common.clean_text(" ".join(parts))


def rule_filter(img_file: Path, block_type: str) -> str | None:
    """[A] 규칙필터. 탈락사유 문자열 또는 통과 None. chart 는 크기 무관 통과(스파크라인 보호)."""
    if block_type == "chart":
        return None
    try:
        from PIL import Image
        with Image.open(img_file) as im:
            w, h = im.size
    except Exception:
        return "unreadable"
    if w < CFG["MIN_IMG_PX"] or h < CFG["MIN_IMG_PX"]:
        return "too_small"
    aspect = w / h if h else 999.0
    if aspect > CFG["MAX_ASPECT"] or aspect < 1.0 / CFG["MAX_ASPECT"]:
        return "extreme_aspect"
    if w * h < CFG["MIN_AREA_PX"]:
        return "too_small_area"
    return None


def copy_crop(src: Path, base_dir: Path, doc_id: str, image_id: str) -> Path:
    import shutil
    dst = base_dir / doc_id / f"{image_id}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------- VLM 결과 정규화

def normalize_vlm(res: dict, block_type: str) -> dict:
    vtype = res.get("type")
    vtype = vtype.strip() if isinstance(vtype, str) and vtype.strip() else "other"
    useful = res.get("useful")
    if isinstance(useful, str):
        useful = useful.strip().lower() in ("true", "1", "yes", "y", "예", "참")
    else:
        useful = bool(useful)
    try:
        conf = float(res.get("confidence"))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5
    def _s(v):
        return v if isinstance(v, str) else ("" if v is None else str(v))
    ents = res.get("entities")
    if isinstance(ents, str):
        ents = [ents.strip()] if ents.strip() else []
    elif isinstance(ents, list):
        ents = [str(e).strip() for e in ents if str(e).strip()]
    else:
        ents = []
    return {"type": vtype, "useful": useful, "confidence": conf,
            "title": _s(res.get("title")), "ocr_text": _s(res.get("ocr_text")),
            "summary": _s(res.get("summary")), "entities": ents}


def build_embed_text(caption: str, vlm: dict) -> str:
    parts = [caption, common.clean_text(vlm.get("title") or ""),
             common.clean_text(vlm.get("summary") or ""),
             common.clean_text(vlm.get("ocr_text") or "")[:500]]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------- 인계(table)

def handoff_table(image_id: str, doc_id: str, page: int, item: dict,
                  crop_rel: str | None, caption: str, source: str = "mineru_table") -> None:
    common.append_jsonl(CFG["HANDOFF_TABLES_JSONL"], {
        "image_id": image_id, "doc_id": doc_id, "page": page,
        "bbox": item.get("bbox") or None, "crop_path": crop_rel,
        "caption": caption or None, "source": source, "handoff_ver": "1",
    })


# ---------------------------------------------------------------- 메인 파이프라인

def process(args: argparse.Namespace) -> None:
    common.ensure_dirs()
    meta = {r["doc_id"]: r for r in common.read_metadata()}
    docs = common.find_parsed_docs(args.parsed_root)
    if args.category:
        docs = [(d, m) for d, m in docs if d.startswith(args.category)]
    if args.doc:
        docs = [(d, m) for d, m in docs if d == args.doc]
    if not docs:
        logger.info("처리할 파싱 결과가 없습니다. 먼저 s1_parse.py 를 실행하세요.")
        return

    prompt_ver = CFG["PROMPT_VER"]
    model = CFG["VLM_MODEL"]
    use_cache = not args.no_cache

    cards = common.jsonl_index(CFG["IMAGE_CARDS_JSONL"], "image_id")

    vlm_ok = False
    if args.rules_only:
        logger.info("--rules-only: VLM 생략, 규칙필터만 (통과분 pending)")
    else:
        vlm_ok = common.ollama_alive() and common.has_model(model)
        if not vlm_ok:
            logger.info(f"경고: Ollama/{model} 사용 불가 — rules-only 로 전환")

    # pHash 레지스트리: 기존 카드(판정 완료분)에서 시드 → 이번 실행분 append
    phash_reg: list[tuple[str, dict]] = []
    for c in cards.values():
        if c.get("phash") and c.get("vlm"):
            phash_reg.append((c["phash"], c))

    clf_ok = False
    if args.route == "clf":
        import figure_classifier
        clf_ok = figure_classifier.available()
        logger.info(f"라우팅=clf: 그림 분류기 게이트 {'ON' if clf_ok else '로드실패→full로 폴백'}")

    stats = {k: 0 for k in ("total", "useful", "discarded_rule", "discarded_vlm",
                            "discarded_clf", "review_queue", "table_handoff", "cache_hit",
                            "dedup_exact", "similar_flagged", "parse_error", "skipped",
                            "pending")}
    vlm_calls = 0
    fail_streak = 0
    stop = False

    for di, (doc_id, mdir) in enumerate(docs, 1):
        if stop:
            break
        common.log_progress(logger, di, len(docs), doc_id)
        content = common.load_content_list(mdir)
        m = meta.get(doc_id, {})
        category = (m.get("category") or "").strip() or doc_id.rsplit("_", 1)[0]
        doc_title = (m.get("title") or "").strip()
        lp = (m.get("local_path") or "").strip().replace("\\", "/")
        source_pdf = f"data/raw/{lp}" if lp else None
        report_date = m.get("report_date_iso")
        broker = (m.get("broker") or "").strip() or None

        counters: dict[str, int] = {}
        doc_rows: list[dict] = []

        for item in content:
            btype = item.get("type")
            if btype not in ("chart", "image", "table"):
                continue
            page = int(item.get("page_idx", 0)) + 1
            ckey = f"{page}:{btype}"
            counters[ckey] = counters.get(ckey, 0) + 1
            image_id = f"{doc_id}_p{page}_{btype}{counters[ckey]}"

            img_file = find_crop(Path(mdir), item)
            caption = get_caption(item)

            # ---- table: 판정하지 않고 인계 목록에만 (스펙 §2) ----
            if btype == "table":
                stats["total"] += 1
                crop_rel = rel_to_root(img_file) if img_file else None
                old = cards.get(image_id)
                if old and old.get("filter_stage") == "handoff_table" and not args.force:
                    stats["skipped"] += 1
                    continue
                handoff_table(image_id, doc_id, page, item, crop_rel, caption)
                card = _base_card(image_id, doc_id, category, page, btype, item,
                                  crop_rel, caption, source_pdf, report_date, broker,
                                  prompt_ver)
                card.update(status="handoff", filter_stage="handoff_table")
                _record(card, cards, doc_rows)
                stats["table_handoff"] += 1
                continue

            stats["total"] += 1

            # resume: 완료분 스킵
            old = cards.get(image_id)
            if old is not None and not args.force:
                st = old.get("status")
                redo = args.redo_errors and old.get("filter_stage") == "vlm_parse_error"
                done = st in ("useful", "discarded_rule", "discarded_vlm")
                if (done and not redo) or (st == "pending" and not vlm_ok):
                    stats["skipped"] += 1
                    continue

            if img_file is None:
                logger.info(f"  크롭 없음 — 스킵: {image_id}")
                stats["skipped"] += 1
                continue

            card = _base_card(image_id, doc_id, category, page, btype, item,
                              rel_to_root(img_file), caption, source_pdf, report_date,
                              broker, prompt_ver)

            # ---- [A] 규칙필터 ----
            reason = rule_filter(img_file, btype)
            if reason == "unreadable":
                logger.info(f"  크롭 손상 — 스킵: {image_id}")
                stats["skipped"] += 1
                continue
            if reason:
                dst = copy_crop(img_file, CFG["DISCARDED_DIR"] / "rule", doc_id, image_id)
                card.update(status="discarded_rule", filter_stage=f"rule:{reason}",
                            file=rel_to_root(dst))
                _record(card, cards, doc_rows)
                stats["discarded_rule"] += 1
                continue

            # ---- 해시 계산 (캐시·중복 키) ----
            try:
                raw = img_file.read_bytes()
            except Exception:
                stats["skipped"] += 1
                continue
            chash = common.content_hash(raw)
            phash = common.dhash(img_file)
            card["content_hash"] = chash
            card["phash"] = phash
            key = common.cache_key(chash, prompt_ver, model)

            if args.rules_only or not vlm_ok:
                card.update(status="pending", filter_stage="pending_vlm")
                _record(card, cards, doc_rows)
                stats["pending"] += 1
                continue

            # ---- [B] 캐시·중복 게이트 ----
            vlm = None
            source = "vlm"
            # B-1. content_hash 캐시 (L1)
            if use_cache:
                cached = common.cache_get("vlm", key)
                if cached is not None:
                    vlm = cached
                    source = "cache"
                    stats["cache_hit"] += 1
            # B-2. pHash 완전중복 → 판정 복사(dedup_of)
            if vlm is None:
                best_id, best_ham = None, 999
                for ph, c in phash_reg:
                    d = common.hamming(phash, ph)
                    if d < best_ham:
                        best_ham, best_id = d, c["image_id"]
                if best_id is not None and best_ham <= CFG["PHASH_DUP_MAX"]:
                    src_card = cards.get(best_id)
                    if src_card and src_card.get("vlm"):
                        vlm = dict(src_card["vlm"])
                        source = "dedup"
                        card["dedup_of"] = best_id
                        stats["dedup_exact"] += 1
                elif best_id is not None and best_ham <= CFG["PHASH_SIMILAR_MAX"]:
                    card["similar_of"] = best_id
                    card["similar_ham"] = best_ham
                    stats["similar_flagged"] += 1

            # ---- [S3] 그림 분류기 게이트 (--route clf) : 명백한 junk는 VLM 없이 컷 ----
            if vlm is None and clf_ok:
                import figure_classifier
                clf = figure_classifier.classify(img_file)
                if clf:
                    card["clf_label"] = clf["label"]
                    card["clf_conf"] = clf["confidence"]
                    if clf["route"] == "junk":
                        dst = copy_crop(img_file, CFG["DISCARDED_DIR"] / "clf", doc_id, image_id)
                        card.update(status="discarded_clf",
                                    filter_stage=f"clf_junk:{clf['label']}",
                                    file=rel_to_root(dst))
                        _record(card, cards, doc_rows)
                        if phash:
                            phash_reg.append((phash, card))
                        stats["discarded_clf"] += 1
                        continue

            # ---- [C] VLM 판정 ----
            if vlm is None:
                if args.limit is not None and vlm_calls >= args.limit:
                    logger.info(f"VLM 호출 상한({args.limit}) 도달 — 종료")
                    stop = True
                    break
                prompt = PROMPT_V2.format(caption=caption or "없음",
                                          doc_title=doc_title or "미상",
                                          category=category)
                t0 = time.time()
                res = common.ollama_chat(model, prompt, images=[str(img_file)],
                                         num_ctx=CFG["VLM_NUM_CTX"],
                                         img_max_edge=CFG["VLM_MAX_EDGE"],
                                         think=CFG["VLM_THINK"])
                dt = time.time() - t0
                vlm_calls += 1
                if res is None:
                    fail_streak += 1
                    logger.info(f"  VLM 호출 실패({fail_streak}) — 다음 실행 재시도: {image_id}")
                    stats["skipped"] += 1
                    if fail_streak >= 5:
                        logger.info("Ollama 5연속 실패 — 배치 중단")
                        stop = True
                        break
                    continue
                fail_streak = 0
                common.record_timing("s2_vlm", image_id, dt)
                if res.get("_parse_error"):
                    vlm = {"type": btype, "useful": btype == "chart", "confidence": 0.3,
                           "title": "", "ocr_text": str(res.get("_raw", ""))[:1000],
                           "summary": "", "entities": []}
                    card["filter_stage"] = "vlm_parse_error"
                    stats["parse_error"] += 1
                else:
                    vlm = normalize_vlm(res, btype)
                    if use_cache:
                        common.cache_put("vlm", key, vlm)

            # ---- [D] 저장 ----
            card["vlm"] = vlm
            card["cache_source"] = source
            conf = float(vlm.get("confidence", 0.5))
            if conf < CFG["CONF_THRESHOLD"]:
                card["review_queue"] = True
                stats["review_queue"] += 1
            vtype = vlm.get("type", "")
            is_table_image = vtype in _TABLE_IMAGE_TYPES
            # useful 결정: VLM useful 플래그 (type이 유용유형이면 보정)
            useful = bool(vlm.get("useful")) or vtype in _USEFUL_TYPES
            vlm["useful"] = useful

            if useful:
                dst = copy_crop(img_file, CFG["USEFUL_DIR"], doc_id, image_id)
                card.update(status="useful", file=rel_to_root(dst),
                            embed_text=build_embed_text(caption, vlm))
                if card.get("filter_stage") is None:
                    card["filter_stage"] = f"useful:{source}"
                # VLM이 표 이미지로 판정 → 인계 목록에도 이중 등록
                if is_table_image:
                    handoff_table(image_id, doc_id, page, item, card["file"],
                                  caption, source="vlm_reclass")
            else:
                dst = copy_crop(img_file, CFG["DISCARDED_DIR"] / "vlm", doc_id, image_id)
                card.update(status="discarded_vlm", file=rel_to_root(dst))
                if card.get("filter_stage") in (None, "vlm_parse_error"):
                    card["filter_stage"] = card.get("filter_stage") or f"discarded:{source}"

            _record(card, cards, doc_rows)
            if phash:
                phash_reg.append((phash, card))
            stats[card["status"]] += 1

        if doc_rows:
            try:
                common.upsert("image_cards", [_card_to_row(c) for c in doc_rows])
            except Exception as e:
                logger.info(f"  Supabase upsert 실패({doc_id}) — 로컬 JSONL만: {e}")

    _summary(stats, vlm_calls)


def _base_card(image_id, doc_id, category, page, btype, item, file_rel, caption,
               source_pdf, report_date, broker, prompt_ver) -> dict:
    return {
        "image_id": image_id, "doc_id": doc_id, "category": category,
        "page": page, "block_type": btype, "bbox": item.get("bbox") or None,
        "file": file_rel, "caption": caption, "content_hash": None, "phash": None,
        "dedup_of": None, "similar_of": None, "vlm": None, "cache_source": None,
        "confidence": None, "review_queue": False, "reviewed": False,
        "embed_text": "", "filter_stage": None, "prompt_ver": prompt_ver,
        "source_pdf": source_pdf, "report_date": report_date, "broker": broker,
        "storage_path": None, "status": "pending", "ts": common.now_iso(),
    }


def _record(card: dict, cards: dict, doc_rows: list) -> None:
    if card.get("vlm"):
        card["confidence"] = card["vlm"].get("confidence")
    common.append_jsonl(CFG["IMAGE_CARDS_JSONL"], card)
    cards[card["image_id"]] = card
    doc_rows.append(card)


def _card_to_row(card: dict) -> dict:
    """Supabase image_cards 컬럼 매핑 (스키마 §5). Supabase 미설정 시 미사용."""
    vlm = card.get("vlm") or {}
    return {
        "image_id": card["image_id"], "doc_id": card["doc_id"], "page": card["page"],
        "block_type": card["block_type"], "bbox": card.get("bbox"),
        "caption": card.get("caption") or None, "content_hash": card.get("content_hash"),
        "phash": card.get("phash"), "dedup_of": card.get("dedup_of"),
        "vlm_type": vlm.get("type"), "vlm_useful": vlm.get("useful"),
        "confidence": vlm.get("confidence"),
        "ocr_text": (vlm.get("ocr_text") or "")[:8000] or None,
        "summary": vlm.get("summary") or None, "entities": vlm.get("entities") or [],
        "embed_text": card.get("embed_text") or None,
        "review_queue": card.get("review_queue", False), "reviewed": False,
        "prompt_ver": card.get("prompt_ver"), "filter_stage": card.get("filter_stage"),
        "storage_path": card.get("storage_path"), "local_path": card.get("file"),
    }


def _summary(stats: dict, vlm_calls: int) -> None:
    logger.info("=== s2 고도화 완료 요약 ===")
    logger.info(
        f"총 {stats['total']} | useful {stats['useful']} | rule탈락 {stats['discarded_rule']} | "
        f"VLM탈락 {stats['discarded_vlm']} | table인계 {stats['table_handoff']} | "
        f"pending {stats['pending']}")
    logger.info(
        f"[고도화] VLM호출 {vlm_calls} | 캐시적중 {stats['cache_hit']} | 완전중복복사 {stats['dedup_exact']} | "
        f"유사표시 {stats['similar_flagged']} | review_queue {stats['review_queue']} | "
        f"파싱에러 {stats['parse_error']} | 스킵 {stats['skipped']}")
    if stats["discarded_clf"]:
        logger.info(f"[분류기] junk 선컷(VLM 미호출) {stats['discarded_clf']}건")


def main() -> None:
    p = argparse.ArgumentParser(description="s2 고도화: 규칙+캐시+pHash+VLM → image_cards")
    p.add_argument("--category", default=None, help="doc_id 접두 필터 (예: industry)")
    p.add_argument("--doc", default=None, help="특정 doc_id만 처리")
    p.add_argument("--limit", type=int, default=None, help="이번 실행 VLM 호출 상한")
    p.add_argument("--rules-only", action="store_true", help="규칙필터만, VLM 생략")
    p.add_argument("--route", choices=("full", "clf"), default="full",
                   help="full=모든 chart/image를 VLM / clf=그림분류기로 junk 선컷 후 VLM")
    p.add_argument("--no-cache", action="store_true", help="VLM 캐시 미사용(전량 재판정)")
    p.add_argument("--redo-errors", action="store_true", help="vlm_parse_error 카드 재판정")
    p.add_argument("--force", action="store_true", help="완료분도 재처리")
    p.add_argument("--parsed-root", default=None, help="파싱 결과 루트 (기본 data/parsed)")
    args = p.parse_args()
    process(args)


if __name__ == "__main__":
    main()
