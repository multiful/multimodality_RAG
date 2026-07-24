# -*- coding: utf-8 -*-
"""s2_onestop_mineru: MinerU 컴포넌트만으로 완결되는 원스톱 이미지 파이프라인.

의존 파일은 common.py(공용 유틸) · figure_classifier.py(선택 단계) · s1_parse.py(온디맨드 파싱)뿐 —
비전 LLM(Qwen3-VL 등) 판정에 의존하지 않는다. 전량 CPU 실행 가능(GPU 서버 불필요 — 배포 환경
CPU 전제로 검증됨. narrative 단계만 텍스트전용 LLM을 로컬 Ollama로 호출).

파이프라인 4단계 (뒤 두 단계는 선택):
  [1] 객체탐지  MinerU layout 모델 결과(middle.json para_blocks type·score)를 그대로 신뢰
  [2] OCR      크롭 내부 텍스트를 MinerU 내장 OCR(PytorchPaddleOCR)로 직접 추출 (~0.5s/장)
  [3] 세부분류  (--with-classifier, 기본 on) DocumentFigureClassifier-v2.5 로 26종 세부라벨.
               MinerU 탐지와 상호검증용 신호로만 병기 — 판정 자체는 바꾸지 않음(review_queue 후보만 표시)
  [4] 차트분석  (--with-chart-analysis, 기본 off — 초당 비용 큼) useful 판정된 chart 블록에 한해:
               4a. MinerU 내장 VLM(MinerU2.5-Pro-2605-1.2B)의 "Image Analysis" 로 근사 데이터 표 추출
                   ⚠ 이 모델은 자유 지시를 따르지 않는 좁은 특화모델 — prompts 커스터마이즈 시도는
                   무시되고 학습된 고정 패턴을 반복 생성하므로(실측 확인됨) 프롬프트를 바꾸지 말 것.
                   기본 max_new_tokens=None(무제한)이라 특정 차트에서 EOS 없이 폭주하는 버그를
                   실측으로 발견 — sampling_params로 상한을 반드시 건다(기본 1024).
               4b. 4a의 표를 텍스트전용 LLM(qwen3:8b, 비전 없음 → 훨씬 빠름 ~3.5s/장)에 다시 넣어
                   애널리스트 문체 서술형 해석으로 변환. 원본 이미지를 재확인하지 않으므로
                   4a의 근사치를 벗어난 디테일을 지어낼 위험이 있음(narrative는 참고용, 표가 근거).

판정 규칙 (기본 경로는 LLM 0회 호출):
  chart → useful (layout 탐지 신뢰, 스파크라인 보호를 위해 크기 무관 통과)
  image → 규칙필터 통과 후 OCR 텍스트 길이 ≥ OCR_MIN_CHARS 면 useful, 미만이면 discard(로고류)
  table → 인계(handoff)만 — 팀 규약 유지(판정 안 함, 텍스트/테이블 파트 소관)

PDF 1개 입력 → 파싱본 없으면 MinerU CLI 실행 → 카드 생성까지 원스톱.
resume 기본(이미 채워진 필드는 --force 없이 재계산 안 함) — 단계를 나중에 추가로 켜서
재실행해도 이미 끝난 단계는 스킵하고 새로 요청한 단계만 이어서 계산한다.

출력(기존 파이프라인과 분리): data/onestop/{doc_id}/
  onestop_cards.jsonl · useful/ · discarded/ · summary.json

카드 스키마 (다른 파이프라인이 참조할 안정 필드 vs 단계별 선택 필드):
  안정: image_id, doc_id, page, block_type, bbox, det_score, caption, footnote,
        status, filter_stage, embed_text, crop, ts
  [2]:  ocr {text, n_boxes, mean_conf, seconds}, ocr_lines[]
  [3]:  clf_label, clf_confidence, clf_route, clf_agree(bool, MinerU탐지와 일치 여부)
  [4]:  chart_table, chart_table_sec, narrative, narrative_sec
  [5]:  structured_metadata (--with-structured-output, image_type/entities_mentioned/
        described_content/key_values_or_trend/time_period — pdf_pipeline/structured_output.py의
        extract_image_metadata(), 텍스트/표 라우팅 끝과 동일한 OpenAI 구조화 출력 패턴)
  embed_text = caption+footnote+ocr(500자)까지는 항상 채움, narrative가 있으면 이어 붙임
  → 텍스트/RAG 파트는 embed_text만 읽으면 되고, 나머지는 파고들 때만 참조.

실행 환경: mineru[pipeline,vlm]가 설치된 파이썬으로 실행해야 한다
  (requirements.txt 설치 시 torch/torchvision/transformers 전부 포함 — --with-classifier 도 별도 venv 불필요).
  예) python pdf_pipeline/image_processing/s2_onestop_mineru.py --doc industry_15 --with-chart-analysis
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import common

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [39] pdf_pipeline/structured_output.py
from structured_output import add_structured_metadata_to_cards  # noqa: E402

CFG = common.CONFIG
logger = common.get_logger("s2_onestop_mineru")

ONESTOP_DIR = CFG["DATA_DIR"] / "onestop"
OCR_MIN_CHARS = 20          # image 크롭: OCR 텍스트가 이 미만이면 정보성 없음(로고류)으로 discard
OCR_MAX_EDGE = 1600         # OCR 입력 축소 상한 (과대 크롭 속도 보호)
CHART_VLM_MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"
CHART_VLM_LOCAL_DIR = Path.home() / ".cache" / "mineru_vlm" / "MinerU2.5-Pro-2605-1.2B"
CHART_MAX_NEW_TOKENS = 1024  # 폭주 방지 상한 (모듈 docstring 참조)

NARRATIVE_PROMPT = """아래는 한국 증권사 리서치 리포트 차트에서 MinerU가 자동 추출한 근사 데이터 표입니다.
[캡션: {caption}]
[추출된 표]
{table}

이 표를 애널리스트가 리포트 본문에 쓰듯 자연어로 해석하세요. 반드시 포함:
- 시기별 대략적인 평균 수준(근사 수치)
- 시기별 추세(보합/급등/급락)
- 추세가 바뀌는 변곡점과 전후 비교
표·목록·마크다운 없이 자연스러운 한국어 문장 2~4개로만 답하세요."""


# ---------------------------------------------------------------- 크롭/필터 유틸

def find_crop(mdir: Path, item: dict) -> Path | None:
    """content_list 항목의 img_path로 실제 크롭 파일을 찾는다 (경로 불일치 시 파일명으로 재탐색)."""
    rel = item.get("img_path") or item.get("image_path") or ""
    if not isinstance(rel, str) or not rel:
        return None
    p = Path(mdir) / rel
    if p.exists():
        return p
    for cand in Path(mdir).rglob(Path(rel).name):
        return cand
    return None


def rule_filter(img_file: Path, block_type: str) -> str | None:
    """[1] 규칙필터. 탈락사유 문자열 또는 통과 None. chart 는 크기 무관 통과(스파크라인 보호)."""
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


# ---------------------------------------------------------------- MinerU 신호 로드

def load_block_scores(mdir: Path) -> dict[tuple[int, str, int], dict]:
    """middle.json para_blocks → {(page, type, occurrence): {score, bbox}}.

    content_list 항목에는 탐지 score·bbox가 없으므로, 같은 (페이지, 타입) 안에서의
    등장 순서로 매칭한다 (둘 다 MinerU 읽기 순서라 순서가 일치)."""
    middle = common.load_middle_json(mdir) or {}
    out: dict[tuple[int, str, int], dict] = {}
    for p in middle.get("pdf_info", []):
        page = int(p.get("page_idx", 0)) + 1
        counters: dict[str, int] = {}
        for b in p.get("para_blocks", []):
            btype = b.get("type")
            if btype not in ("chart", "image", "table"):
                continue
            counters[btype] = counters.get(btype, 0) + 1
            out[(page, btype, counters[btype])] = {
                "score": b.get("score"), "bbox": b.get("bbox")}
    return out


def get_texts(item: dict) -> tuple[str, str]:
    """content_list 항목의 (캡션, 각주) — 파싱 때 MinerU OCR가 이미 추출한 텍스트."""
    cap_parts: list[str] = []
    foot_parts: list[str] = []
    for t in ("chart", "image", "table"):
        v = item.get(f"{t}_caption")
        if isinstance(v, list):
            cap_parts.extend(str(x) for x in v)
        v = item.get(f"{t}_footnote")
        if isinstance(v, list):
            foot_parts.extend(str(x) for x in v)
    return (common.clean_text(" ".join(cap_parts)),
            common.clean_text(" ".join(foot_parts)))


def build_embed_text(caption: str, footnote: str, ocr_text: str, narrative: str | None) -> str:
    parts = [caption, footnote, (ocr_text or "")[:500]]
    if narrative:
        parts.append(narrative)
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------- [2] MinerU OCR

_ocr_engine = None


def get_ocr(lang: str):
    """MinerU 내장 OCR(PytorchPaddleOCR) 싱글턴 — s1 파싱과 동일한 모델·가중치."""
    global _ocr_engine
    if _ocr_engine is None:
        t0 = time.time()
        from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
        _ocr_engine = PytorchPaddleOCR(lang=lang)
        logger.info(f"MinerU OCR 로드 완료 (lang={lang}, {time.time()-t0:.1f}s)")
    return _ocr_engine


def ocr_crop(img_file: Path, lang: str) -> dict:
    """크롭 1장을 MinerU OCR로 → {text, lines, n_boxes, mean_conf, seconds}.

    [수정] common.py에 이미 구현돼 있었지만 어디서도 안 쓰이던 content-hash 캐시(cache_get/
    cache_put, VLM 캐시 L2 키용으로 설계된 것)를 여기서 처음 배선 — 크롭 픽셀 내용이 같으면
    (같은 브로커의 반복되는 템플릿 차트/워터마크 등, 문서가 달라도) OCR을 다시 안 돌린다.
    같은 문서 재실행 시의 카드 단위 resume(호출측의 `old_cards`)과는 별개 계층 — 이건 "다른
    문서/다른 이미지_id라도 크롭 픽셀이 같으면" 캐시가 걸린다."""
    import cv2
    import numpy as np
    chash = common.content_hash(img_file)
    key = common.cache_key(chash, "ocr_v1", lang)
    cached = common.cache_get("ocr", key)
    if cached is not None:
        return cached
    engine = get_ocr(lang)
    t0 = time.time()
    img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"text": "", "lines": [], "n_boxes": 0, "mean_conf": None,
                "seconds": 0.0, "error": "unreadable"}
    h, w = img.shape[:2]
    if max(h, w) > OCR_MAX_EDGE:
        s = OCR_MAX_EDGE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    res = engine.ocr(img)
    dt = time.time() - t0
    lines: list[dict] = []
    for entry in (res[0] or []) if res else []:
        box, rec = entry
        text, conf = (rec[0], float(rec[1])) if isinstance(rec, (list, tuple)) else (str(rec), None)
        text = common.clean_text(str(text))
        if text:
            lines.append({"text": text, "conf": round(conf, 3) if conf is not None else None})
    confs = [l["conf"] for l in lines if l["conf"] is not None]
    result = {"text": " ".join(l["text"] for l in lines),
              "lines": lines, "n_boxes": len(lines),
              "mean_conf": round(sum(confs) / len(confs), 3) if confs else None,
              "seconds": round(dt, 2)}
    common.cache_put("ocr", key, result)
    return result


# ---------------------------------------------------------------- [3] 그림 분류기 (선택)

def classify_crop(img_file: Path) -> dict | None:
    """DocumentFigureClassifier 라벨 (별도 로직 변경 없이 신호만 병기)."""
    import figure_classifier as fc
    if not fc.available():
        return None
    return fc.classify(img_file)


_CHART_LABELS = {"line_chart", "bar_chart", "pie_chart", "scatter_plot", "box_plot"}


def clf_agrees_with_mineru(clf: dict | None, block_type: str) -> bool | None:
    if clf is None:
        return None
    return (block_type == "chart") == (clf.get("label") in _CHART_LABELS)


# ---------------------------------------------------------------- [4a] MinerU 내장 VLM (차트 표추출)

_chart_vlm_client = None


def get_chart_vlm(max_new_tokens: int):
    """MinerU2.5-Pro VLM 싱글턴. 로컬 캐시 없으면 최초 1회 다운로드(symlink 회피 위해 local_dir 사용)."""
    global _chart_vlm_client
    if _chart_vlm_client is not None:
        return _chart_vlm_client
    from huggingface_hub import snapshot_download
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.mineru_client import MinerUSamplingParams

    t0 = time.time()
    if not (CHART_VLM_LOCAL_DIR / "config.json").exists():
        logger.info(f"MinerU VLM 최초 다운로드: {CHART_VLM_MODEL_ID} → {CHART_VLM_LOCAL_DIR}")
        snapshot_download(CHART_VLM_MODEL_ID, local_dir=str(CHART_VLM_LOCAL_DIR))

    # ⚠ prompts는 절대 커스터마이즈하지 말 것 — 이 모델은 자유 지시를 무시하고
    # 학습된 고정 패턴(특수토큰 포함)을 반복 생성한다(실측 확인, 모듈 docstring 참조).
    # sampling_params만 오버라이드해 무제한 생성(EOS 없는 폭주) 버그를 방지한다.
    sp = MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05,
                              max_new_tokens=max_new_tokens)
    _chart_vlm_client = MinerUClient(
        backend="transformers", model_path=str(CHART_VLM_LOCAL_DIR),
        sampling_params={"chart": sp, "image": sp, "[default]": sp},
        image_analysis=True, use_tqdm=False,
    )
    logger.info(f"MinerU VLM 로드 완료 ({time.time()-t0:.1f}s)")
    return _chart_vlm_client


def chart_table_extract(img_file: Path, block_type: str, max_new_tokens: int) -> tuple[str | None, float]:
    """[수정] content-hash 캐시 배선(common.cache_get/cache_put, [4a]가 이 파이프라인에서 가장
    비싼 단계 — 15.5초/장) — 같은 크롭이 다른 문서에서 다시 나오면(반복 템플릿 차트 등) 재호출
    안 함. 실패(None) 결과는 캐시하지 않는다(일시적 실패를 영구 캐시하면 재시도 기회를 잃음)."""
    chash = common.content_hash(img_file)
    key = common.cache_key(chash, f"chart_extract_v1:{block_type}:{max_new_tokens}", CHART_VLM_MODEL_ID)
    cached = common.cache_get("chart_table", key)
    if cached is not None:
        return cached["table"], cached["seconds"]

    from PIL import Image
    client = get_chart_vlm(max_new_tokens)
    im = Image.open(img_file).convert("RGB")
    t0 = time.time()
    try:
        out = client.content_extract(im, type=block_type)
    except Exception as e:
        logger.info(f"  MinerU VLM 호출 실패: {e}")
        return None, round(time.time() - t0, 2)
    dt = round(time.time() - t0, 2)
    table = str(out) if out else None
    if table:
        common.cache_put("chart_table", key, {"table": table, "seconds": dt})
    return table, dt


# ---------------------------------------------------------------- [4b] 서술형 해석 (텍스트전용 LLM)

def narrative_from_table(caption: str, table: str, model: str) -> tuple[str | None, float]:
    """[수정] content-hash 캐시 배선 — 여기서는 이미지가 아니라 (caption, table) 텍스트 내용이
    입력이므로 그 문자열을 해시한다(3.5초/장, [4a]만큼 크진 않지만 표가 같으면 서술도 결정적으로
    같아야 하므로 재호출 비용을 아낄 수 있음)."""
    chash = common.content_hash(f"{caption or ''}|{table}".encode("utf-8"))
    key = common.cache_key(chash, "narrative_v1", model)
    cached = common.cache_get("narrative", key)
    if cached is not None:
        return cached["narrative"], cached["seconds"]

    prompt = NARRATIVE_PROMPT.format(caption=caption or "없음", table=table)
    t0 = time.time()
    res = common.ollama_chat(model, prompt, images=None, expect_json=False,
                             num_ctx=CFG["VLM_NUM_CTX"], think=False)
    dt = round(time.time() - t0, 2)
    if res:
        common.cache_put("narrative", key, {"narrative": res, "seconds": dt})
    return res, dt


# ---------------------------------------------------------------- 파싱 보장 (원스톱)

def ensure_parsed(doc_id: str, timeout_sec: int) -> Path | None:
    """파싱본이 있으면 재사용, 없으면 MinerU CLI로 파싱 (s1 로직 재사용)."""
    for did, mdir in common.find_parsed_docs():
        if did == doc_id:
            logger.info(f"파싱본 재사용: {mdir}")
            return Path(mdir)
    rows = [r for r in common.read_metadata() if r["doc_id"] == doc_id]
    if not rows:
        logger.info(f"metadata.csv에 {doc_id} 없음 — 파싱 불가")
        return None
    logger.info(f"파싱본 없음 → MinerU CLI 파싱 시작: {rows[0]['pdf_abs']}")
    import s1_parse
    t0 = time.time()
    out = s1_parse.parse_doc(rows[0], timeout_sec)
    logger.info(f"MinerU 파싱 완료 ({time.time()-t0:.1f}s): {out}")
    return out


# ---------------------------------------------------------------- 메인

def process(args: argparse.Namespace) -> None:
    doc_id = args.doc
    mdir = ensure_parsed(doc_id, args.timeout_sec)
    if mdir is None:
        return
    content = common.load_content_list(mdir)
    scores = load_block_scores(mdir)
    meta = {r["doc_id"]: r for r in common.read_metadata()}.get(doc_id, {})

    out_dir = ONESTOP_DIR / doc_id
    (out_dir / "useful").mkdir(parents=True, exist_ok=True)
    (out_dir / "discarded").mkdir(parents=True, exist_ok=True)
    cards_path = out_dir / "onestop_cards.jsonl"
    old_cards = {} if args.force else common.jsonl_index(cards_path, "image_id")

    stats = {k: 0 for k in ("total", "useful", "discarded_rule", "discarded_ocr",
                            "table_handoff", "skipped", "clf_disagree",
                            "chart_analyzed", "chart_analysis_failed")}
    ocr_secs: list[float] = []
    counters: dict[str, int] = {}
    rows: list[dict] = []
    t_all = time.time()
    fail_streak = 0

    for item in content:
        btype = item.get("type")
        if btype not in ("chart", "image", "table"):
            continue
        page = int(item.get("page_idx", 0)) + 1
        ckey = f"{page}:{btype}"
        counters[ckey] = counters.get(ckey, 0) + 1
        occ = counters[ckey]
        image_id = f"{doc_id}_p{page}_{btype}{occ}"
        stats["total"] += 1

        old = old_cards.get(image_id, {})
        img_file = find_crop(Path(mdir), item)
        caption, footnote = get_texts(item)
        det = scores.get((page, btype, occ), {})
        card = dict(old) if old else {
            "image_id": image_id, "doc_id": doc_id, "page": page,
            "block_type": btype, "det_score": det.get("score"),
            "bbox": det.get("bbox"), "caption": caption, "footnote": footnote,
            "crop": str(img_file) if img_file else None,
            "ocr": None, "ocr_lines": None, "clf_label": None, "clf_confidence": None,
            "clf_route": None, "clf_agree": None, "chart_table": None,
            "chart_table_sec": None, "narrative": None, "narrative_sec": None,
            "status": None, "filter_stage": None, "embed_text": "",
            "route": "onestop_mineru", "ts": common.now_iso(),
        }

        if btype == "table":
            card.update(status="handoff", filter_stage="handoff_table")
            rows.append(card)
            stats["table_handoff"] += 1
            continue
        if img_file is None:
            card.update(status="skipped", filter_stage="no_crop")
            rows.append(card)
            stats["skipped"] += 1
            continue

        # ---- [1] 규칙필터 (chart는 크기 무관 통과) ----
        resumed = old.get("status") in ("useful", "discarded_rule", "discarded_ocr")
        if resumed:
            pass  # 이미 판정 완료 — 규칙필터·OCR 재실행 안 함(resume)
        else:
            reason = rule_filter(img_file, btype)
            if reason:
                dst = out_dir / "discarded" / f"{image_id}.jpg"
                shutil.copy2(img_file, dst)
                card.update(status="discarded_rule", filter_stage=f"rule:{reason}")
                rows.append(card)
                stats["discarded_rule"] += 1
                continue

            # ---- [2] MinerU OCR ----
            ocr = ocr_crop(img_file, args.lang)
            ocr_secs.append(ocr["seconds"])
            card["ocr"] = {k: ocr[k] for k in ("text", "n_boxes", "mean_conf", "seconds")}
            card["ocr_lines"] = ocr["lines"]

            useful = btype == "chart" or len(ocr["text"]) >= OCR_MIN_CHARS
            if useful:
                dst = out_dir / "useful" / f"{image_id}.jpg"
                shutil.copy2(img_file, dst)
                card.update(status="useful",
                            filter_stage="onestop:chart" if btype == "chart" else "onestop:image_ocr")
            else:
                dst = out_dir / "discarded" / f"{image_id}.jpg"
                shutil.copy2(img_file, dst)
                card.update(status="discarded_ocr", filter_stage=f"onestop:ocr_lt_{OCR_MIN_CHARS}")
        stats[card["status"]] += 1

        # ---- [3] 그림 분류기 (선택, 저장 게이트에는 영향 없음 — 신호만 병기) ----
        if args.with_classifier and card.get("clf_label") is None:
            clf = classify_crop(img_file)
            if clf:
                card["clf_label"] = clf["label"]
                card["clf_confidence"] = clf["confidence"]
                card["clf_route"] = clf["route"]
                agree = clf_agrees_with_mineru(clf, btype)
                card["clf_agree"] = agree
                if agree is False:
                    stats["clf_disagree"] += 1

        # ---- [4] 차트 분석 (선택, useful chart만) ----
        if (args.with_chart_analysis and card["status"] == "useful" and btype == "chart"
                and card.get("chart_table") is None and fail_streak < 5):
            table, t_sec = chart_table_extract(img_file, btype, args.chart_max_new_tokens)
            card["chart_table"], card["chart_table_sec"] = table, t_sec
            if table:
                fail_streak = 0
                narr, n_sec = narrative_from_table(caption, table, args.narrative_model)
                card["narrative"], card["narrative_sec"] = narr, n_sec
                stats["chart_analyzed"] += 1
            else:
                fail_streak += 1
                stats["chart_analysis_failed"] += 1
                if fail_streak >= 5:
                    logger.info("MinerU VLM 5연속 실패 — 차트분석 단계 중단(이후는 표만 비움)")

        card["embed_text"] = build_embed_text(card.get("caption", ""), card.get("footnote", ""),
                                              (card.get("ocr") or {}).get("text", ""),
                                              card.get("narrative"))
        rows.append(card)

    # ---- [5] 구조화 출력 (선택, --with-structured-output, useful 카드만) ----
    if args.with_structured_output:
        need = [c for c in rows if c["status"] == "useful" and (args.force or not c.get("structured_metadata"))]
        if need:
            t_so = time.time()
            updated_by_id = {c["image_id"]: c for c in add_structured_metadata_to_cards(need)}
            rows = [updated_by_id.get(c["image_id"], c) for c in rows]
            logger.info(f"[구조화출력] {len(need)}건 처리 ({time.time() - t_so:.1f}s)")

    common.write_jsonl(cards_path, rows)

    total_dt = time.time() - t_all
    n_ocr = len(ocr_secs)
    summary = {
        "doc_id": doc_id, "title": (meta.get("title") or "").strip(),
        "stats": stats, "ocr_calls": n_ocr,
        "ocr_sec_mean": round(sum(ocr_secs) / n_ocr, 2) if n_ocr else None,
        "wall_sec": round(total_dt, 1), "ocr_min_chars": OCR_MIN_CHARS,
        "with_classifier": args.with_classifier, "with_chart_analysis": args.with_chart_analysis,
        "lang": args.lang, "ts": common.now_iso(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=== 원스톱(MinerU 전담) 완료 ===")
    logger.info(f"총 {stats['total']} | useful {stats['useful']} | rule탈락 {stats['discarded_rule']} | "
                f"OCR탈락 {stats['discarded_ocr']} | table인계 {stats['table_handoff']} | 스킵 {stats['skipped']}")
    if args.with_classifier:
        logger.info(f"[분류기] MinerU탐지 불일치 {stats['clf_disagree']}건 → review_queue 후보")
    if args.with_chart_analysis:
        logger.info(f"[차트분석] 성공 {stats['chart_analyzed']} / 실패 {stats['chart_analysis_failed']}")
    logger.info(f"전체 {total_dt:.1f}s | 출력: {out_dir}")


def export_txt(doc_id: str) -> Path:
    """카드에서 캡션+서술형+원본표를 사람이 읽기 좋은 텍스트로 뽑는다 (--export-txt)."""
    out_dir = ONESTOP_DIR / doc_id
    cards = common.load_jsonl(out_dir / "onestop_cards.jsonl")
    cards = [c for c in cards if c["block_type"] in ("chart", "image")]
    out_path = out_dir / "vlm_chart_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{doc_id} — MinerU 원스톱 차트 분석\n")
        f.write("=" * 70 + "\n\n")
        for c in sorted(cards, key=lambda x: (x["page"], x["image_id"])):
            f.write(f"[{c['image_id']}]  p.{c['page']}  ({c['block_type']})\n")
            if c.get("caption"):
                f.write(f"캡션: {c['caption']}\n")
            f.write("-" * 70 + "\n")
            if c.get("narrative"):
                f.write("[서술형 해석]\n" + c["narrative"].strip() + "\n\n")
            if c.get("chart_table"):
                f.write("[MinerU 추출표 원본]\n" + c["chart_table"].strip() + "\n")
            elif not c.get("narrative"):
                f.write(f"(status={c['status']} — 차트분석 미실행)\n")
            f.write("\n" + "=" * 70 + "\n\n")
    logger.info(f"텍스트 리포트 생성: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="원스톱: MinerU 컴포넌트만으로 이미지 카드 생성 (CPU 배포 전제)")
    p.add_argument("--doc", required=True, help="doc_id (예: industry_15)")
    p.add_argument("--lang", default="korean", help="MinerU OCR 언어 (기본 korean)")
    p.add_argument("--timeout-sec", type=int, default=1800, help="MinerU 파싱 타임아웃")
    p.add_argument("--with-classifier", dest="with_classifier", action="store_true", default=True,
                   help="[3] 그림 분류기 병기 (기본 on, 장당 ~10ms로 사실상 무료)")
    p.add_argument("--no-classifier", dest="with_classifier", action="store_false")
    p.add_argument("--with-chart-analysis", action="store_true",
                   help="[4] MinerU VLM 표추출+서술형해석 (기본 off, chart당 ~19초로 비용 큼)")
    p.add_argument("--chart-max-new-tokens", type=int, default=CHART_MAX_NEW_TOKENS,
                   help="MinerU VLM 생성 토큰 상한 (무제한 시 폭주 버그 있음, 기본 1024)")
    p.add_argument("--narrative-model", default=CFG["LLM_MODEL"],
                   help="서술형 해석용 텍스트전용 LLM (기본 config의 LLM_MODEL)")
    p.add_argument("--with-structured-output", action="store_true",
                   help="[5] 텍스트/표 라우팅과 동일한 OpenAI 구조화 출력(structured_output.py)을 "
                        "useful 카드에 적용 (기본 off, 유료 API + [4]처럼 카드당 호출 비용 있음)")
    p.add_argument("--force", action="store_true", help="완료분도 전 단계 재계산")
    p.add_argument("--export-txt", action="store_true", help="완료 후 사람이 읽는 .txt 리포트도 생성")
    args = p.parse_args()
    common.ensure_dirs()
    process(args)
    if args.export_txt:
        export_txt(args.doc)


if __name__ == "__main__":
    main()
