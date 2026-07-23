# -*- coding: utf-8 -*-
"""s1_parse_batch: metadata.csv의 전체 PDF를 MinerU CLI로 배치 파싱해
data/parsed/{doc_id}/ 에 평탄화 저장하고 Supabase documents 테이블에 기록한다."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

import common

logger = common.get_logger("s1_parse_batch")


def row_category(row: dict) -> str:
    """metadata의 category 컬럼 우선, 없으면 doc_id 접두어('company_01'→'company')."""
    return (row.get("category") or "").strip() or row["doc_id"].rsplit("_", 1)[0]


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """doc_id 충돌 감지 — 같은 doc_id에 서로 다른 pdf_url이면 경고 후 뒤엣것 스킵."""
    seen: dict[str, str] = {}
    out: list[dict] = []
    for row in rows:
        doc_id = row["doc_id"]
        url = (row.get("pdf_url") or "").strip()
        if doc_id in seen:
            if url and url != seen[doc_id]:
                logger.warning(f"doc_id 충돌: {doc_id} — pdf_url 상이, 뒤엣것 스킵 ({url})")
            continue
        seen[doc_id] = url
        out.append(row)
    return out


def preregister_documents(rows: list[dict]) -> None:
    """documents 선등록 (parse_ok/parsed_at 미포함 → 기존 값 보존)."""
    docs = [{
        "doc_id": r["doc_id"],
        "category": row_category(r),
        "title": (r.get("title") or "").strip() or None,
        "broker": (r.get("broker") or "").strip() or None,
        "report_date": r.get("report_date_iso"),
        "pdf_url": (r.get("pdf_url") or "").strip() or None,
        "nid": (r.get("nid") or "").strip() or None,
        "local_path": (r.get("local_path") or "").strip() or None,
    } for r in rows]
    n = common.upsert("documents", docs, on_conflict="doc_id")
    logger.info(f"documents 선등록: {len(docs)}건 (DB upsert {n}건)")


def record_failure(doc_id: str, pdf_path: str, reason: str) -> None:
    """실패 1건을 parse_failures.csv에 append하고 parse_ok=False upsert."""
    path = common.CONFIG["PARSE_FAILURES_CSV"]
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["doc_id", "pdf_path", "reason", "ts"])
        w.writerow([doc_id, pdf_path, reason.replace("\n", " ").strip(), common.now_iso()])
    common.upsert("documents", [{"doc_id": doc_id, "parse_ok": False}], on_conflict="doc_id")
    logger.info(f"실패 기록: {doc_id} — {reason[:200]}")


def _fs_retry(fn, tries: int = 5, delay: float = 1.0):
    """파일시스템 작업을 짧은 백오프로 재시도 (OneDrive/인덱서 일시 잠금 대응)."""
    for attempt in range(tries):
        try:
            return fn()
        except (PermissionError, OSError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def parse_doc(row: dict, timeout_sec: int) -> Path:
    """PDF 1개를 MinerU로 파싱해 data/parsed/{doc_id}/ 로 평탄화 이동. 실패 시 예외."""
    doc_id = row["doc_id"]
    pdf = Path(row["pdf_abs"])
    if not pdf.exists():
        raise FileNotFoundError(f"PDF 없음: {pdf}")

    # a. 경로 길이·한글 파일명 회피 — _tmp/{doc_id}.pdf 로 복사
    tmp_root = common.CONFIG["PARSED_DIR"] / "_tmp"
    tmp_out = tmp_root / "out"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_pdf = tmp_root / f"{doc_id}.pdf"
    shutil.copyfile(pdf, tmp_pdf)
    if (tmp_out / doc_id).exists():  # 이전 실행 잔여물 제거
        shutil.rmtree(tmp_out / doc_id, ignore_errors=True)

    # b. MinerU CLI 실행 (페이지 제한 없음)
    env = os.environ.copy()
    env.setdefault("MINERU_MODEL_SOURCE", "huggingface")
    cmd = [common.CONFIG["MINERU_EXE"], "-p", str(tmp_pdf), "-o", str(tmp_out),
           "-b", "pipeline", "-l", "korean"]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout_sec)
    finally:
        tmp_pdf.unlink(missing_ok=True)
    if r.returncode != 0:
        tail = ((r.stdout or "")[-500:] + " | " + (r.stderr or "")[-500:]).strip()
        raise RuntimeError(f"returncode={r.returncode}: {tail}")

    # c. tmp_out/{doc_id}/auto/ (또는 평면 구조) → data/parsed/{doc_id}/ 평탄화 이동
    src = None
    for pat in (f"{doc_id}/*/", f"{doc_id}/"):
        for d in sorted(tmp_out.glob(pat)):
            if list(Path(d).glob("*middle.json")):
                src = Path(d)
                break
        if src:
            break
    if src is None:
        raise RuntimeError("MinerU 출력에 middle.json 없음")
    dest = common.CONFIG["PARSED_DIR"] / doc_id
    # OneDrive 동기화/인덱서가 새로 만든 폴더를 잠깐 잠가 rmtree/move가 WinError 5로 실패할 수 있어
    # 짧은 백오프로 재시도한다 (일시적 잠금이라 대개 1~2초면 풀림).
    _fs_retry(lambda: shutil.rmtree(dest) if dest.exists() else None)
    _fs_retry(lambda: shutil.move(str(src), str(dest)))
    shutil.rmtree(tmp_out / doc_id, ignore_errors=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="metadata.csv의 PDF 전체를 MinerU로 배치 파싱 (resume 안전)")
    ap.add_argument("--limit", type=int, default=0, help="테스트용 문서 수 제한")
    ap.add_argument("--category", choices=common.CONFIG["CATEGORIES"],
                    help="해당 카테고리만 파싱")
    ap.add_argument("--timeout-sec", type=int, default=1800,
                    help="문서당 MinerU 타임아웃 초 (기본 1800)")
    ap.add_argument("--force", metavar="DOC_ID", default=None,
                    help="해당 문서는 완료분이어도 재파싱")
    args = ap.parse_args()

    common.ensure_dirs()
    rows = common.read_metadata()
    if not rows:
        logger.info(f"metadata.csv가 없거나 비어 있습니다: {common.CONFIG['METADATA_CSV']}")
        logger.info("collect_naver_research.ps1 수집을 먼저 실행하세요.")
        return

    rows = dedupe_rows(rows)
    preregister_documents(rows)  # 전 행 선등록 (필터 무관)

    targets = [r for r in rows if not args.category or row_category(r) == args.category]
    if args.limit and args.limit > 0:
        targets = targets[:args.limit]

    n = len(targets)
    n_ok = n_fail = n_skip = 0
    logger.info(f"파싱 대상 {n}건 (category={args.category or '전체'}, "
                f"timeout={args.timeout_sec}s, force={args.force or '없음'})")

    for i, row in enumerate(targets, 1):
        doc_id = row["doc_id"]
        dest = common.CONFIG["PARSED_DIR"] / doc_id
        if args.force != doc_id and list(dest.glob("*middle.json")):
            n_skip += 1
            common.log_progress(logger, i, n, f"{doc_id} 스킵 (완료분)")
            continue
        common.log_progress(logger, i, n, f"{doc_id} 파싱 시작")
        t0 = time.time()
        try:
            out_dir = parse_doc(row, args.timeout_sec)
            middle = common.load_middle_json(out_dir)
            pages = len((middle or {}).get("pdf_info", []))
            dt = time.time() - t0
            common.upsert("documents",
                          [{"doc_id": doc_id, "parse_ok": True,
                            "parsed_at": common.now_iso(), "pages": pages}],
                          on_conflict="doc_id")
            common.record_timing("s1_parse", doc_id, dt)
            n_ok += 1
            common.log_progress(logger, i, n, f"{doc_id} 완료 ({pages}p, {dt:.1f}s)")
        except subprocess.TimeoutExpired:
            n_fail += 1
            record_failure(doc_id, row["pdf_abs"], f"timeout>{args.timeout_sec}s")
        except Exception as e:
            n_fail += 1
            record_failure(doc_id, row["pdf_abs"], f"{type(e).__name__}: {e}"[:300])

    shutil.rmtree(common.CONFIG["PARSED_DIR"] / "_tmp", ignore_errors=True)
    logger.info(f"종료 — 성공 {n_ok} / 실패 {n_fail} / 스킵 {n_skip} (전체 {n})")
    if n_fail:
        logger.info(f"실패 목록: {common.CONFIG['PARSE_FAILURES_CSV']}")


if __name__ == "__main__":
    main()
