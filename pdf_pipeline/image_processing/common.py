# -*- coding: utf-8 -*-
"""pdfex 파이프라인 공통 유틸.

경로/설정(CONFIG), Supabase 클라이언트·upsert·resume, Ollama 호출,
BGE-M3 임베딩, Kiwi 토큰화, 로컬 dense 폴백 검색, RRF, 로깅/타이밍.

모든 s*_*.py 스크립트는 이 모듈만 통해 외부 자원에 접근한다.
Supabase 미설정(.env 없음) 시에도 로컬 파일 기반으로 동작하도록
sb()가 None을 반환하면 DB 쓰기는 건너뛴다 (로컬 JSONL이 원본).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows 콘솔(cp949)에서 한글/이모지 출력 깨짐 방지
for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 환경 제약: torch(cu128 nightly)가 로드된 뒤 pandas→pyarrow 네이티브 확장을 로드하면
# Windows에서 access violation으로 프로세스가 죽는다 (sentence_transformers 임포트 크래시).
# 모든 스크립트가 common을 가장 먼저 임포트하므로 여기서 pyarrow/pandas를 선로드해 회피한다.
try:
    import pyarrow  # noqa: F401
    import pandas  # noqa: F401
except Exception:
    pass

def _resolve_project_root() -> Path:
    """[수정 — 재일] 데이터 루트를 "실제로 산출물이 있는 곳"으로 찾는다.

    발견한 버그: 이 모듈이 리포 루트에서 `pdf_pipeline/image_processing/` 아래로 옮겨지면서
    `parent.parent`가 리포 루트가 아니라 `pdf_pipeline/`을 가리키게 됐다. 그런데 실제 산출물
    (`data/onestop`, `data/parsed`, `data/raw`, `data/images`)은 여전히 **리포 루트**에 있어서,
    코드는 텅 빈 `pdf_pipeline/data/`만 쳐다보고 있었다. 결과로 (a) 이미 만들어둔 onestop 카드
    (industry_15: 차트 105장 중 chart_table/narrative 103건 완비)를 아무도 못 읽고, (b) 데모가
    "onestop_cards.jsonl 없음"으로 판단해 **LGCNS 예시 카드로 폴백**했으며, (c) s2를 다시 돌려도
    `data/parsed`의 기존 파싱본을 재사용 못 하고 처음부터 파싱하게 된다.

    고정 경로 대신, 후보를 위로 훑으면서 `data/` 밑에 실제 산출물 디렉토리가 있는 쪽을 고른다.
    `PIPELINE_DATA_ROOT` 환경변수로 명시 지정도 가능(배포/테스트용)."""
    env = os.environ.get("PIPELINE_DATA_ROOT")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    here = Path(__file__).resolve().parent.parent          # pdf_pipeline/
    for cand in (here, here.parent):                       # pdf_pipeline/ -> 리포 루트
        data = cand / "data"
        if data.is_dir() and any((data / s).is_dir() for s in ("onestop", "parsed", "raw", "images")):
            return cand
    return here                                            # 둘 다 없으면 기존 동작 유지


PROJECT_ROOT = _resolve_project_root()


def load_env() -> None:
    """`.env`를 1회 로드 (없으면 조용히 무시)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


load_env()

# MinerU 실행파일 탐색: 환경변수 > 로컬 venv > PATH
def _find_mineru() -> str:
    env = os.environ.get("MINERU_EXE")
    if env and Path(env).exists():
        return env
    cands = [
        PROJECT_ROOT / "demo_venv" / "Scripts" / "mineru.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "mineru.exe",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return "mineru"


_mineru_exe_str = _find_mineru()

CONFIG = {
    "DATA_DIR": PROJECT_ROOT / "data",
    "RAW_DIR": PROJECT_ROOT / "data" / "raw",
    "PARSED_DIR": PROJECT_ROOT / "data" / "parsed",
    "IMAGES_DIR": PROJECT_ROOT / "data" / "images",
    "USEFUL_DIR": PROJECT_ROOT / "data" / "images" / "useful",
    "DISCARDED_DIR": PROJECT_ROOT / "data" / "images" / "discarded",
    "CACHE_DIR": PROJECT_ROOT / "data" / "cache",
    "HANDOFF_DIR": PROJECT_ROOT / "data" / "handoff",
    "CHUNKS_DIR": PROJECT_ROOT / "data" / "chunks",
    "TABLES_DIR": PROJECT_ROOT / "data" / "tables",
    "ENRICH_DIR": PROJECT_ROOT / "data" / "enrich",
    "BM25_DIR": PROJECT_ROOT / "db" / "bm25",
    "DENSE_DIR": PROJECT_ROOT / "db" / "dense",
    "LOGS_DIR": PROJECT_ROOT / "logs",
    "EVAL_DIR": PROJECT_ROOT / "eval",
    "METADATA_CSV": PROJECT_ROOT / "data" / "raw" / "metadata.csv",
    "IMAGE_CARDS_JSONL": PROJECT_ROOT / "data" / "images" / "image_cards.jsonl",
    "HANDOFF_TABLES_JSONL": PROJECT_ROOT / "data" / "handoff" / "handoff_tables.jsonl",
    "CHUNKS_JSONL": PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl",
    "PARSE_FAILURES_CSV": PROJECT_ROOT / "data" / "parsed" / "parse_failures.csv",
    "TIMINGS_JSONL": PROJECT_ROOT / "logs" / "timings.jsonl",
    "COMPANIES_CSV": PROJECT_ROOT / "data" / "enrich" / "companies.csv",
    "DOC_COMPANIES_JSONL": PROJECT_ROOT / "data" / "enrich" / "doc_companies.jsonl",
    # 모델
    "EMBED_MODEL": "BAAI/bge-m3",
    "EMBED_DIM": 1024,
    "EMBED_BATCH": 32,
    "VLM_MODEL": os.environ.get("QWEN_VL_MODEL", "qwen3-vl:8b"),
    "LLM_MODEL": os.environ.get("QWEN_LLM_MODEL", "qwen3:8b"),
    "OLLAMA_URL": os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    # VLM 호출 튜닝: 넓은 표/차트는 비전 토큰+긴 OCR로 기본 컨텍스트(4096)를 넘겨
    # 빈 응답이 나므로 컨텍스트를 크게, 이미지 장변은 제한해 토큰 수를 줄인다.
    # qwen3-vl은 thinking에 수천 토큰을 써서 num_predict를 낮게 잡으면 content가 비므로
    # think=False로 추론을 억제하고 num_predict는 제한하지 않는다.
    "VLM_NUM_CTX": int(os.environ.get("QWEN_VL_NUM_CTX", "16384")),
    "VLM_MAX_EDGE": int(os.environ.get("QWEN_VL_MAX_EDGE", "1400")),
    "VLM_THINK": os.environ.get("QWEN_VL_THINK", "false").lower() == "true",
    "MINERU_EXE": _mineru_exe_str,
    # Supabase
    "STORAGE_BUCKET": "report-images",
    "UPSERT_BATCH": 500,
    # 이미지 1차 규칙 필터 (고도화: 면적 하한 추가, chart는 크기 무관 통과)
    "MIN_IMG_PX": 100,
    "MAX_ASPECT": 8.0,
    "MIN_AREA_PX": 15000,
    # === 고도화 파라미터 ===
    "PROMPT_VER": os.environ.get("PROMPT_VER", "v2"),   # 캐시 키 구성요소 (올리면 전량 미스)
    "CONF_THRESHOLD": 0.6,                              # 이 미만이면 review_queue=true
    "PHASH_DUP_MAX": 0,                                 # 해밍거리 ≤0 → 완전중복 → 판정 복사(dedup_of)
    "PHASH_SIMILAR_MAX": 6,                             # 1~6 → '유사' 표시만, VLM 재실행(시계열 오복사 방지)
    # 청킹
    "CHUNK_MIN": 500,
    "CHUNK_MAX": 800,
    "CHUNK_OVERLAP": 100,
    # 기타
    "MOJIBAKE_THRESHOLD": 0.05,
    "CATEGORIES": ["market_info", "invest", "company", "industry", "economy", "debenture"],
}


def ensure_dirs() -> None:
    for k in ("PARSED_DIR", "USEFUL_DIR", "DISCARDED_DIR", "CACHE_DIR", "HANDOFF_DIR",
              "CHUNKS_DIR", "TABLES_DIR", "ENRICH_DIR", "BM25_DIR", "DENSE_DIR",
              "LOGS_DIR", "EVAL_DIR"):
        CONFIG[k].mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- 로깅/타이밍

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    CONFIG["LOGS_DIR"].mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(CONFIG["LOGS_DIR"] / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_progress(logger: logging.Logger, i: int, n: int, msg: str = "") -> None:
    logger.info(f"[{i}/{n}] {msg}")


def record_timing(script: str, item_id: str, seconds: float) -> None:
    """M10(처리 성능) 측정용 — logs/timings.jsonl에 1건 append."""
    CONFIG["LOGS_DIR"].mkdir(parents=True, exist_ok=True)
    with open(CONFIG["TIMINGS_JSONL"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"script": script, "item_id": item_id,
                            "seconds": round(seconds, 3), "ts": now_iso()},
                           ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- JSONL 유틸

def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def append_jsonl(path: Path | str, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def jsonl_index(path: Path | str, key: str) -> dict[str, dict]:
    """JSONL을 key 기준 dict로 (중복 시 마지막 레코드 우선 → append 갱신 패턴 지원)."""
    return {r[key]: r for r in load_jsonl(path) if key in r}


# ---------------------------------------------------------------- 고도화: 해시·캐시·pHash

import hashlib


def content_hash(data: bytes | Path | str) -> str:
    """이미지 바이트의 sha256 (VLM 캐시 L2 키·중복 판정 기준). 파일 경로도 허용."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    return hashlib.sha256(data).hexdigest()


def cache_key(chash: str, prompt_ver: str, model: str) -> str:
    """캐시 키 = 내용해시 + 프롬프트버전 + 모델 (파일명·시간 기준 금지)."""
    return hashlib.sha256(f"{chash}|{prompt_ver}|{model}".encode()).hexdigest()


def _cache_path(stage: str, key: str) -> Path:
    return CONFIG["CACHE_DIR"] / stage / key[:2] / f"{key}.json"


def cache_get(stage: str, key: str) -> dict | None:
    """L1 로컬 캐시 조회 → dict 또는 None. data/cache/{stage}/{key[:2]}/{key}.json."""
    p = _cache_path(stage, key)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_put(stage: str, key: str, value: dict) -> None:
    """L1 로컬 캐시 저장 (원자적 쓰기)."""
    p = _cache_path(stage, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    tmp.replace(p)


def dhash(path: Path | str, hash_size: int = 8) -> str | None:
    """차이 해시(dHash) 64bit → 16자리 hex. 근접중복(near-dup) 탐지용.

    imagehash 의존성 없이 PIL만으로 구현: 그레이스케일 (size+1)×size 축소 후
    가로 인접 픽셀 대소 비교 → hash_size² 비트. 리사이즈·JPEG압축·미세변화에 강건."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            px = list(im.getdata())
    except Exception:
        return None
    w = hash_size + 1
    bits = 0
    for r in range(hash_size):
        for c in range(hash_size):
            bits = (bits << 1) | (1 if px[r * w + c] < px[r * w + c + 1] else 0)
    return f"{bits:0{hash_size * hash_size // 4}x}"


def hamming(h1: str | None, h2: str | None) -> int:
    """두 hex 해시의 해밍거리 (비트 차이 수). 하나라도 None이면 큰 값(999)."""
    if not h1 or not h2:
        return 999
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


# ---------------------------------------------------------------- 메타데이터

_DOC_ID_RE = re.compile(r"^([a-z_]+_\d+)_")


def doc_id_from_filename(name: str) -> str | None:
    """'company_01_260721_컴투스...' → 'company_01'."""
    m = _DOC_ID_RE.match(Path(name).name)
    return m.group(1) if m else None


def parse_report_date(s: str | None) -> str | None:
    """'26.07.21' → '2026-07-21' (이미 ISO면 그대로)."""
    if not s:
        return None
    s = s.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{2})$", s)
    if m:
        return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def read_metadata() -> list[dict]:
    """metadata.csv → [{...원본컬럼, doc_id, pdf_abs, report_date_iso}]. 파일 없으면 []."""
    path = CONFIG["METADATA_CSV"]
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lp = (row.get("local_path") or "").strip()
            if not lp:
                continue
            row["doc_id"] = doc_id_from_filename(lp) or Path(lp).stem
            row["pdf_abs"] = str(CONFIG["RAW_DIR"] / lp)
            row["report_date_iso"] = parse_report_date(row.get("report_date"))
            rows.append(row)
    return rows


def find_parsed_docs(root: Path | str | None = None) -> list[tuple[str, Path]]:
    """파싱 결과 스캔 → [(doc_id, content_list.json이 있는 디렉터리)].

    두 레이아웃 지원:
      - data/parsed/{doc_id}/*_content_list.json          (s1 산출, 평탄화)
      - <root>/<stem>/auto/*_content_list.json            (MinerU 원본/데모 산출)
    """
    root = Path(root) if root else CONFIG["PARSED_DIR"]
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        cands = list(d.glob("*content_list*.json")) + list(d.glob("*/*content_list*.json"))
        cands = [c for c in cands if "_v2" not in c.name] or cands
        if not cands:
            continue
        doc_id = doc_id_from_filename(d.name) or d.name
        out.append((doc_id, cands[0].parent))
    return out


def load_content_list(mdir: Path | str) -> list[dict]:
    """MinerU content_list.json 로드 (v2 아닌 쪽 우선)."""
    mdir = Path(mdir)
    cands = sorted(mdir.glob("*content_list*.json"))
    cands = [c for c in cands if "_v2" not in c.name] or cands
    if not cands:
        return []
    with open(cands[0], encoding="utf-8") as f:
        return json.load(f)


def load_middle_json(mdir: Path | str) -> dict | None:
    mdir = Path(mdir)
    cands = sorted(mdir.glob("*middle.json"))
    if not cands:
        return None
    with open(cands[0], encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 텍스트 품질

def mojibake_ratio(text: str) -> float:
    """EUC-KR 오디코딩 흔적(U+FFFD) 비율. 0.05 이상이면 손상으로 간주."""
    if not text:
        return 0.0
    return text.count("�") / len(text)


def clean_text(text: str) -> str:
    text = text.replace("�", "")
    return re.sub(r"[ \t]+", " ", text).strip()


# ---------------------------------------------------------------- Supabase

_sb = None
_sb_warned = False


def sb():
    """supabase 클라이언트 싱글턴. .env 미설정이면 None (경고 1회)."""
    global _sb, _sb_warned
    if _sb is not None:
        return _sb
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        if not _sb_warned:
            print("[common] SUPABASE_URL/SUPABASE_SERVICE_KEY 미설정 — DB 쓰기 생략, 로컬 파일만 사용")
            _sb_warned = True
        return None
    from supabase import create_client
    _sb = create_client(url, key)
    return _sb


def upsert(table: str, rows: list[dict], on_conflict: str | None = None) -> int:
    """500행 단위 일괄 upsert + 재시도 3회. Supabase 미설정 시 0 반환."""
    client = sb()
    if client is None or not rows:
        return 0
    total = 0
    batch_size = CONFIG["UPSERT_BATCH"]
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        for attempt in range(3):
            try:
                q = client.table(table).upsert(batch, on_conflict=on_conflict) \
                    if on_conflict else client.table(table).upsert(batch)
                q.execute()
                total += len(batch)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"upsert({table}) 3회 실패: {e}") from e
                time.sleep(2 ** attempt)
    return total


def done_ids(table: str, key: str, eq: dict | None = None) -> set[str]:
    """완료 목록 조회 (1000행 페이징) — 모든 스크립트의 resume 기반."""
    client = sb()
    if client is None:
        return set()
    out: set[str] = set()
    start, page = 0, 1000
    while True:
        q = client.table(table).select(key)
        if eq:
            for col, val in eq.items():
                q = q.eq(col, val)
        res = q.range(start, start + page - 1).execute()
        rows = res.data or []
        out.update(r[key] for r in rows if r.get(key) is not None)
        if len(rows) < page:
            break
        start += page
    return out


def ensure_bucket() -> bool:
    client = sb()
    if client is None:
        return False
    name = CONFIG["STORAGE_BUCKET"]
    try:
        client.storage.get_bucket(name)
        return True
    except Exception:
        pass
    try:
        client.storage.create_bucket(name, options={"public": True})
        return True
    except Exception as e:
        print(f"[common] Storage 버킷 생성 실패({name}): {e}")
        return False


def upload_image(local_path: Path | str, storage_path: str) -> str | None:
    """이미지를 Storage 버킷에 업로드 (덮어쓰기). 성공 시 storage_path 반환."""
    client = sb()
    if client is None:
        return None
    data = Path(local_path).read_bytes()
    bucket = client.storage.from_(CONFIG["STORAGE_BUCKET"])
    try:
        bucket.upload(storage_path, data,
                      {"content-type": "image/jpeg", "upsert": "true"})
        return storage_path
    except Exception:
        try:
            bucket.update(storage_path, data, {"content-type": "image/jpeg"})
            return storage_path
        except Exception as e:
            print(f"[common] Storage 업로드 실패({storage_path}): {e}")
            return None


def vec_to_pg(vec) -> str:
    """임베딩 벡터 → pgvector 문자열 '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


# ---------------------------------------------------------------- Ollama

def ollama_alive() -> bool:
    import requests
    try:
        r = requests.get(f"{CONFIG['OLLAMA_URL']}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ollama_models() -> list[str]:
    import requests
    try:
        r = requests.get(f"{CONFIG['OLLAMA_URL']}/api/tags", timeout=5)
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def has_model(model: str) -> bool:
    base = model.split(":")[0]
    return any(base in m for m in ollama_models())


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def encode_image(path: Path | str, max_edge: int | None = None) -> str:
    """이미지 파일 → base64. max_edge 지정 시 장변을 그 이하로 축소(JPEG 재인코딩).

    큰 표/차트는 비전 토큰이 많아 Ollama 컨텍스트를 넘겨 빈 응답을 유발하므로
    장변을 제한해 토큰 수를 줄인다 (작은 이미지는 원본 유지)."""
    import base64
    if not max_edge:
        return base64.b64encode(Path(path).read_bytes()).decode()
    try:
        import io
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return base64.b64encode(Path(path).read_bytes()).decode()


def ollama_chat(model: str, prompt: str, images: list[str] | None = None,
                expect_json: bool = True, timeout: int = 600,
                retries: int = 2, temperature: float = 0.1,
                system: str | None = None,
                num_ctx: int | None = None, num_predict: int | None = None,
                img_max_edge: int | None = None, think: bool | None = None):
    """Ollama /api/chat 호출.

    images: 이미지 '파일 경로' 리스트 (내부에서 base64 인코딩; img_max_edge로 축소 가능).
    num_ctx: 컨텍스트 창 토큰 수 (VLM에 큰 이미지+긴 OCR을 넣을 땐 크게 잡아야 함).
    num_predict: 최대 생성 토큰 수 (None이면 미제한 — thinking 모델은 제한 시 추론에
        예산을 다 써 content가 비므로 주의).
    think: qwen3 계열 thinking 토글. False로 주면 추론을 억제해 JSON 응답을 안정화한다.
    expect_json=True → 응답에서 JSON 객체를 추출해 dict 반환.
        파싱 실패 시 {"_parse_error": True, "_raw": 원문} 반환.
    expect_json=False → 응답 문자열 그대로 반환.
    호출 자체가 계속 실패하면 None 반환.
    """
    import requests
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = [encode_image(p, img_max_edge) for p in images]
    msgs.append(msg)
    options = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = num_ctx
    if num_predict:
        options["num_predict"] = num_predict
    payload = {"model": model, "messages": msgs, "stream": False, "options": options}
    if think is not None:
        payload["think"] = think
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{CONFIG['OLLAMA_URL']}/api/chat", json=payload,
                              timeout=timeout)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            if not expect_json:
                return content
            m = _JSON_RE.search(content)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            return {"_parse_error": True, "_raw": content}
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    print(f"[common] ollama_chat 실패({model}): {last_err}")
    return None


# ---------------------------------------------------------------- 임베딩(BGE-M3)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        print(f"[common] BGE-M3 로딩 (device={device}) …")
        _embedder = SentenceTransformer(CONFIG["EMBED_MODEL"], device=device)
    return _embedder


def embed_texts(texts: list[str], batch_size: int | None = None):
    """텍스트 리스트 → 정규화된 numpy 배열 (n, 1024). BGE-M3는 쿼리 프리픽스 불필요."""
    model = get_embedder()
    return model.encode(list(texts),
                        batch_size=batch_size or CONFIG["EMBED_BATCH"],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True)


# ---------------------------------------------------------------- 로컬 dense 폴백

def save_dense(name: str, ids: list[str], vectors) -> None:
    """db/dense/{name}.npy + {name}_ids.json 저장 (Supabase 폴백용 로컬 벡터스토어)."""
    import numpy as np
    CONFIG["DENSE_DIR"].mkdir(parents=True, exist_ok=True)
    np.save(CONFIG["DENSE_DIR"] / f"{name}.npy", np.asarray(vectors, dtype="float32"))
    with open(CONFIG["DENSE_DIR"] / f"{name}_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def load_dense(name: str):
    """→ (ids, vectors) 또는 None."""
    import numpy as np
    npy = CONFIG["DENSE_DIR"] / f"{name}.npy"
    idsf = CONFIG["DENSE_DIR"] / f"{name}_ids.json"
    if not npy.exists() or not idsf.exists():
        return None
    with open(idsf, encoding="utf-8") as f:
        ids = json.load(f)
    return ids, np.load(npy)


def local_dense_search(name: str, qvec, topk: int = 20) -> list[tuple[str, float]]:
    """정규화 벡터 내적(=코사인) 기반 로컬 검색 → [(id, score)]."""
    import numpy as np
    loaded = load_dense(name)
    if loaded is None:
        return []
    ids, mat = loaded
    scores = mat @ np.asarray(qvec, dtype="float32")
    order = np.argsort(-scores)[:topk]
    return [(ids[i], float(scores[i])) for i in order]


# ---------------------------------------------------------------- Kiwi/BM25

_kiwi = None


def get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def tokenize_ko(text: str) -> list[str]:
    """한국어 형태소 토큰화 — 명사(N*)·외래어(SL)·숫자(SN)·한자(SH)만 취함."""
    kw = get_kiwi()
    toks = []
    for t in kw.tokenize(text):
        if t.tag.startswith("N") or t.tag in ("SL", "SH", "SN"):
            toks.append(t.form.lower())
    return toks


# ---------------------------------------------------------------- RRF

def rrf(rank_lists: list[list[str]], k: int = 60, topk: int | None = None
        ) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. rank_lists = [[id, ...](순위순), ...] → [(id, score)]."""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, _id in enumerate(lst):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:topk] if topk else ranked


# ---------------------------------------------------------------- 셀프 체크

def _selfcheck() -> None:
    ensure_dirs()
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"metadata.csv : {'OK ('+str(len(read_metadata()))+'행)' if CONFIG['METADATA_CSV'].exists() else '없음'}")
    print(f"MinerU exe   : {CONFIG['MINERU_EXE']}")
    print(f"Supabase     : {'연결설정 OK' if sb() is not None else '미설정 (.env 필요)'}")
    alive = ollama_alive()
    print(f"Ollama       : {'실행 중 — 모델: ' + ', '.join(ollama_models()) if alive else '미실행/미설치'}")
    docs = find_parsed_docs()
    print(f"파싱 결과    : {len(docs)}건 (data/parsed)")
    print(f"이미지 카드  : {len(load_jsonl(CONFIG['IMAGE_CARDS_JSONL']))}건")
    print(f"텍스트 청크  : {len(load_jsonl(CONFIG['CHUNKS_JSONL']))}건")


if __name__ == "__main__":
    _selfcheck()
