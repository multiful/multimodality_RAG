"""공통 유틸: 청킹, 텍스트 정규화, 근접성(association) 채점, BGE+BM25 하이브리드 검색.

4개 축(baseline/enhanced/docling/mineru) 공통으로 쓰는 도구.
- 검색 백본은 4축 전부 동일(BGE-m3-ko dense 0.7 + BM25 0.3, min-max 가중합) — 파싱 품질만 분리 비교.
- 정답 매칭은 '근접성(proximity)' 기반: 라벨과 값이 같은 청크 안에서 window 이내로 붙어 있어야 hit.
  (표 셀 정렬이 무너져 값이 엉뚱한 위치로 흩어지면 co-occur 해도 fail → 셀 연관성 품질을 잡아냄)
"""
import re, json, unicodedata
from pathlib import Path

# ---------------- 정규화 ----------------
def norm_text(s: str) -> str:
    """근접성 채점용 경량 정규화. 위치를 크게 흩뜨리지 않으면서:
    - 유니코드 NFKC
    - 숫자 사이 천단위 콤마 제거 (4,659,226 -> 4659226)
    - 숫자와 % 사이 공백 제거 (60 % -> 60%)
    - 연속 공백 1개로
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"(?<=\d)\s+%", "%", s)
    s = re.sub(r"\s+", " ", s)
    return s

def norm_key(k: str) -> str:
    return norm_text(k)

def _find_all(hay: str, needle: str):
    out, i = [], hay.find(needle)
    while i != -1:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out

def keys_hit(text: str, keys, window: int = 220):
    """text 안에서 모든 key가 등장하고(=coverage), 서로 window(정규화 문자수) 이내로
    붙어 있으면(=association) proximity hit. 반환 (all_present:bool, prox_hit:bool, min_span:int|None)."""
    T = norm_text(text)
    positions = []
    for k in keys:
        nk = norm_key(k)
        occ = _find_all(T, nk)
        if not occ:
            return (False, False, None)
        positions.append([(p, p + len(nk)) for p in occ])
    if len(positions) == 1:
        return (True, True, 0)
    # 2개 이상: 각 key의 등장 위치 조합 중 최소 span(가장 왼쪽 시작~가장 오른쪽 끝) 찾기(근사: 그리디)
    best = None
    # 첫 key의 각 위치를 기준으로 나머지 key의 가장 가까운 위치를 골라 span 계산
    for anchor in positions[0]:
        lo, hi = anchor
        ok = True
        for plist in positions[1:]:
            # anchor 중심에 가장 가까운 구간 선택
            nearest = min(plist, key=lambda pr: 0 if (pr[1] >= lo and pr[0] <= hi) else min(abs(pr[0]-hi), abs(lo-pr[1])))
            lo = min(lo, nearest[0]); hi = max(hi, nearest[1])
        span = hi - lo
        if best is None or span < best:
            best = span
    prox = best is not None and best <= window
    return (True, prox, best)

# ---------------- 마크다운/텍스트 청킹 ----------------
def _split_table_rows(rows, header_ctx, max_chars=1500):
    """표 row 리스트 -> 행 경계 유지하며 max_chars 이하 청크로 분할, 각 청크에 header_ctx 접두."""
    out, cur = [], []
    cur_len = len(header_ctx)
    for r in rows:
        if cur and cur_len + len(r) > max_chars:
            out.append(header_ctx + "\n".join(cur)); cur = []; cur_len = len(header_ctx)
        cur.append(r); cur_len += len(r) + 1
    if cur:
        out.append(header_ctx + "\n".join(cur))
    return out

def chunk_markdown(md: str, max_chars: int = 1400):
    """docling/mineru 마크다운 -> 블록 청크. 표는 행 경계를 절대 안 쪼갬(라벨-값 co-occur 보존).
    - 파이프표(| ... |) / HTML표(<table>...</table>) 둘 다 처리.
    - 표 바로 앞의 캡션(표 N./도표 N. 등)과 현재 섹션 헤더를 표 청크에 접두(컬럼 컨텍스트 동반).
    - 큰 표는 행 단위로 나누되 헤더행을 각 조각에 반복 접두.
    반환: [{"text":..., "kind":"table|text|heading"}]
    """
    lines = md.splitlines()
    chunks = []
    cur_header = ""
    last_text_line = ""   # 표 캡션 후보
    i = 0
    buf = []
    def flush_text():
        nonlocal buf
        if buf:
            t = "\n".join(buf).strip()
            if t:
                body = (cur_header + "\n" + t).strip() if cur_header else t
                for j in range(0, len(body), max_chars):
                    chunks.append({"text": body[j:j+max_chars], "kind": "text"})
            buf = []
    def caption():
        parts = []
        if cur_header: parts.append(cur_header)
        if last_text_line: parts.append(last_text_line)
        return (" / ".join(parts) + "\n") if parts else ""
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        is_pipe = stripped.startswith("|") and ln.count("|") >= 2
        is_html_tbl = "<table" in ln.lower()
        if is_pipe:
            flush_text()
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].count("|") >= 2:
                tbl.append(lines[i]); i += 1
            head_ctx = caption()
            # 헤더행(첫 1~2행)을 각 조각에 반복
            hdr_rows = "\n".join(tbl[:2]) + "\n" if len(tbl) > 3 else ""
            for piece in _split_table_rows(tbl, head_ctx + hdr_rows, max_chars):
                chunks.append({"text": piece, "kind": "table"})
            last_text_line = ""
            continue
        if is_html_tbl:
            flush_text()
            # <table> ... </table> 수집(여러 줄 가능)
            block = [ln]
            while "</table>" not in " ".join(block).lower() and i+1 < len(lines):
                i += 1; block.append(lines[i])
            i += 1
            html = " ".join(block)
            head_ctx = caption()
            # <tr> 단위로 분해
            import re as _re
            rows = _re.findall(r"<tr.*?</tr>", html, flags=_re.S|_re.I)
            if not rows:
                rows = [html]
            hdr_rows = ("".join(rows[:2])) if len(rows) > 3 else ""
            for piece in _split_table_rows(rows, head_ctx + hdr_rows, max_chars):
                chunks.append({"text": piece, "kind": "table"})
            last_text_line = ""
            continue
        if stripped.startswith("#"):
            flush_text()
            cur_header = stripped.lstrip("#").strip()
            chunks.append({"text": cur_header, "kind": "heading"})
        elif stripped == "":
            flush_text()
        else:
            buf.append(ln)
            last_text_line = stripped
        i += 1
    flush_text()
    return [c for c in chunks if c["text"].strip()]

def chunk_pages_raw(page_texts, max_chars: int = 1500):
    """baseline: 페이지 원문을 페이지 단위 청크(구조 인식 없음). 너무 길면 분할."""
    chunks = []
    for pi, t in enumerate(page_texts, start=1):
        t = (t or "").strip()
        if not t:
            continue
        for j in range(0, len(t), max_chars):
            chunks.append({"text": t[j:j+max_chars], "kind": "page", "page": pi})
    return chunks

# ---------------- 구조 지표: 캡션/도표 보존 ----------------
CHART_RE = re.compile(r"도표\s*\d+")
TABLE_CAP_RE = re.compile(r"(?<!도)표\s*\d+")  # '도표' 제외한 '표 N'

def count_captions(full_text: str):
    T = norm_text(full_text)
    charts = set(re.findall(r"도표\s*\d+", T))
    # '표 N' 중 '도표 N' 아닌 것
    tabs = set(m for m in re.findall(r"표\s*\d+", T))
    charts_norm = set(re.findall(r"\d+", " ".join(charts)))
    # 표 캡션: '표 N' 문자열이 '도표 N'의 부분일 수 있으니 별도 정규식
    tabcaps = set(re.findall(r"(?:^|[^도])표\s*(\d+)", T))
    chartnums = set(re.findall(r"도표\s*(\d+)", T))
    return {"chart_titles": sorted(int(x) for x in chartnums),
            "table_caps": sorted(int(x) for x in tabcaps)}

# ---------------- 검색 백본 (BGE-m3-ko + BM25, 4축 공통) ----------------
_BM_TOK = re.compile(r"[가-힣]+|[A-Za-z0-9]+")
def _tok(t): return _BM_TOK.findall(t.lower())

class HybridIndex:
    def __init__(self, chunks, embed_fn):
        from rank_bm25 import BM25Okapi
        import numpy as np
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.emb = np.asarray(embed_fn(texts)) if texts else np.zeros((0, 1024))
        self.bm25 = BM25Okapi([_tok(t) for t in texts]) if texts else None
        self.embed_fn = embed_fn
    def search(self, query, top_k=5, dw=0.7, bw=0.3):
        import numpy as np
        if not self.chunks:
            return []
        q = np.asarray(self.embed_fn([query])[0])
        dense = self.emb @ q
        bm = np.asarray(self.bm25.get_scores(_tok(query)))
        def mm(a):
            s = a.max() - a.min()
            return (a - a.min())/s if s > 0 else np.zeros_like(a)
        fused = dw*mm(dense) + bw*mm(bm)
        order = np.argsort(-fused)[:top_k]
        return [{"i": int(i), "text": self.chunks[i]["text"], "score": float(fused[i]),
                 "kind": self.chunks[i].get("kind")} for i in order]

def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def dump_json(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
