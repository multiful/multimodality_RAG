# -*- coding: utf-8 -*-
"""4축 엔티티+메타데이터 추출: 각 축 full_text를 동일 ~1800자 윈도우로 잘라
파이프라인의 실제 구조화출력 extract_text_chunk_metadata(gpt-4o-mini) 배치 투입.
엔티티/감성/섹터/지표/시점 union + 지연 저장. 4축 공정성 위해 윈도우 크기 통일."""
import sys, os, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG")
PP = ROOT/"pdf_pipeline"
for d in [str(HERE), str(PP), str(PP/"table_processing")]:
    sys.path.insert(0, d)
import common_exp as C

# --- .env 로드 (OPENAI_API_KEY) ---
for line in (ROOT/".env").read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip())
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY 없음"

from structured_output import extract_text_chunk_metadata

WIN = 1800
BATCH = 8
DOC_TITLE = "스마트폰 수요 우려는 예견된 수순 (반도체 및 소부장 Weekly)"

def windows(text):
    text = text or ""
    return [text[i:i+WIN] for i in range(0, len(text), WIN) if text[i:i+WIN].strip()]

def run_axis(axis):
    o = C.load_json(HERE/f"out_{axis}.json")
    wins = windows(o["full_text"])
    chunks = [{"raw_chunk": w, "section_path": []} for w in wins]
    entities, sentiments, sectors, metrics, periods, topics = [], [], [], [], [], []
    n_calls = 0
    t0 = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        res = extract_text_chunk_metadata(batch, doc_title=DOC_TITLE, sector="반도체")
        n_calls += 1
        for r in res:
            if not r: continue
            entities += r.get("entities") or []
            metrics += r.get("metric_mentions") or []
            if r.get("sentiment"): sentiments.append(r["sentiment"])
            if r.get("sector_mentioned"): sectors.append(r["sector_mentioned"])
            if r.get("time_period"): periods.append(r["time_period"])
            if r.get("topic"): topics.append(r["topic"])
    elapsed = time.time() - t0
    # dedup 유지 순서
    def uniq(xs):
        seen=set(); out=[]
        for x in xs:
            k=str(x).strip().lower()
            if k and k not in seen: seen.add(k); out.append(str(x).strip())
        return out
    out = {
        "axis": axis,
        "n_windows": len(wins), "n_api_calls": n_calls,
        "extract_latency_s": round(elapsed, 2),
        "entities_raw_count": len(entities),
        "entities": uniq(entities),
        "metric_mentions": uniq(metrics),
        "sentiments": sentiments,
        "sectors": uniq(sectors),
        "time_periods": uniq(periods),
    }
    C.dump_json(HERE/f"entity_out_{axis}.json", out)
    from collections import Counter
    sc = Counter(sentiments)
    print(f"[{axis}] windows={len(wins)} calls={n_calls} {elapsed:.1f}s "
          f"entities={len(out['entities'])} (raw {len(entities)}) sentiments={dict(sc)} sectors={out['sectors'][:3]}")
    return out

if __name__ == "__main__":
    axes = sys.argv[1:] or ["baseline", "enhanced", "docling", "mineru"]
    for a in axes:
        run_axis(a)
