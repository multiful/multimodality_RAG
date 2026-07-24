# -*- coding: utf-8 -*-
"""다문서 엔티티 추출 — 파이프라인 기본 추출기(extract_text_chunk_metadata, sidefield).
구조 차이가 가장 잘 드러나는 약추출기 사용(균일 1800자 윈도우, 4파서 공정)."""
import sys, os, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG"); PP=ROOT/"pdf_pipeline"
for d in [str(HERE),str(PP),str(PP/"table_processing")]: sys.path.insert(0,d)
import common_exp as C
for line in (ROOT/".env").read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
from structured_output import extract_text_chunk_metadata
MD=HERE/"multidoc"; WIN=1800; BATCH=8
DOCS=["doc1_construction","doc2_pharma","doc3_preview","doc4_retail","doc5_steel","doc06_aircraft","doc07_mart","doc08_solar","doc09_bank","doc10_food","doc11_delivery","doc12_tourism","doc13_construction2","doc14_semi","doc15_earnings"]; AXES=["baseline","enhanced_v5","docling","mineru"]

def windows(t): return [t[i:i+WIN] for i in range(0,len(t or ''),WIN) if (t[i:i+WIN]).strip()]

def run(docid,axis):
    p=MD/f"out_{docid}_{axis}.json"
    if not p.exists(): print(f"  skip {docid}/{axis} (no parse)"); return
    if (MD/f"entity_{docid}_{axis}.json").exists(): print(f"  skip {docid}/{axis}"); return
    o=C.load_json(p); wins=windows(o["full_text"])
    chunks=[{"raw_chunk":w,"section_path":[]} for w in wins]
    ents=[]; calls=0; t0=time.time()
    for i in range(0,len(chunks),BATCH):
        res=extract_text_chunk_metadata(chunks[i:i+BATCH],sector=None); calls+=1
        for r in res:
            if r: ents+=r.get("entities") or []
    def uniq(xs):
        seen=set();out=[]
        for x in xs:
            k=str(x).strip().lower()
            if k and k not in seen: seen.add(k);out.append(str(x).strip())
        return out
    C.dump_json(MD/f"entity_{docid}_{axis}.json",{"docid":docid,"axis":axis,"n_windows":len(wins),
        "extract_latency_s":round(time.time()-t0,2),"entities":uniq(ents)})
    print(f"[{docid}/{axis}] {len(wins)}win {round(time.time()-t0,1)}s ents={len(uniq(ents))}",flush=True)

if __name__=="__main__":
    for docid in DOCS:
        for axis in AXES: run(docid,axis)
