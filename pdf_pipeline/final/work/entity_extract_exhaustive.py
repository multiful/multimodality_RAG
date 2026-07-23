# -*- coding: utf-8 -*-
"""엔티티 추출 강화 패스: 파이프라인 구조화출력의 side-field entities는 표에서 장꼬리 종목을
덜 뽑는다(실측). 팀 extract_entities_and_eval의 '빠짐없이 나열' 전용 프롬프트를 같은 모델
(gpt-4o-mini)로 uniform 윈도우에 적용해 exhaustive 엔티티를 얻고, 기존 side-field 결과와 union.
4축 전부 동일 처리(공정)."""
import sys, os, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(r"c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG")
sys.path.insert(0, str(HERE))
import common_exp as C
for line in (ROOT/".env").read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip())
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

WIN=1800; BATCH=4; MODEL="gpt-4o-mini"
PROMPT=("다음은 증권사 반도체 리포트에서 추출한 텍스트 조각들입니다. 이 안에 등장하는 "
        "모든 기업/기관 이름을 빠짐없이 나열하세요. 표의 행이나 차트 범례 안에서만 나온 기업도 "
        "반드시 포함하세요(예: 밸류에이션 표의 모든 종목). 한 줄에 기업명 하나씩만 출력하고 다른 설명·번호·수치는 쓰지 마세요.\n\n{block}")

def windows(t):
    t=t or ""; return [t[i:i+WIN] for i in range(0,len(t),WIN) if t[i:i+WIN].strip()]

def run(axis):
    o=C.load_json(HERE/f"out_{axis}.json")
    wins=windows(o["full_text"])
    ents=[]; calls=0; t0=time.time()
    for i in range(0,len(wins),BATCH):
        block="\n\n---\n\n".join(wins[i:i+BATCH])
        r=client.chat.completions.create(model=MODEL, temperature=0,
            messages=[{"role":"user","content":PROMPT.format(block=block)}])
        calls+=1
        for ln in (r.choices[0].message.content or "").splitlines():
            ln=ln.strip(" -*·0123456789.\t")
            if ln and len(ln)<=40: ents.append(ln)
    elapsed=time.time()-t0
    # 기존 side-field 결과와 union + 메타데이터 보존
    ex=C.load_json(HERE/f"entity_out_{axis}.json")
    def uniq(xs):
        seen=set(); out=[]
        for x in xs:
            k=str(x).strip().lower()
            if k and k not in seen: seen.add(k); out.append(str(x).strip())
        return out
    merged=uniq((ex.get("entities") or []) + ents)
    ex["entities_sidefield_only"]=ex.get("entities")
    ex["entities_exhaustive_only"]=uniq(ents)
    ex["entities"]=merged
    ex["exhaustive_latency_s"]=round(elapsed,2)
    ex["exhaustive_calls"]=calls
    C.dump_json(HERE/f"entity_out_{axis}.json", ex)
    print(f"[{axis}] exhaustive {elapsed:.1f}s calls={calls} exhaustive_ents={len(uniq(ents))} -> merged={len(merged)}")

if __name__=="__main__":
    for a in (sys.argv[1:] or ["baseline","enhanced","enhanced_v2","docling","mineru"]):
        run(a)
