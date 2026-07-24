# -*- coding: utf-8 -*-
"""[재일] RAGAS 생성 품질 평가 — gpt-4o-mini vs gpt-4.1을 '토큰'이 아니라 '성능'으로 비교.

측정(RAGAS 0.2.15, 격리 venv에서 실행 — 파이프라인 환경의 langchain 버전과 충돌해서 분리):
  faithfulness        : 답변의 각 주장이 주어진 컨텍스트로 뒷받침되는가 = **환각 없음**
  answer_relevancy    : 답변이 질문에 실제로 답했는가
  context_precision   : 준 컨텍스트 중 정답에 기여한 것이 상위에 왔는가(검색 품질, 두 모델 공통)
  context_recall      : 정답(reference)의 내용이 컨텍스트로 커버됐는가(검색 품질, 두 모델 공통)
  factual_correctness : 정답 대비 사실 일치도

추가로 RAGAS가 안 재는 '사용자가 설득될 만한 논리력'을 포인트와이즈 루브릭(1~5)으로 별도 채점 —
답변을 하나씩 독립 채점하므로 이전 pairwise judge에서 관측된 위치편향이 원리적으로 생기지 않는다.

주의: db_context(기업 DB 재무요약)도 컨텍스트에 포함해야 한다 — 안 넣으면 DB에서 온 수치가
전부 '근거 없는 주장'으로 잡혀 faithfulness가 부당하게 깎인다."""
import os, json, sys
from pathlib import Path
PP = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/pdf_pipeline")
ENV = Path("c:/Users/wodlf/OneDrive/Desktop/프로젝트2/multimodality_RAG/.env")
for line in open(ENV, encoding="utf-8"):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
IN = PP/"final"/"ragas_input_construct.json"
OUT = PP/"final"/"results_ragas.json"
MODELS = ["gpt-4o-mini", "gpt-4.1"]
DBCTX_CHUNK = 1500   # db_context를 이 길이로 쪼개 컨텍스트 항목으로 추가

from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference,
                            LLMContextRecall, FactualCorrectness)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
raw = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

RUBRIC = """다음은 증권사 리포트 RAG 시스템의 답변이다. 사용자(투자자) 입장에서 이 답변이 얼마나
'설득되는 논리'를 갖췄는지 1~5로 채점하라. 다른 답변과 비교하지 말고 이 답변만 보고 절대평가하라.

5 = 주장마다 구체 수치 근거가 붙고, 근거→해석→시사점의 인과가 명시적이며, 상충 요인/한계까지 짚음
4 = 대부분의 주장에 수치 근거가 있고 논리 흐름이 이어지나 시사점이나 한계 언급이 얕음
3 = 사실 나열은 정확하나 "그래서 무엇인가"의 해석이 약해 설득력이 부족
2 = 근거가 희박하거나 질문과 부분적으로만 연결됨
1 = 근거 없이 일반론만 서술

<질문>
{q}
</질문>
<답변>
{a}
</답변>

숫자 하나만 출력하라."""


def rubric_score(q, a, tries=6):
    """조직 TPM 한도(gpt-4o 30K)에 걸리므로 429는 지수 백오프로 재시도."""
    import re, time as _t
    for i in range(tries):
        try:
            r = raw.chat.completions.create(model="gpt-4o", temperature=0,
                                            messages=[{"role": "user", "content": RUBRIC.format(q=q, a=a[:4000])}])
            m = re.search(r"[1-5]", r.choices[0].message.content)
            return int(m.group()) if m else None
        except Exception as e:
            if "rate_limit" not in str(e).lower() or i == tries - 1:
                print(f"    !! rubric 실패: {str(e)[:90]}"); return None
            _t.sleep(2 ** i)
    return None


def main():
    data = json.loads(IN.read_text(encoding="utf-8"))
    samples = data["samples"]
    res = {"n_samples": len(samples), "models": {}}
    for model in MODELS:
        rows = []
        for s in samples:
            db = s["db_context"]
            ctxs = s["contexts"] + [db[i:i+DBCTX_CHUNK] for i in range(0, len(db), DBCTX_CHUNK)]
            rows.append({"user_input": s["question"], "retrieved_contexts": ctxs,
                          "response": s["answers"][model]["text"], "reference": s["reference"]})
        ds = EvaluationDataset.from_list(rows)
        r = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(),
                                   LLMContextPrecisionWithReference(), LLMContextRecall(),
                                   FactualCorrectness()],
                     llm=judge_llm, embeddings=judge_emb,
                     run_config=RunConfig(max_workers=2, timeout=300, max_retries=10))
        df = r.to_pandas()
        scores = {c: round(float(df[c].mean(skipna=True)), 3)
                  for c in df.columns if df[c].dtype.kind == "f"}
        import time as _t
        _t.sleep(20)   # RAGAS가 방금 TPM을 다 쓴 직후라 루브릭 채점 전에 한도 회복 대기
        rub = [rubric_score(s["question"], s["answers"][model]["text"]) for s in samples]
        rub = [x for x in rub if x]
        scores["logic_rubric_1to5"] = round(sum(rub)/len(rub), 2) if rub else None
        scores["latency_s_mean"] = round(sum(s["answers"][model]["latency_s"] for s in samples)/len(samples), 2)
        scores["out_tok_mean"] = round(sum(s["answers"][model]["out_tok"] for s in samples)/len(samples))
        res["models"][model] = scores
        res.setdefault("per_query", {})[model] = {
            s["id"]: {c: (None if df[c].isna().iloc[i] else round(float(df[c].iloc[i]), 3))
                      for c in df.columns if df[c].dtype.kind == "f"}
            for i, s in enumerate(samples)}
        res["per_query"][model + "_rubric"] = {s["id"]: rub[i] if i < len(rub) else None
                                                for i, s in enumerate(samples)}
        print(f"\n[{model}] {scores}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT.name} 저장")


if __name__ == "__main__":
    main()
