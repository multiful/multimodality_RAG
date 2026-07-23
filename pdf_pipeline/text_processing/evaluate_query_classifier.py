"""[44] 사용자 지적("추상질의/키워드질의 분류를 잘 하는지 모르겠네") 검증 — 규칙 기반
classify_query_type()을 이진 축(keyword_specific/abstract)으로 접어 query_type_labeled_set.json
(Claude가 직접 라벨링, 트리거 단어를 의도적으로 피한 신규 표현 6개 포함)과 비교하고, LLM 기반
classify_query_type_llm()을 gpt-4o-mini/gpt-4o 두 모델로 같은 세트에 돌려 정확도+지연+비용을 비교."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=str(ROOT / ".env"))

from openai import OpenAI
from index_text import classify_query_type, classify_query_type_llm

client = OpenAI()
data = json.loads((Path(__file__).resolve().parent / "query_type_labeled_set.json").read_text(encoding="utf-8"))
items = data["items"]

_RULE_TO_BINARY = {"summary": "abstract", "factoid": "keyword_specific",
                    "list": "keyword_specific", "comparison": "keyword_specific"}


def eval_method(name, predict_fn):
    correct, lats, wrong = 0, [], []
    for item in items:
        t0 = time.perf_counter()
        pred = predict_fn(item["query"])
        lats.append(time.perf_counter() - t0)
        if pred == item["label"]:
            correct += 1
        else:
            wrong.append((item["query"], item["label"], pred))
    acc = correct / len(items)
    print(f"{name:24s} 정확도={acc:.1%} ({correct}/{len(items)})  평균지연={sum(lats)/len(lats)*1000:.0f}ms")
    if wrong:
        print("  오분류:")
        for q, gold, pred in wrong:
            print(f"    [{gold}->{pred}] {q}")
    return acc, sum(lats) / len(lats)


print(f"평가 세트: {len(items)}개 (keyword_specific {sum(1 for i in items if i['label']=='keyword_specific')}개, "
      f"abstract {sum(1 for i in items if i['label']=='abstract')}개)\n")

eval_method("규칙 기반(classify_query_type)", lambda q: _RULE_TO_BINARY[classify_query_type(q)])
print()
eval_method("LLM(gpt-4o-mini)", lambda q: classify_query_type_llm(q, client=client, model="gpt-4o-mini"))
print()
eval_method("LLM(gpt-4o)", lambda q: classify_query_type_llm(q, client=client, model="gpt-4o"))
