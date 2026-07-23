"""data/corpus/kospi200_financial.jsonl을 임베딩해 Supabase(financial_chunks)에 적재한다.

사전 작업:
    1. Supabase SQL Editor에서 sql/schema.sql 실행
    2. .env에 SUPABASE_URL / SUPABASE_SERVICE_KEY / OPENAI_API_KEY 설정
    3. python data_collection/parse_kospi200_financials.py 로 corpus 생성

Usage:
    python build_index_supabase.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from embeddings import GPTEmbedder
from supabase_store import SupabaseVectorStore

CORPUS_FILE = Path("data/corpus/kospi200_financial.jsonl")
EMBED_BATCH_SIZE = 20  # OpenAI 임베딩 API 호출당 청크 수


def load_corpus() -> list[dict]:
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"{CORPUS_FILE} 이 없습니다. 먼저 python data_collection/parse_kospi200_financials.py 를 실행하세요."
        )
    with CORPUS_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    load_dotenv()
    records = load_corpus()
    store = SupabaseVectorStore(GPTEmbedder())

    for i in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[i : i + EMBED_BATCH_SIZE]
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        metadatas = [
            {"ticker": r["ticker"], "statement_type": r["statement_type"], "period_type": r["period_type"]}
            for r in batch
        ]
        store.add(ids=ids, texts=texts, metadatas=metadatas)
        print(f"{min(i + EMBED_BATCH_SIZE, len(records))}/{len(records)} 적재 완료")

    print(f"완료: {len(records)}개 청크 -> Supabase.financial_chunks")


if __name__ == "__main__":
    main()
