"""KOSPI200_output/kospi200_profiles/*.md를 임베딩해 Supabase(company_profile_chunks)에 적재한다.

사전 작업:
    1. Supabase SQL Editor(혹은 직접 연결)에서 sql/schema.sql 실행
    2. .env에 SUPABASE_URL / SUPABASE_SERVICE_KEY / OPENAI_API_KEY 설정

Usage:
    python build_index_profiles.py
"""

import re
from pathlib import Path

from dotenv import load_dotenv

from embeddings.gpt_embedder import GPTEmbedder
from supabase_store import SupabaseVectorStore

PROFILE_DIR = Path("KOSPI200_output/kospi200_profiles")
EMBED_BATCH_SIZE = 20  # OpenAI 임베딩 API 호출당 파일 수

FILENAME_RE = re.compile(r"^(.+)_profile\.md$")


def load_profiles() -> list[dict]:
    if not PROFILE_DIR.exists():
        raise FileNotFoundError(f"{PROFILE_DIR} 이 없습니다.")
    records = []
    for path in sorted(PROFILE_DIR.glob("*_profile.md")):
        m = FILENAME_RE.match(path.name)
        ticker = m.group(1)
        records.append({"ticker": ticker, "text": path.read_text(encoding="utf-8")})
    return records


def main():
    load_dotenv()
    records = load_profiles()
    store = SupabaseVectorStore(GPTEmbedder(), table="company_profile_chunks")

    for i in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[i : i + EMBED_BATCH_SIZE]
        ids = [r["ticker"] for r in batch]
        texts = [r["text"] for r in batch]
        metadatas = [{"ticker": r["ticker"]} for r in batch]
        store.add(ids=ids, texts=texts, metadatas=metadatas)
        print(f"{min(i + EMBED_BATCH_SIZE, len(records))}/{len(records)} 적재 완료")

    print(f"완료: {len(records)}개 프로필 -> Supabase.company_profile_chunks")


if __name__ == "__main__":
    main()
