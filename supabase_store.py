import os

from supabase import Client, create_client

from embeddings.base import BaseEmbedder


class SupabaseVectorStore:
    """financial_chunks 테이블(pgvector, supabase/schema.sql 참고)에 청크를 적재/검색한다."""

    def __init__(self, embedder: BaseEmbedder, table: str = "financial_chunks", batch_size: int = 100):
        self.embedder = embedder
        self.table = table
        self.batch_size = batch_size
        self.client: Client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        vectors = self.embedder.embed(texts)
        rows = [
            {"id": id_, "content": text, "embedding": vector, **meta}
            for id_, text, vector, meta in zip(ids, texts, vectors, metadatas)
        ]
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i : i + self.batch_size]
            self.client.table(self.table).upsert(batch).execute()

    def query(self, text: str, top_k: int = 5, ticker: str | None = None):
        vector = self.embedder.embed([text])[0]
        resp = self.client.rpc(
            "match_financial_chunks",
            {"query_embedding": vector, "match_count": top_k, "filter_ticker": ticker},
        ).execute()
        return resp.data
