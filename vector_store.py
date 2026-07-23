import chromadb

from embeddings.base import BaseEmbedder


class VectorStore:
    """Chroma collection bound to one embedder. Swap embedders to get a separate, comparable index."""

    def __init__(self, embedder: BaseEmbedder, persist_dir: str = "./chroma_db"):
        self.embedder = embedder
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(name=embedder.name)

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict] | None = None):
        vectors = self.embedder.embed(texts)
        self.collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    def query(self, text: str, top_k: int = 5):
        return self.query_batch([text], top_k=top_k)

    def query_batch(self, texts: list[str], top_k: int = 5):
        """Embed all texts in one call, then issue one Chroma query for all of them at once."""
        vectors = self.embedder.embed(texts)
        return self.collection.query(query_embeddings=vectors, n_results=top_k)
