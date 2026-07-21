from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class BGEKoEmbedder(BaseEmbedder):
    """Local, free embedding via dragonkue/BGE-m3-ko — BGE-M3 fine-tuned for Korean."""

    name = "bge-m3-ko"

    def __init__(self, model_name: str = "dragonkue/BGE-m3-ko"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=12).tolist()
