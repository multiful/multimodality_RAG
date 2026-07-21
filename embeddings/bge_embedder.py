from FlagEmbedding import BGEM3FlagModel

from .base import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    """Local, free embedding via BAAI/bge-m3. Multilingual + cross-lingual, up to 8192 tokens."""

    name = "bge-m3"

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(texts, batch_size=12, max_length=8192)
        return output["dense_vecs"].tolist()
