from __future__ import annotations

import os

from openai import OpenAI

from .base import BaseEmbedder


class GPTEmbedder(BaseEmbedder):
    """Paid, zero-setup embedding via OpenAI text-embedding-3-small."""

    name = "text-embedding-3-small"

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]
