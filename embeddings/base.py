from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Common interface so BGE and GPT embeddings are interchangeable in the RAG pipeline."""

    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text, same order as input."""
        raise NotImplementedError
