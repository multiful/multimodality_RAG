import requests

from .base import BaseGenerator
from .prompt import build_messages


class QwenGenerator(BaseGenerator):
    """Zero-shot RAG baseline via Qwen2.5-7B-Instruct (no fine-tuning) — plays the paper's
    "open-source baseline" role. Runs locally through Ollama (CPU-friendly quantized inference).

    Requires Ollama running locally with the model pulled:
        ollama pull qwen2.5:7b
    """

    name = "qwen2.5-7b-instruct"

    def __init__(self, model_name: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate(self, query: str, context: str) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_name,
                "messages": build_messages(query, context),
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
