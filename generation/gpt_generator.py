import os

from openai import OpenAI

from .base import BaseGenerator
from .prompt import build_messages


class GPTGenerator(BaseGenerator):
    """Zero-shot RAG baseline via OpenAI GPT-4o-mini — plays the paper's "commercial baseline" role."""

    name = "gpt-4o-mini"

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def generate(self, query: str, context: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=build_messages(query, context),
            temperature=0.2,
        )
        return resp.choices[0].message.content
