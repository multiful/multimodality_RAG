from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Common interface so generation backends are interchangeable behind the same prompt template."""

    name: str

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError
