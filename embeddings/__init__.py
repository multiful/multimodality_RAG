from .base import BaseEmbedder
from .gpt_embedder import GPTEmbedder

__all__ = ["BaseEmbedder", "GPTEmbedder"]

# BGE 계열은 FlagEmbedding/sentence-transformers가 설치돼 있을 때만 로드한다.
# (GPTEmbedder만 쓰는 환경에서 무거운 선택적 의존성 설치를 강제하지 않기 위함)
try:
    from .bge_embedder import BGEEmbedder

    __all__.append("BGEEmbedder")
except ImportError:
    pass

try:
    from .bge_ko_embedder import BGEKoEmbedder

    __all__.append("BGEKoEmbedder")
except ImportError:
    pass
