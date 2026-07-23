"""[9] 임베딩 모델 채택 — BGE-M3 vs dragonkue/BGE-m3-ko vs OpenAI text-embedding-3-small 3종
비교(`evaluate_embeddings.py`) 결과 **dragonkue/BGE-m3-ko 채택**. recall@1은 3종 전부 동률(87%)
이었지만 BGE-m3-ko만 유일하게 recall@3 100%(정답이 항상 top-3 안에 있음) + 가장 높은 MRR(0.922)
+ 가장 빠른 코퍼스 임베딩 속도(0.738s/21청크) — 세 지표 전부에서 우세하거나 동률이라 명확한 채택.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "dragonkue/BGE-m3-ko"
_model = None


def get_embedding_model() -> SentenceTransformer:
    """싱글턴 — 한 번만 로드해 재사용(반복 로드 방지)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list, normalize: bool = True):
    """반환: L2 정규화된 임베딩 배열(코사인 유사도 = 내적으로 바로 계산 가능)."""
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=normalize)
