"""[9] 임베딩 모델 채택 — BGE-M3 vs dragonkue/BGE-m3-ko vs OpenAI text-embedding-3-small 3종
비교(`evaluate_embeddings.py`) 결과 **dragonkue/BGE-m3-ko 채택**. recall@1은 3종 전부 동률(87%)
이었지만 BGE-m3-ko만 유일하게 recall@3 100%(정답이 항상 top-3 안에 있음) + 가장 높은 MRR(0.922)
+ 가장 빠른 코퍼스 임베딩 속도(0.738s/21청크) — 세 지표 전부에서 우세하거나 동률이라 명확한 채택.
"""

import threading

from sentence_transformers import SentenceTransformer

MODEL_NAME = "dragonkue/BGE-m3-ko"
_model = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """싱글턴 — 한 번만 로드해 재사용(반복 로드 방지).

    [39] 콜드로드(~6.8s)를 다른 초기화 단계와 백그라운드 스레드로 병렬화하는 호출부가 생기면서
    락 추가 — 이전엔 단일 스레드에서만 호출돼 경합이 없었지만, 두 스레드가 동시에 `_model is
    None`을 보면 SentenceTransformer를 두 번 생성(느리고 낭비)하거나 내부 캐시 디렉토리 동시
    쓰기로 손상될 수 있어 방지. 반환하는 모델 객체/동작은 기존과 동일, 동시 호출 시 안전성만
    추가."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list, normalize: bool = True):
    """반환: L2 정규화된 임베딩 배열(코사인 유사도 = 내적으로 바로 계산 가능)."""
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=normalize)
