"""[11] PDF 섹터(21종) 분류 — 사용자 요청: 입력 PDF가 [음식료품, 섬유의복, ..., 서비스업] 21개
섹터 중 어디에 해당하는지 판별. `sector_schema.yaml`에 이미 있는 섹터별 큐레이션된 용어(필드
alias)를 재사용해 두 방식을 구현:

1. 임베딩 기반(embedding_classify): 섹터별 alias를 모아 "섹터 설명 텍스트"를 만들고, [9]에서
   채택한 BGE-m3-ko로 섹터 설명과 PDF 텍스트를 임베딩해 코사인 유사도가 가장 높은 섹터를 선택.
   LLM 호출 없음, 빠름.
2. LLM 기반(llm_classify): 로컬 Qwen2.5-VL에 PDF 텍스트 + 21개 섹터 목록을 주고 하나를 고르게
   하는 zero-shot 분류. 일반 상식(회사명→업종)까지 활용 가능하나 LLM 호출이 필요해 상대적으로 느림.

`evaluate_sector_classifier.py`에서 두 방식을 golden set(2개 실제 PDF + 10개 합성 케이스)으로
비교해 채택.
"""

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # embedding.py가 pdf_pipeline/에 있음

SCHEMA_PATH = Path(__file__).resolve().parent / "sector_schema.yaml"


def _load_sector_descriptions() -> dict:
    """섹터별로 그 섹터가 쓰는 필드들의 alias를 전부 모아 하나의 설명 텍스트로 조립."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    fields = schema["fields"]
    descriptions = {}
    for sector, sdata in schema["sectors"].items():
        field_keys = set()
        for field_list in sdata["table_types"].values():
            field_keys.update(field_list)
        aliases = []
        for fk in field_keys:
            aliases.extend(fields.get(fk, {}).get("aliases", []))
        descriptions[sector] = f"{sector} 섹터: " + ", ".join(dict.fromkeys(aliases))
    return descriptions


SECTOR_DESCRIPTIONS = _load_sector_descriptions()
SECTOR_NAMES = list(SECTOR_DESCRIPTIONS.keys())


def embedding_classify(text: str, embed_model, sector_embs=None) -> dict:
    """BGE-m3-ko로 text와 21개 섹터 설명 임베딩 간 코사인 유사도 계산, 최고 섹터 반환.
    sector_embs를 미리 계산해서 넘기면 매 호출마다 섹터 설명 재임베딩을 안 해도 됨(배치 처리용)."""
    import numpy as np
    if sector_embs is None:
        sector_embs = embed_model.encode(list(SECTOR_DESCRIPTIONS.values()), normalize_embeddings=True)
    text_emb = embed_model.encode([text], normalize_embeddings=True)[0]
    sims = np.array(sector_embs) @ text_emb
    ranking = np.argsort(-sims)
    best_idx = ranking[0]
    return {
        "sector": SECTOR_NAMES[best_idx],
        "confidence": float(sims[best_idx]),
        "top3": [(SECTOR_NAMES[i], float(sims[i])) for i in ranking[:3]],
    }


_SECTOR_LIST_STR = ", ".join(SECTOR_NAMES)


def classify_pdf_sector(pdf_path, embed_model=None) -> dict:
    """[11] 프로덕션 진입점 — 채택된 임베딩 기반 분류(golden set 정확도 92% vs LLM zero-shot
    58%, [11] 참고)로 PDF의 섹터를 판별. 첫 페이지 텍스트(제목+본문 앞부분)만 사용 — 실측상
    섹터를 규정하는 신호(회사명, 업종 특화 용어)가 첫 페이지에 이미 충분히 나타남([1]에서도
    확인했듯 표/차트가 아닌 서술형 텍스트 자체가 문서의 20~30%뿐이라 첫 페이지가 상대적으로
    텍스트 밀도가 높은 페이지인 경우가 많음)."""
    import fitz
    if embed_model is None:
        from embedding import get_embedding_model
        embed_model = get_embedding_model()
    doc = fitz.open(str(pdf_path))
    text = doc[0].get_text()
    doc.close()
    return embedding_classify(text, embed_model)


def llm_classify(text: str, qwen_model, qwen_processor, device, max_new_tokens: int = 20) -> dict:
    """로컬 Qwen2.5-VL에 zero-shot으로 섹터 하나를 고르게 함."""
    import torch
    prompt = (
        f"다음은 증권사 산업/기업 리포트의 첫 부분입니다. 이 문서가 다음 21개 섹터 중 "
        f"어디에 가장 해당하는지 정확히 하나만 고르세요: {_SECTOR_LIST_STR}\n\n"
        f"문서 내용:\n{text}\n\n"
        f"섹터 이름 하나만 출력하세요(다른 설명 없이)."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat_text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[chat_text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    raw = qwen_processor.decode(trimmed, skip_special_tokens=True).strip()
    del inputs, out
    if device == "mps":
        torch.mps.empty_cache()
    # 모델이 목록에 없는 표현을 섞어 낼 수 있어(예: "이 문서는 건설업입니다") 부분문자열로 매칭
    matched = next((s for s in SECTOR_NAMES if s in raw), None)
    return {"sector": matched, "raw_output": raw}
