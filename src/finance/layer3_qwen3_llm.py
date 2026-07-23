"""로컬 Qwen3 텍스트 LLM 래퍼 — Layer3 뉴스 선정의 "LLM reasoning 검증" 단계에 사용.

pdf_pipeline/run_baseline.py의 Qwen2.5-VL 로딩 방식(device: mps, dtype: bfloat16)을
텍스트 전용 모델에 맞게 재사용한다.

Qwen3-4B는 이 프로젝트를 개발한 M2 맥(통합 메모리 공유)에서 다른 프로그램들과 함께 돌리면
메모리 압박으로 스와핑이 심해져 응답이 지나치게 느려졌다(50분+). 검증 품질보다 응답 속도를
우선해 같은 Qwen3 계열의 경량 모델(Qwen3-0.6B)로 기본값을 낮췄다 — 필요하면 QWEN3_MODEL_SIZE
환경변수로 다시 올릴 수 있다.

사전 다운로드(선택, models/에 캐시): python scripts/download_models.py qwen3
미리 받아두지 않으면 최초 호출 시 Hugging Face Hub에서 직접 받는다(~/.cache/huggingface).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent

# 메모리 여유가 있으면 "4B", "1.7B" 등으로 올릴 수 있음 (예: QWEN3_MODEL_SIZE=4B)
MODEL_SIZE = os.environ.get("QWEN3_MODEL_SIZE", "0.6B")
HUB_REPO_ID = f"Qwen/Qwen3-{MODEL_SIZE}"
LOCAL_MODEL_PATH = ROOT / "models" / f"Qwen3-{MODEL_SIZE}"

_model = None
_tokenizer = None


def _model_source() -> str:
    return str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else HUB_REPO_ID


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    source = _model_source()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[qwen3] loading {source} on {device}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 800, enable_thinking: bool = False) -> str:
    """Qwen3에 단일 사용자 메시지를 넣고 생성 텍스트만 반환한다 (결정적 디코딩)."""
    model, tokenizer = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()