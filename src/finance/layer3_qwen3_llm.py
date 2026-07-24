"""로컬 Qwen3 텍스트 LLM 래퍼 — 두 가지 용도의 모델을 각각 캐싱해서 제공한다.

1. 기본 모델 (load_model/generate): Layer3 뉴스 선정의 "LLM reasoning 검증" 단계.
   Qwen3-4B는 이 프로젝트를 개발한 M2 맥(통합 메모리 공유)에서 다른 프로그램들과 함께
   돌리면 메모리 압박으로 스와핑이 심해져 응답이 지나치게 느려졌다(50분+). 검증 품질보다
   응답 속도를 우선해 경량 모델(Qwen3-0.6B)을 기본값으로 쓴다 — 필요하면 QWEN3_MODEL_SIZE
   환경변수로 다시 올릴 수 있다.
2. Reasoning 전용 모델 (load_reasoning_model/generate_reasoning): Layer3 감성분석에서
   snunlp/KR-FinBert-SC(판별 모델)가 이미 정한 라벨을 자연어로 설명하는 용도.
   Qwen3-4B-Instruct-2507은 이 맥(M2, 통합 메모리 공유) 환경에서 다운로드+로딩이 너무
   오래 걸려(다른 프로그램들과 메모리 경합) Qwen3-1.7B로 낮췄다. QWEN3_REASONING_MODEL_SIZE
   환경변수로 다시 올리거나 내릴 수 있다.

pdf_pipeline/run_baseline.py의 Qwen2.5-VL 로딩 방식(device: mps, dtype: bfloat16)을
텍스트 전용 모델에 맞게 재사용한다.

사전 다운로드(선택, models/에 캐시): python scripts/download_models.py qwen3 (또는 qwen3-reasoning)
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

# 메모리/속도 여유가 있으면 "4B-Instruct-2507" 등으로 올릴 수 있음
REASONING_MODEL_SIZE = os.environ.get("QWEN3_REASONING_MODEL_SIZE", "1.7B")
REASONING_REPO_ID = f"Qwen/Qwen3-{REASONING_MODEL_SIZE}"
REASONING_LOCAL_MODEL_PATH = ROOT / "models" / f"Qwen3-{REASONING_MODEL_SIZE}"

_MODEL_CACHE: dict[str, tuple] = {}


def _load(repo_id: str, local_path: Path) -> tuple:
    if repo_id in _MODEL_CACHE:
        return _MODEL_CACHE[repo_id]

    source = str(local_path) if local_path.exists() else repo_id
    # [수정 — 재일] "mps 아니면 cpu" 이진 분기라 Windows에서 CUDA GPU를 놔두고 CPU로 돌았다
    # — 뉴스 첫 수집이 종목당 ~5분 걸리던 주범(교차리뷰의 "MPS 전용 분기" 클래스). cuda 최우선.
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[qwen3] loading {source} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    _MODEL_CACHE[repo_id] = (model, tokenizer)
    return model, tokenizer


def load_model():
    """기본 모델 (뉴스 선정 LLM reasoning 검증용, 기본 Qwen3-0.6B)."""
    return _load(HUB_REPO_ID, LOCAL_MODEL_PATH)


def load_reasoning_model():
    """Reasoning 전용 모델 (Qwen3-4B-Instruct-2507, 감성분석 설명 생성용)."""
    return _load(REASONING_REPO_ID, REASONING_LOCAL_MODEL_PATH)


def _generate_with(model, tokenizer, prompt: str, max_new_tokens: int, enable_thinking: bool) -> str:
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


def generate(prompt: str, max_new_tokens: int = 800, enable_thinking: bool = False) -> str:
    """기본 모델로 단일 사용자 메시지에 대한 생성 텍스트를 반환한다 (결정적 디코딩)."""
    model, tokenizer = load_model()
    return _generate_with(model, tokenizer, prompt, max_new_tokens, enable_thinking)


def generate_reasoning(prompt: str, max_new_tokens: int = 200) -> str:
    """Reasoning 전용 모델(Qwen3-4B-Instruct-2507)로 생성 텍스트를 반환한다 (결정적 디코딩)."""
    model, tokenizer = load_reasoning_model()
    return _generate_with(model, tokenizer, prompt, max_new_tokens, enable_thinking=False)