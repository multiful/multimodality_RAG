"""Download the VLM/LLM weights used by this project into models/ (gitignored).

Usage:
    python scripts/download_models.py            # download all
    python scripts/download_models.py qwen        # only Qwen2.5-VL-7B-Instruct
    python scripts/download_models.py llava        # only LLaVA-OneVision-7B-OV
    python scripts/download_models.py qwen3        # only Qwen3-0.6B (Layer3 뉴스 선정용)
    python scripts/download_models.py qwen3-reasoning  # Qwen3-1.7B (Layer3 감성분석 reasoning용)
    python scripts/download_models.py finbert       # KR-FinBert-SC (Layer3 감성분석 분류용)
"""

import os
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct"),
    "llava": ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-7b-ov-hf"),
    "qwen3": ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    "qwen3-reasoning": ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    "finbert": ("snunlp/KR-FinBert-SC", "KR-FinBert-SC"),
}


def download(key: str) -> None:
    repo_id, dirname = MODELS[key]
    target_dir = os.path.join(ROOT, "models", dirname)
    print(f"Downloading {repo_id} -> {target_dir}")
    path = snapshot_download(repo_id=repo_id, local_dir=target_dir, max_workers=4)
    print(f"Done: {path}")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(MODELS.keys())
    for key in keys:
        if key not in MODELS:
            raise SystemExit(f"Unknown model key '{key}', choose from {list(MODELS)}")
        download(key)
