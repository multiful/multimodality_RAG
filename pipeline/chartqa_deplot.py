# -*- coding: utf-8 -*-
"""chartqa_deplot: google/deplot (Pix2Struct) 로 차트 이미지를 구조화 데이터표로 변환.

ChartQA 계열 전문모델 — '차트→표(chart-to-table)'. Qwen3-VL의 짧은 ocr_text 대신
수치·축을 표로 뽑는다. 단 영어 학습 모델이라 한글 라벨은 취약(범례·제목 gibberish 가능).
변형3(베이스라인+분류기+ChartQA)의 '차트 추출' 단계에서 사용."""
from __future__ import annotations

from pathlib import Path
import time

MODEL_ID = "google/deplot"
PROMPT = "Generate underlying data table of the figure below:"

_model = None
_proc = None
_dev = None


def _load():
    global _model, _proc, _dev
    if _model is not None:
        return
    import torch
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _proc = Pix2StructProcessor.from_pretrained(MODEL_ID)
    _model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_ID).to(_dev)
    _model.eval()


def extract_table(path: Path | str, max_new_tokens: int = 512) -> dict:
    """차트 이미지 → {table(str), rows(int), seconds(float)}. 실패 시 table="" ."""
    _load()
    import torch
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return {"table": "", "rows": 0, "seconds": 0.0}
    t = time.time()
    inp = _proc(images=im, text=PROMPT, return_tensors="pt").to(_dev)
    with torch.no_grad():
        out = _model.generate(**inp, max_new_tokens=max_new_tokens)
    txt = _proc.decode(out[0], skip_special_tokens=True)
    # DePlot는 <0x0A>를 줄바꿈으로 사용
    txt = txt.replace("<0x0A>", "\n").strip()
    rows = sum(1 for ln in txt.splitlines() if "|" in ln)
    return {"table": txt, "rows": rows, "seconds": round(time.time() - t, 2)}


def available() -> bool:
    try:
        _load()
        return True
    except Exception as e:
        print(f"[chartqa_deplot] 로드 실패: {e}")
        return False


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = extract_table(p)
        print(f"\n=== {p} ({r['seconds']}s, {r['rows']}행) ===\n{r['table'][:800]}")
