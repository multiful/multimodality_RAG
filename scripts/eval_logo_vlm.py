"""Zero-shot logo -> company evaluation for Qwen2.5-VL-7B / LLaVA-OneVision-7B (PRD 5.1-②).

Usage:
    python scripts/eval_logo_vlm.py --model qwen  --data <logos_dir> --out results/qwen.jsonl
    python scripts/eval_logo_vlm.py --model llava --data <logos_dir> --out results/llava.jsonl

Each image gets one VQA call asking for the top-3 candidate companies; the free-text
answer is normalized to tickers via entity linking. Results are appended to JSONL
(resumable). Metrics are computed afterwards by eval_metrics.py.
"""

import argparse
import io
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entity_linking import link

MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    # local_dir download (HF cache symlinks unsupported on this Windows box)
    "qwen3": os.environ.get("QWEN3_VL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
}

PROMPTS = {
    "logo": (
        "This image shows a company logo. Which company does it belong to? "
        "Answer with exactly 3 candidate company names, most likely first, "
        "separated by commas. Company names only, no other text."
    ),
    "product": (
        "This image shows a product. Which company makes or sells it? "
        "Answer with exactly 3 candidate company names, most likely first, "
        "separated by commas. Company names only, no other text."
    ),
}

MAX_SIDE = 768
MIN_SIDE = 56  # Qwen vision tower needs >=28px per side after patching; be safe


def load_image(path: str) -> Image.Image:
    if path.lower().endswith(".svg"):
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM

        drawing = svg2rlg(path)
        if drawing is None:
            raise ValueError("svg2rlg returned None")
        scale = 512 / max(drawing.width, drawing.height, 1)
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)
        png_bytes = renderPM.drawToString(drawing, fmt="PNG")
        img = Image.open(io.BytesIO(png_bytes))
    else:
        img = Image.open(path)

    # Flatten transparency onto white so dark logos stay visible.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        s = MIN_SIDE / min(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def build_model(key: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    repo = MODELS[key]
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(
        repo, quantization_config=quant, device_map="cuda:0"
    )
    model.eval()
    return processor, model


@torch.inference_mode()
def infer(processor, model, img: Image.Image, prompt: str) -> tuple[str, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0

    gen = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(gen, skip_special_tokens=True).strip()
    return answer, dt


def collect_samples(data_dir: str) -> list[tuple[str, str]]:
    samples = []
    for folder in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, folder)
        if not os.path.isdir(fpath):
            continue
        ticker = folder.split("_")[0]
        for fn in sorted(os.listdir(fpath)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".svg")):
                samples.append((os.path.join(fpath, fn), ticker))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", choices=list(PROMPTS), default="logo")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = collect_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass
    todo = [(p, t) for p, t in samples if os.path.relpath(p, args.data) not in done]
    print(f"[{args.model}] total={len(samples)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    processor, model = build_model(args.model)
    print(f"[{args.model}] model loaded, VRAM={torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)

    n_err = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, (path, ticker) in enumerate(todo):
            rel = os.path.relpath(path, args.data)
            rec = {"file": rel, "label": ticker}
            try:
                img = load_image(path)
                answer, dt = infer(processor, model, img, PROMPTS[args.task])
                rec.update({"answer": answer, "pred": link(answer, top_k=3), "ms": round(dt, 1)})
            except Exception as e:
                n_err += 1
                rec.update({"error": f"{type(e).__name__}: {e}"})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i == len(todo) - 1:
                print(f"[{args.model}] {i + 1}/{len(todo)} errors={n_err}", flush=True)

    print(f"[{args.model}] DONE errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
