"""PDF 페이지 라우팅 베이스라인: 이미지/텍스트/테이블 각각 최소 구현으로 처리해
memory_store.json(+ 페이지·단계별 소요시간)을 만든다.

- 텍스트: pdfplumber 텍스트 추출 (BGE-M3/BM25 인덱싱은 DB가 없는 현재 단계라 생략,
  엔티티 추출에 필요한 원문만 사용)
- 테이블: pdfplumber table 추출 → 마크다운 변환 (TableFormer/Docling은 고도화 단계에서 적용 예정)
- 이미지: 임베디드 래스터 이미지 유무 또는 벡터 드로잉(차트) 밀도로 판별 →
  해당 페이지 렌더링본을 Qwen2.5-VL로 설명/엔티티 추출
- 이미지·테이블에서 나온 내용은 memory_store.json에 저장해 추후 LLM 답변 생성 시 같이 제공

엔티티 추출/recall/precision/latency 집계는 extract_entities_and_eval.py에서 이어서 수행
(긴 컨텍스트를 한 번에 넣으면 MPS OOM이 나서 페이지 단위로 분리했음).
"""

import json
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "20260721_company_279243000.pdf"
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
PAGE_IMG_DIR = OUT_DIR / "rendered_pages"
MEMORY_PATH = OUT_DIR / "memory_store.json"

VECTOR_DRAWING_THRESHOLD = 40  # 이 이상이면 차트/그래픽으로 간주(휴리스틱, 고도화 시 조정 대상)
MAX_IMG_SIDE = 900  # VLM 입력 리사이즈 상한 — 원본 150dpi 풀페이지를 그대로 넣으면 추론이 극도로 느려짐


def resize_for_vlm(img: Image.Image) -> Image.Image:
    if max(img.size) <= MAX_IMG_SIDE:
        return img
    ratio = MAX_IMG_SIDE / max(img.size)
    return img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[model] loading Qwen2.5-VL-7B-Instruct on {device}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor, device


def vlm_generate(model, processor, device, prompt: str, image: Optional[Image.Image] = None, max_new_tokens: int = 400) -> str:
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def classify_page(page_pdfplumber, page_fitz) -> dict:
    tables = page_pdfplumber.find_tables()
    has_table = len(tables) > 0
    raster_images = page_fitz.get_images()
    drawings = page_fitz.get_drawings()
    has_image = len(raster_images) > 0 or len(drawings) > VECTOR_DRAWING_THRESHOLD
    text = page_pdfplumber.extract_text() or ""
    has_text = len(text.strip()) > 20
    return {
        "has_text": has_text,
        "has_table": has_table,
        "has_image": has_image,
        "n_tables": len(tables),
        "n_raster_images": len(raster_images),
        "n_drawings": len(drawings),
    }


def table_to_markdown(table_rows) -> str:
    if not table_rows:
        return ""
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in row] for row in table_rows]
    header = rows[0]
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    PAGE_IMG_DIR.mkdir(exist_ok=True)

    doc_fitz = fitz.open(str(PDF_PATH))
    memory = {"pages": [], "timing": {"model_load_s": None, "pages": []}}

    t0 = time.time()
    model, processor, device = load_model()
    memory["timing"]["model_load_s"] = round(time.time() - t0, 2)
    print(f"[timing] model load: {memory['timing']['model_load_s']}s", flush=True)

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            print(f"\n=== page {i} ===", flush=True)
            page_timing = {"page": i}

            t = time.time()
            cls = classify_page(page_pp, page_fz)
            page_timing["classify_s"] = round(time.time() - t, 3)
            print(f"  classify: {cls}  ({page_timing['classify_s']}s)", flush=True)

            page_record = {"page": i, "classification": cls, "text": "", "tables_markdown": [], "image_descriptions": []}

            # 텍스트 라우팅 (baseline: pdfplumber 텍스트 추출)
            t = time.time()
            if cls["has_text"]:
                text = page_pp.extract_text() or ""
                page_record["text"] = text
            page_timing["text_extract_s"] = round(time.time() - t, 3)

            # 테이블 라우팅 (baseline: pdfplumber table 추출 → markdown)
            t = time.time()
            if cls["has_table"]:
                for t_idx, table in enumerate(page_pp.extract_tables(), start=1):
                    md = table_to_markdown(table)
                    if md:
                        page_record["tables_markdown"].append(md)
                        print(f"  table {t_idx}: {len(table)} rows extracted", flush=True)
            page_timing["table_extract_s"] = round(time.time() - t, 3)

            # 이미지 라우팅 (baseline: 페이지 렌더링본을 VLM으로 설명/엔티티 추출)
            t = time.time()
            if cls["has_image"]:
                pix = page_fz.get_pixmap(dpi=150)
                img_path = PAGE_IMG_DIR / f"page_{i}.png"
                pix.save(str(img_path))
                img = resize_for_vlm(Image.open(img_path).convert("RGB"))
                desc = vlm_generate(
                    model, processor, device,
                    "이 이미지는 증권사 리포트 페이지입니다. 페이지 안의 로고, 차트, 그래프에 나타난 "
                    "기업명/브랜드명과 핵심 수치(비중, 금액 등)만 간단히 목록으로 정리해주세요. "
                    "본문 텍스트는 요약하지 말고, 시각 자료(로고/차트/그래프)에 있는 정보만 추출하세요.",
                    image=img,
                    max_new_tokens=300,
                )
                page_record["image_descriptions"].append(desc)
                print(f"  image desc: {desc[:120]}...", flush=True)
            page_timing["image_vlm_s"] = round(time.time() - t, 3)

            page_timing["page_total_s"] = round(
                page_timing["classify_s"] + page_timing["text_extract_s"]
                + page_timing["table_extract_s"] + page_timing["image_vlm_s"], 3
            )
            print(f"  [timing] page {i} total: {page_timing['page_total_s']}s "
                  f"(classify {page_timing['classify_s']}s / text {page_timing['text_extract_s']}s / "
                  f"table {page_timing['table_extract_s']}s / image {page_timing['image_vlm_s']}s)", flush=True)

            memory["pages"].append(page_record)
            memory["timing"]["pages"].append(page_timing)

    doc_fitz.close()

    memory["timing"]["total_pipeline_s"] = round(
        memory["timing"]["model_load_s"] + sum(p["page_total_s"] for p in memory["timing"]["pages"]), 2
    )

    MEMORY_PATH.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[memory] saved to {MEMORY_PATH}", flush=True)
    print(f"[timing] total (model load + all pages): {memory['timing']['total_pipeline_s']}s", flush=True)


if __name__ == "__main__":
    main()
