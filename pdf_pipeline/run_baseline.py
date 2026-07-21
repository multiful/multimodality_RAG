"""PDF 페이지 라우팅 베이스라인: 이미지/텍스트/테이블 각각 최소 구현으로 처리 후
엔티티를 추출하고, ground_truth_*.json 대비 recall을 측정한다.

- 텍스트: pdfplumber 텍스트 추출 (BGE-M3/BM25 인덱싱은 DB가 없는 현재 단계라 생략,
  엔티티 추출에 필요한 원문만 사용)
- 테이블: pdfplumber table 추출 → 마크다운 변환 (TableFormer/Docling은 고도화 단계에서 적용 예정)
- 이미지: 임베디드 래스터 이미지 유무 또는 벡터 드로잉(차트) 밀도로 판별 →
  해당 페이지 렌더링본을 Qwen2.5-VL로 설명/엔티티 추출
- 이미지·테이블에서 나온 내용은 memory_store.json에 저장해 추후 LLM 답변 생성 시 같이 제공
- 엔티티 추출은 페이지별 결과를 모두 합쳐 Qwen2.5-VL(텍스트 전용 프롬프트) 1회 호출로 수행
"""

import json
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "20260721_company_279243000.pdf"
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
OUT_DIR = Path(__file__).resolve().parent
PAGE_IMG_DIR = OUT_DIR / "rendered_pages"
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth_064400.json"
MEMORY_PATH = OUT_DIR / "memory_store.json"
ENTITIES_PATH = OUT_DIR / "extracted_entities.json"
REPORT_PATH = OUT_DIR / "recall_report.md"

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
    memory = {"pages": []}
    all_text_parts = []

    model, processor, device = load_model()

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for i, (page_pp, page_fz) in enumerate(zip(pdf.pages, doc_fitz), start=1):
            print(f"\n=== page {i} ===", flush=True)
            cls = classify_page(page_pp, page_fz)
            print(f"  classify: {cls}", flush=True)

            page_record = {"page": i, "classification": cls, "text": "", "tables_markdown": [], "image_descriptions": []}

            # 텍스트 라우팅 (baseline: pdfplumber 텍스트 추출)
            if cls["has_text"]:
                text = page_pp.extract_text() or ""
                page_record["text"] = text
                all_text_parts.append(f"[p.{i} 텍스트]\n{text}")

            # 테이블 라우팅 (baseline: pdfplumber table 추출 → markdown)
            if cls["has_table"]:
                for t_idx, table in enumerate(page_pp.extract_tables(), start=1):
                    md = table_to_markdown(table)
                    if md:
                        page_record["tables_markdown"].append(md)
                        all_text_parts.append(f"[p.{i} 표{t_idx}]\n{md}")
                        print(f"  table {t_idx}: {len(table)} rows extracted", flush=True)

            # 이미지 라우팅 (baseline: 페이지 렌더링본을 VLM으로 설명/엔티티 추출)
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
                all_text_parts.append(f"[p.{i} 이미지/차트 설명]\n{desc}")
                print(f"  image desc: {desc[:120]}...", flush=True)

            memory["pages"].append(page_record)

    doc_fitz.close()

    MEMORY_PATH.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[memory] saved to {MEMORY_PATH}", flush=True)

    # 엔티티 추출 (baseline: 전체 페이지 결과를 합쳐 1회 LLM 호출)
    full_context = "\n\n".join(all_text_parts)
    entity_prompt = (
        "다음은 한 증권사 기업분석 리포트에서 텍스트/표/이미지·차트로부터 추출한 내용입니다. "
        "이 문서에 등장하는 모든 '기업/기관 엔티티'를 빠짐없이 나열하세요. "
        "표 안에서만 언급된 계약 상대방 기업, 차트 범례/파이차트에만 나온 기업도 반드시 포함하세요. "
        "형식: 한 줄에 하나씩 '기업명 (알고 있다면 종목코드)' 형태로만 출력하고 다른 설명은 하지 마세요.\n\n"
        f"{full_context}"
    )
    print("\n[entity extraction] running over aggregated context "
          f"({len(full_context)} chars)...", flush=True)
    entity_raw = vlm_generate(model, processor, device, entity_prompt, image=None, max_new_tokens=500)
    print(f"[entity extraction] raw output:\n{entity_raw}", flush=True)

    ENTITIES_PATH.write_text(entity_raw, encoding="utf-8")

    # Recall 평가
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    target_set = ground_truth["entity_recall_target_set"]
    extracted_lower = entity_raw.lower()

    hits, misses = [], []
    for ent in target_set:
        norm = ent.lower().replace(" ", "")
        if norm in extracted_lower.replace(" ", ""):
            hits.append(ent)
        else:
            misses.append(ent)

    recall = len(hits) / len(target_set)
    report_lines = [
        "# Baseline 엔티티 Recall 리포트",
        "",
        f"- 대상 문서: {PDF_PATH.name}",
        f"- 정답 엔티티 수: {len(target_set)}",
        f"- 추출 성공(hit): {len(hits)}",
        f"- 누락(miss): {len(misses)}",
        f"- **Recall: {recall:.1%}**",
        "",
        "## Hit",
        *[f"- {h}" for h in hits],
        "",
        "## Miss",
        *[f"- {m}" for m in misses],
        "",
        "## 추출된 원본 엔티티 리스트 (LLM 출력)",
        "```",
        entity_raw,
        "```",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[recall] {recall:.1%} ({len(hits)}/{len(target_set)})", flush=True)
    print(f"[report] saved to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
