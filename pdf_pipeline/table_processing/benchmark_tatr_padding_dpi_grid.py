"""[17] TATR로 (No Padding vs Padding vs Adaptive Padding) x (DPI 150/200/300) 3x3 그리드 테스트.

[16]에서 검증한 "TATR(구조 탐지) + pdfplumber(텍스트 채움)" 파이프라인을 그대로 쓰되, 이번엔 표
크롭 자체를 9가지 방식으로 만들어서 TATR의 구조 인식 품질(=최종 usable rows)과 지연이 어떻게
달라지는지 측정한다.

패딩은 "물리적 크기"(pt) 기준으로 정의해서 DPI와 독립적으로 만든다(패딩량이 DPI에 따라 픽셀 수만
달라지고 실제 여백 크기는 동일해야 공정 비교):
  - no_padding: 여백 0
  - padding: 사방 균일 5.76pt(= [12]/패딩전용 실험의 150dpi 12px와 동일한 물리적 크기)
  - adaptive_padding: 상단 16.8pt(150dpi 35px 상당) / 좌우·하단 5.76pt(150dpi 12px 상당)

각 조합마다 PyMuPDF `clip`으로 원본 PDF에서 해당 DPI로 직접 재렌더링(150dpi 사전탐지 + 300dpi
부분 재렌더링을 썼던 [12]와 달리, 이번은 "그 DPI로 처음부터 렌더링하면 TATR이 더 잘 뽑아내는가"를
보는 것이므로 탐지 없이 바로 지정 DPI로 렌더링).
"""

import json
import time
from pathlib import Path

import fitz
import pdfplumber
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_table_router import RouterThresholds, detect_and_route  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "pdf_pipeline" / "reference" / "LGCNS" / "20260721_company_279243000.pdf"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "result_tatr_padding_dpi_grid.json"

DETECT_DPI = 150  # 표 위치 탐지는 항상 150dpi(기존과 동일 — 그리드는 "최종 크롭 렌더링 DPI"만 바꿈)
DETECT_SCALE = DETECT_DPI / 72
CONF_THRESHOLD = 0.6

# 패딩량은 150dpi px 기준값을 pt로 환산해 고정(DPI 무관하게 물리적 크기 동일)
PADDING_STYLES = {
    "no_padding": {"top_pt": 0.0, "side_pt": 0.0},
    "padding": {"top_pt": 12 / DETECT_SCALE, "side_pt": 12 / DETECT_SCALE},
    "adaptive_padding": {"top_pt": 35 / DETECT_SCALE, "side_pt": 12 / DETECT_SCALE},
}
DPI_LIST = [150, 200, 300]
# [18] 자가검증: 300dpi가 3개 후보 중 최고였을 뿐 진짜 최적점인지 확인하기 위해 더 높은 DPI까지 확장
EXTENDED_DPI_LIST = [300, 400, 500, 600]


def get_base_tables():
    """[13] 라우터로 COMPLEX 표 12개의 원본 bbox(px@150dpi)를 pt로 환산해 반환."""
    routed = detect_and_route(RouterThresholds())
    complex_ = [r for r in routed if r["complexity"] == "complex"]
    tables = []
    for r in complex_:
        x1, y1, x2, y2 = r["bbox_px"]
        bbox_pt = (x1 / DETECT_SCALE, y1 / DETECT_SCALE, x2 / DETECT_SCALE, y2 / DETECT_SCALE)
        tables.append({"page": r["page"], "table_idx": r["table_idx"], "bbox_pt": bbox_pt})
    return tables


def apply_padding(bbox_pt, page_w_pt, page_h_pt, top_pt, side_pt):
    x1, y1, x2, y2 = bbox_pt
    return (
        max(0.0, x1 - side_pt), max(0.0, y1 - top_pt),
        min(page_w_pt, x2 + side_pt), min(page_h_pt, y2 + side_pt),
    )


def run_combo(pad_name, pad_cfg, dpi, tables, doc_fitz, pdf_pp, model, processor):
    scale = dpi / 72
    total_usable = 0
    total_s = 0.0
    per_table = []
    for t in tables:
        page_fz = doc_fitz[t["page"] - 1]
        page_pp = pdf_pp.pages[t["page"] - 1]
        padded_pt = apply_padding(t["bbox_pt"], page_fz.rect.width, page_fz.rect.height,
                                   pad_cfg["top_pt"], pad_cfg["side_pt"])
        rect = fitz.Rect(*padded_pt)
        pix = page_fz.get_pixmap(dpi=dpi, clip=rect)
        img_path = OUT_DIR / f"_tmp_grid_{pad_name}_{dpi}_p{t['page']}_{t['table_idx']}.png"
        pix.save(str(img_path))
        img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=CONF_THRESHOLD, target_sizes=target_sizes
        )[0]

        usable = 0
        for label_id, box in zip(results["labels"], results["boxes"]):
            if model.config.id2label[label_id.item()] != "table row":
                continue
            rx1, ry1, rx2, ry2 = [v.item() for v in box]
            row_pt = (padded_pt[0] + rx1 / scale, padded_pt[1] + ry1 / scale,
                      padded_pt[0] + rx2 / scale, padded_pt[1] + ry2 / scale)
            try:
                text = page_pp.crop(row_pt).extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                usable += 1
        elapsed = round(time.perf_counter() - t0, 3)
        total_s += elapsed
        total_usable += usable
        per_table.append({"page": t["page"], "table_idx": t["table_idx"], "elapsed_s": elapsed, "usable_rows": usable})
        img_path.unlink(missing_ok=True)
    return total_usable, round(total_s, 3), per_table


def main(extended: bool = False):
    tables = get_base_tables()
    styles = {"adaptive_padding": PADDING_STYLES["adaptive_padding"]} if extended else PADDING_STYLES
    dpi_list = EXTENDED_DPI_LIST if extended else DPI_LIST
    result_path = OUT_DIR / ("result_tatr_dpi_extended.json" if extended else "result_tatr_padding_dpi_grid.json")
    print(f"대상 표 {len(tables)}개, 그리드 {len(styles)}x{len(dpi_list)}={len(styles)*len(dpi_list)} 조합", flush=True)

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    model.eval()

    doc_fitz = fitz.open(str(PDF_PATH))
    pdf_pp = pdfplumber.open(str(PDF_PATH))

    grid_results = {}
    for pad_name, pad_cfg in styles.items():
        for dpi in dpi_list:
            key = f"{pad_name}_dpi{dpi}"
            usable, elapsed_s, per_table = run_combo(pad_name, pad_cfg, dpi, tables, doc_fitz, pdf_pp, model, processor)
            grid_results[key] = {"padding": pad_name, "dpi": dpi, "total_usable_rows": usable,
                                  "total_elapsed_s": elapsed_s, "per_table": per_table}
            print(f"[{key}] usable_rows={usable}, elapsed={elapsed_s}s", flush=True)

    pdf_pp.close()
    doc_fitz.close()

    result_path.write_text(json.dumps(grid_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 요약 ===")
    print(f"{'조합':<28}{'usable_rows':>14}{'elapsed_s':>12}")
    for key, r in grid_results.items():
        print(f"{key:<28}{r['total_usable_rows']:>14}{r['total_elapsed_s']:>12}")
    print(f"\n[비교] Docling(TableFormer) 기준: 228행, 20.24s / TATR(무패딩,150dpi 상당) 기준: [16]에서 228행, 3.25s")
    print(f"[result] saved to {result_path}")


if __name__ == "__main__":
    import sys as _sys
    main(extended="extended" in _sys.argv)
