"""[15] 표 라우팅(Docling/TableFormer) 단계 전용 latency 추가 실험 —
지금까지 안 건드린 두 레버: (a) TableFormerMode.FAST vs ACCURATE(기본값), (b) AcceleratorDevice MPS vs CPU.
프로세스 병렬 변수를 배제하기 위해 12개 COMPLEX 크롭을 순차 처리로 비교(설정 자체의 순수 효과만 측정).
"""

import sys
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

CROP_DIR = Path(__file__).resolve().parent / "table_crops_router_complex"


def build_converter(mode: str, device: str) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.table_structure_options.mode = TableFormerMode.FAST if mode == "fast" else TableFormerMode.ACCURATE
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=device)
    return DocumentConverter(format_options={InputFormat.IMAGE: PdfFormatOption(pipeline_options=opts)})


def run(mode: str, device: str):
    converter = build_converter(mode, device)
    crop_paths = sorted(CROP_DIR.glob("*.png"))
    # 워밍업 1회(모델 로딩 비용 제외)
    converter.convert(str(crop_paths[0]))

    t0 = time.perf_counter()
    total_rows = 0
    per_file = []
    for p in crop_paths:
        tf0 = time.perf_counter()
        res = converter.convert(str(p))
        n_rows = sum(t.export_to_dataframe(res.document).shape[0] for t in res.document.tables)
        elapsed = round(time.perf_counter() - tf0, 3)
        total_rows += n_rows
        per_file.append((p.name, elapsed, n_rows))
        print(f"  [{mode}/{device}] {p.name}: {elapsed}s -> {n_rows}행", flush=True)
    total_s = round(time.perf_counter() - t0, 3)
    print(f"\n[{mode}/{device}] 순차 처리 {len(crop_paths)}개 총 {total_s}s, 총 {total_rows}행\n", flush=True)
    return total_s, total_rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "accurate"
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    run(mode, device)
