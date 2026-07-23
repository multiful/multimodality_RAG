"""수집된 logos/ 이미지를 Qwen2.5-VL로 2차 검수 — 여러 로고가 섞여 있거나
배경이 지저분한(실사/건물/제품 등) 이미지를 제거한다.

픽셀 휴리스틱(가로세로비, 배경색 균일도)은 로고들 사이 여백이 흰/투명하면
통과시켜버리는 한계가 있어(예: 브랜드 아이콘 그리드), VLM으로 실제 내용을 본다.

SVG는 대상에서 제외(공식 브랜드 에셋으로 간주, 기존 정책 유지).
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
LOGOS_DIR = ROOT / "logos"
MODEL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
REPORT_PATH = ROOT / "collect" / "filter_report.txt"
REPORT2_PATH = ROOT / "collect" / "filter_report_pass2.txt"

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
MAX_SIDE = 448  # 판별용이라 크게 필요 없음 — 속도 위해 축소

PROMPT = (
    "당신은 로고 학습 데이터셋을 검수하는 필터입니다. "
    "이 이미지에 하나의 기업 로고만 깔끔하게 나와 있으면 OK라고만 답하세요. "
    "여러 개의 서로 다른 로고/아이콘이 섞여 있거나, 관련 없는 다른 브랜드가 같이 있거나, "
    "실사 사진(건물, 매장, 제품, 사람, 화면 스크린샷)이 배경으로 있으면 REJECT라고만 답하세요. "
    "반드시 OK 또는 REJECT 중 한 단어로만 답하세요."
)


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor, device


def judge(model, processor, device, path: Path) -> str:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return "REJECT"
    if max(img.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(trimmed, skip_special_tokens=True).strip().upper()
    if "REJECT" in answer:
        return "REJECT"
    if "OK" in answer:
        return "OK"
    return "UNCLEAR"  # 파싱 애매하면 보수적으로 살려둠(수동 확인용으로 로그엔 남김)


def judge_grounded(model, processor, device, path: Path, brand: str) -> str:
    """1차 통과작을 대상으로, 폴더명(기대 기업명)을 프롬프트에 명시해 더 엄격히 재검증.
    generic한 '로고 하나만 있나' 질문은 LogoKit류 브랜드 홍보 이미지(여러 로고+UI/코드가
    섞였지만 화면 전체는 나름 '깔끔'해 보이는 경우)를 놓치는 사례가 있어 추가."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return "REJECT"
    if max(img.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))

    prompt = (
        f"이 이미지는 '{brand}' 기업의 로고 학습 데이터 후보입니다. "
        f"이미지에 '{brand}'의 로고 단 하나만 깔끔하게 나와 있으면 OK라고만 답하세요. "
        f"다른 기업/브랜드 로고가 함께 있거나 여러 로고가 나열되어 있거나, "
        f"코드/UI 화면, 문서, 광고 배너처럼 로고 외의 내용이 섞여 있으면 REJECT라고만 답하세요. "
        f"'{brand}'와 관련 없어 보이면 REJECT라고만 답하세요. "
        f"반드시 OK 또는 REJECT 중 한 단어로만 답하세요."
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(trimmed, skip_special_tokens=True).strip().upper()
    if "REJECT" in answer:
        return "REJECT"
    if "OK" in answer:
        return "OK"
    return "UNCLEAR"


def main_pass2():
    model, processor, device = load_model()
    ok_files = []
    with open(REPORT_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[0] == "OK":
                ok_files.append(LOGOS_DIR / parts[1])

    print(f"2차(브랜드 특정) 검수 대상 {len(ok_files)}장", flush=True)
    kept = rejected = unclear = 0
    with open(REPORT2_PATH, "w", encoding="utf-8") as report:
        for i, path in enumerate(ok_files, 1):
            if not path.exists():
                continue
            folder = path.parent.name
            brand = folder.split("_", 1)[1].replace("_", " ") if "_" in folder else folder
            verdict = judge_grounded(model, processor, device, path, brand)
            rel = path.relative_to(LOGOS_DIR)
            report.write(f"{verdict}\t{rel}\n")
            report.flush()
            if verdict == "REJECT":
                rejected += 1
                path.unlink(missing_ok=True)
                print(f"[{i}/{len(ok_files)}] REJECT {rel}", flush=True)
            elif verdict == "OK":
                kept += 1
            else:
                unclear += 1
            if i % 50 == 0:
                print(f"--- 2차 진행 {i}/{len(ok_files)} | OK {kept} / REJECT {rejected} / UNCLEAR {unclear} ---", flush=True)

    print(f"\n2차 완료. OK={kept} REJECT={rejected} UNCLEAR={unclear}", flush=True)
    print(f"리포트: {REPORT2_PATH}", flush=True)


def main():
    model, processor, device = load_model()
    files = sorted(p for p in LOGOS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    print(f"검수 대상 {len(files)}장", flush=True)

    kept = rejected = unclear = 0
    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        for i, path in enumerate(files, 1):
            verdict = judge(model, processor, device, path)
            rel = path.relative_to(LOGOS_DIR)
            report.write(f"{verdict}\t{rel}\n")
            report.flush()
            if verdict == "REJECT":
                rejected += 1
                path.unlink(missing_ok=True)
                print(f"[{i}/{len(files)}] REJECT {rel}", flush=True)
            elif verdict == "OK":
                kept += 1
            else:
                unclear += 1
                print(f"[{i}/{len(files)}] UNCLEAR {rel} (보류, 수동 확인 필요)", flush=True)
            if i % 50 == 0:
                print(f"--- 진행 {i}/{len(files)} | OK {kept} / REJECT {rejected} / UNCLEAR {unclear} ---", flush=True)

    print(f"\n완료. OK={kept} REJECT={rejected} UNCLEAR={unclear}", flush=True)
    print(f"리포트: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pass2":
        main_pass2()
    else:
        main()
