# -*- coding: utf-8 -*-
"""figure_classifier: 경량 그림 유형 분류기 (S3 게이트).

docling-project/DocumentFigureClassifier-v2.5 (EfficientNet-B0, 4.08M, MIT)를 로드해
크롭 1장을 26개 유형 중 하나로 분류하고 useful/junk/other 로 라우팅한다.
VLM(수 초)보다 1000배 싼 밀리초 추론이라, 명백한 junk(로고·사진·아이콘 등)는
VLM을 태우지 않고 여기서 걷어낸다. 애매하면(other·저신뢰) VLM으로 넘겨 FN을 방지.

s2 에서 `--route clf` 일 때만 사용. 검증: 한국 리서치 차트 94/94 VLM 일치(conf~1.0)."""
from __future__ import annotations

from pathlib import Path

MODEL_ID = "docling-project/DocumentFigureClassifier-v2.5"

# 26개 라벨 → 라우팅 (모델카드 verbatim 기준)
USEFUL = {"line_chart", "bar_chart", "pie_chart", "scatter_plot", "box_plot",
          "flow_chart", "table", "geographical_map", "topographical_map",
          "engineering_drawing", "chemistry_structure"}
# 안전 junk = 투자 수치가 없는 게 확실한 것만. (hana_17 실측 검수 근거)
#   로고·제품사진은 VLM이 자주 FP(useful 오판)하므로 여기서 확실히 컷.
JUNK = {"logo", "photograph", "icon", "signature", "stamp", "qr_code", "bar_code",
        "calendar", "music", "crossword_puzzle"}
# 아래는 데이터 포함 가능(인포그래픽·풀페이지 차트·페이지 미리보기)이라 자동 컷 금지 → VLM 판단:
#   full_page_image, page_thumbnail, screenshot_from_computer, screenshot_from_manual, other

# 보수적 임계: junk 라벨이라도 이 확신 미만이면 컷하지 않고 VLM로 넘김 (FN 회피)
JUNK_CONF = 0.70

_model = None
_tf = None
_id2label = None


def _load():
    global _model, _tf, _id2label
    if _model is not None:
        return
    import torch
    import torchvision.transforms as T
    from transformers import EfficientNetForImageClassification
    _model = EfficientNetForImageClassification.from_pretrained(MODEL_ID)
    _model.eval()
    try:
        _model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        pass
    _id2label = _model.config.id2label
    _tf = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.47853944, 0.4732864, 0.47434163]),
    ])


def route_of(label: str, conf: float) -> str:
    """라벨·확신 → 'junk' | 'useful' | 'other'."""
    if label in JUNK and conf >= JUNK_CONF:
        return "junk"
    if label in USEFUL:
        return "useful"
    return "other"


def classify(path: Path | str) -> dict | None:
    """크롭 1장 → {label, confidence, route}. 열기 실패 시 None."""
    _load()
    import torch
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    dev = next(_model.parameters()).device
    with torch.no_grad():
        x = _tf(im).unsqueeze(0).to(dev)
        p = torch.softmax(_model(x).logits, 1)[0]
        i = int(p.argmax())
    label = _id2label[i]
    conf = float(p[i])
    return {"label": label, "confidence": round(conf, 4), "route": route_of(label, conf)}


def available() -> bool:
    try:
        _load()
        return True
    except Exception as e:
        print(f"[figure_classifier] 로드 실패: {e}")
        return False


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, "→", classify(p))
