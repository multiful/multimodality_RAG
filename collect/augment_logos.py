"""필터링된 logos/ 클린 이미지에 기업당 5장씩 가벼운 증강을 추가한다.

로고는 좌우/상하 반전이나 색상 반전을 하면 브랜드가 왜곡되므로 제외하고,
약한 회전 + 밝기/대비 + 미세 확대 조합만 사용한다. (filter_logos_vlm.py로
정리된 이후에 돌려야 함 — 안 그러면 나쁜 이미지까지 증강됨)
"""

import random
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
LOGOS_DIR = ROOT / "logos"
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
N_AUG = 5

random.seed(42)


def augment_once(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")

    angle = random.uniform(-10, 10)
    img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0), resample=Image.BICUBIC)

    zoom = random.uniform(0.95, 1.05)
    w, h = img.size
    nw, nh = max(1, int(w * zoom)), max(1, int(h * zoom))
    img = img.resize((nw, nh), Image.BICUBIC)

    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.88, 1.12))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.88, 1.12))
    alpha = img.split()[3]
    img = Image.merge("RGBA", (*rgb.split(), alpha))
    return img


def main():
    total = 0
    dirs = sorted(p for p in LOGOS_DIR.iterdir() if p.is_dir())
    for d in dirs:
        sources = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in RASTER_EXTS]
        if not sources:
            print(f"{d.name}: 원본 없음, 스킵")
            continue
        for i in range(1, N_AUG + 1):
            src = random.choice(sources)
            try:
                img = Image.open(src)
                out = augment_once(img)
            except Exception as e:
                print(f"  skip {src.name}: {e}")
                continue
            out_path = d / f"{d.name}_aug_{i:02d}.png"
            out.save(out_path)
            total += 1
        print(f"{d.name}: +{N_AUG}")
    print(f"\n완료. 총 {total}장 증강 추가")


if __name__ == "__main__":
    main()
