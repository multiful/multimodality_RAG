"""NASDAQ-100 로고 수집기 (Python판) — collect_logos.ps1과 동일 로직, pwsh 없는 환경용.

$DATA 테이블은 collect_logos.ps1에서 그대로 파싱해서 쓰므로 두 스크립트가 항상
같은 기업 목록/필터 정책을 공유한다. (ps1이 원본, 이 파일은 실행용 포트)
"""

import hashlib
import re
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LOGOS_DIR = ROOT / "logos"
PS1_PATH = Path(__file__).resolve().parent / "collect_logos.ps1"
LOG_PATH = ROOT / "collect_log.txt"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
K = 20

BAD_RE = re.compile(
    r"history|evolution|evoluc|timeline|through the years|over the years|old and new|all logos|logos of|brand logos"
    r"|logo collection|collection of|logo pack|bundle|set of|comparison|versus| vs |infographic|chart|banner|wallpaper"
    r"|mockup|collage|compilation|grid|sprite sheet|icon set|icon pack|top \d+|ranking|alternatives?|competitor"
    r"|portfolio|showcase|our (client|partner|sponsor)|client list|sponsor|screenshot|storefront|building|signage?"
    r"|store front|변천|역사|모음|로고 모음|브랜드 모음|파트너사|고객사|스크린샷|매장|간판|건물",
    re.IGNORECASE,
)
GOOD_RE = re.compile(r"logo|로고|svg|png|vector|icon|brand|emblem|wordmark|symbol|transparent", re.IGNORECASE)
URL_BAD_RE = re.compile(r"history|evolution|collection|banner|wallpaper", re.IGNORECASE)
EXT_RE = re.compile(r"\.(svg|png|webp|jpg|jpeg|gif)(?:[?#]|$)", re.IGNORECASE)

session = requests.Session()
session.headers.update({"User-Agent": UA})


def log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_data():
    text = PS1_PATH.read_text(encoding="utf-8")
    m = re.search(r'\$DATA = @"\n(.*?)\n"@', text, re.DOTALL)
    if not m:
        raise RuntimeError("collect_logos.ps1 에서 $DATA 테이블을 찾지 못했습니다")
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        tick, brand, tok, dom = parts[0], parts[1], parts[2], parts[3]
        rows.append((tick, brand, tok, dom))
    return rows


def get_ext(url: str) -> str:
    m = EXT_RE.search(url.split("?")[0])
    return m.group(1).lower() if m else "png"


def valid_image_bytes(data: bytes) -> bool:
    if len(data) < 3072:
        return False
    head = data[:15]
    if b"<!DOC" in head or b"<html" in head:
        return False
    return True


def dhash_of(img: Image.Image):
    img = img.convert("L").resize((9, 8))
    px = list(img.getdata())
    bits = []
    for y in range(8):
        for x in range(8):
            bits.append(px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def hamming(a, b) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def test_clean_logo(path: Path) -> bool:
    """로고 하나만 깔끔하게 나오는지: 가로세로비 + 투명배경/단색배경 여부."""
    if path.suffix.lower() == ".svg":
        return True
    try:
        img = Image.open(path)
        w, h = img.size
    except Exception:
        return False
    if w < 32 or h < 32:
        return False
    ratio = w / h
    if ratio > 5.0 or ratio < 0.2:
        return False

    pts = [
        (1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2),
        (w // 2, 1), (w // 2, h - 2), (1, h // 2), (w - 2, h // 2),
    ]
    rgba = img.convert("RGBA")
    pixels = [rgba.getpixel(p) for p in pts]
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        transparent = sum(1 for p in pixels if p[3] < 20)
        return transparent >= 4
    rs = [p[0] for p in pixels]
    gs = [p[1] for p in pixels]
    bs = [p[2] for p in pixels]
    avg_r, avg_g, avg_b = sum(rs) / len(rs), sum(gs) / len(gs), sum(bs) / len(bs)
    max_diff = max(abs(r - avg_r) + abs(g - avg_g) + abs(b - avg_b) for r, g, b in zip(rs, gs, bs))
    return max_diff < 45


def try_add(url: str, path: Path, hashes: set, dhashes: list) -> bool:
    if path.exists():
        return False
    try:
        resp = session.get(url, timeout=25, headers={"Accept": "image/*,*/*;q=0.8"})
        resp.raise_for_status()
        data = resp.content
    except Exception:
        return False
    if not valid_image_bytes(data):
        return False
    path.write_bytes(data)
    if not test_clean_logo(path):
        path.unlink(missing_ok=True)
        return False
    md5 = hashlib.md5(data).hexdigest()
    if md5 in hashes:
        path.unlink(missing_ok=True)
        return False
    dh = None
    if path.suffix.lower() != ".svg":
        try:
            dh = dhash_of(Image.open(path))
        except Exception:
            dh = None
    if dh:
        for e in dhashes:
            if hamming(e, dh) <= 2:
                path.unlink(missing_ok=True)
                return False
        dhashes.append(dh)
    hashes.add(md5)
    return True


def ddg_search(query: str):
    try:
        r = session.get(
            "https://duckduckgo.com/",
            params={"q": query, "iax": "images", "ia": "images"},
            timeout=25,
        )
        m = re.search(r"vqd=['\"]?([\d-]+)", r.text)
        if not m:
            return []
        vqd = m.group(1)
        r2 = session.get(
            "https://duckduckgo.com/i.js",
            params={"l": "us-en", "o": "json", "q": query, "vqd": vqd, "p": "1"},
            headers={"Referer": "https://duckduckgo.com/"},
            timeout=25,
        )
        return r2.json().get("results", [])
    except Exception:
        return []


def bing_search(query: str, first: int):
    try:
        r = session.get(
            "https://www.bing.com/images/search",
            params={"q": query, "qft": "+filterui:photo-transparent", "first": first},
            timeout=25,
        )
        html = r.text.replace("&quot;", '"')
        return re.findall(r'\{"murl":"(https?:[^"]+?)"[^\}]*?"t":"([^"]*?)"', html)
    except Exception:
        return []


def collect_one(tick, brand, tok, dom) -> int:
    tok0 = tok.split(" ")[0].lower()
    d = LOGOS_DIR / f"{tick}_{brand}"
    d.mkdir(parents=True, exist_ok=True)
    hashes: set = set()
    dhashes: list = []
    count = 0
    print(f"=== [{tick}] {brand} ===", flush=True)

    p = d / f"{tick}_{brand}_clearbit.png"
    if try_add(f"https://logo.clearbit.com/{dom}?size=512", p, hashes, dhashes):
        count += 1
        print("  + clearbit", flush=True)

    for q in (f"{tok} logo png transparent", f"{tok} logo transparent background", f"{tok} logo"):
        if count >= K:
            break
        results = ddg_search(q)
        di = count + 1
        for r in results:
            if count >= K:
                break
            title = (r.get("title") or "").lower()
            if tok0 not in title:
                continue
            if not GOOD_RE.search(title):
                continue
            if BAD_RE.search(title):
                continue
            image_url = r.get("image") or ""
            if URL_BAD_RE.search(image_url.lower()):
                continue
            if not EXT_RE.search(image_url.split("?")[0]):
                continue
            ext = get_ext(image_url)
            p = d / f"{tick}_{brand}_ddg_{di:02d}.{ext}"
            if try_add(image_url, p, hashes, dhashes):
                count += 1
                di += 1
                print("  + ddg", flush=True)
        time.sleep(0.4)

    for first in (1, 36):
        if count >= K:
            break
        items = bing_search(f"{tok} logo", first)
        bi = count + 1
        for u, title in items:
            if count >= K:
                break
            u = u.replace("\\/", "/")
            title_l = title.lower()
            if tok0 not in title_l:
                continue
            if BAD_RE.search(title_l):
                continue
            if URL_BAD_RE.search(u.lower()):
                continue
            if not EXT_RE.search(u.split("?")[0]):
                continue
            ext = get_ext(u)
            p = d / f"{tick}_{brand}_bing_{bi:02d}.{ext}"
            if try_add(u, p, hashes, dhashes):
                count += 1
                bi += 1
                print("  + bing", flush=True)
        time.sleep(0.4)

    log(f"[{tick}] {brand} : {count} 장")
    print(f"  => {count} 장", flush=True)
    return count


def main():
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    for f in LOGOS_DIR.rglob("*"):
        if f.is_file():
            f.unlink()
    for d in sorted(LOGOS_DIR.glob("*"), reverse=True):
        if d.is_dir():
            d.rmdir()
    log(f"===== python 수집 시작(단일 로고 필터 + 화질 검사): {time.ctime()} =====")

    rows = load_data()
    print(f"총 {len(rows)}개 기업 수집 시작", flush=True)
    total = 0
    for tick, brand, tok, dom in rows:
        total += collect_one(tick, brand, tok, dom)
    log(f"완료: {time.ctime()}  총 {total} 장")
    print(f"\n완료! 총 {total} 장. 로그: {LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
