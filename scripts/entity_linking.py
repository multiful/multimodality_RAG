"""Entity linking for NASDAQ-100 logo recognition (PRD 3.4 / 5.1).

Maps free-text VLM output to canonical tickers via a word-boundary alias table.
Alias table is derived from the logos/ folder names plus manual brand aliases.
"""

import re

# ticker -> aliases (lowercase). Matched with word boundaries on normalized text.
ALIASES = {
    "AAPL": ["apple"],
    "ABNB": ["airbnb"],
    "ADBE": ["adobe"],
    "ADI": ["analog devices"],
    "ADP": ["adp", "automatic data processing"],
    "ADSK": ["autodesk"],
    "AEP": ["american electric power"],
    "ALAB": ["astera labs", "astera"],
    "ALNY": ["alnylam"],
    "AMAT": ["applied materials"],
    "AMD": ["amd", "advanced micro devices"],
    "AMGN": ["amgen"],
    "AMZN": ["amazon"],
    "APP": ["applovin"],
    "ARM": ["arm", "arm holdings"],
    "ASML": ["asml"],
    "AVGO": ["broadcom"],
    "AXON": ["axon", "taser"],
    "BKNG": ["booking", "booking.com", "priceline"],
    "BKR": ["baker hughes"],
    "CCEP": ["coca-cola", "coca cola", "coke"],
    "CDNS": ["cadence"],
    "CEG": ["constellation energy", "constellation"],
    "CMCSA": ["comcast", "xfinity"],
    "COST": ["costco"],
    "CPRT": ["copart"],
    "CRWD": ["crowdstrike"],
    "CRWV": ["coreweave"],
    "CSCO": ["cisco"],
    "CSX": ["csx"],
    "CTAS": ["cintas"],
    "DASH": ["doordash"],
    "DDOG": ["datadog"],
    "DXCM": ["dexcom"],
    "EA": ["ea", "electronic arts", "ea sports"],
    "EXC": ["exelon"],
    "FANG": ["diamondback"],
    "FAST": ["fastenal"],
    "FER": ["ferrovial"],
    "FTNT": ["fortinet"],
    "GEHC": ["ge healthcare", "general electric healthcare"],
    "GILD": ["gilead"],
    "GOOGL": ["google", "alphabet", "youtube", "android"],
    "HON": ["honeywell"],
    "IDXX": ["idexx"],
    "INTC": ["intel"],
    "INTU": ["intuit", "turbotax", "quickbooks"],
    "ISRG": ["intuitive surgical", "da vinci"],
    "KDP": ["keurig", "dr pepper", "keurig dr pepper"],
    "KHC": ["kraft", "heinz", "kraft heinz"],
    "KLAC": ["kla", "kla-tencor", "kla tencor"],
    "LIN": ["linde"],
    "LITE": ["lumentum"],
    "LRCX": ["lam research"],
    "MAR": ["marriott"],
    "MCHP": ["microchip", "microchip technology"],
    "MDLZ": ["mondelez"],
    "MELI": ["mercadolibre", "mercado libre"],
    "META": ["meta", "facebook", "instagram"],
    "MNST": ["monster", "monster energy", "monster beverage"],
    "MPWR": ["monolithic power", "mps"],
    "MRVL": ["marvell"],
    "MSFT": ["microsoft", "windows", "xbox"],
    "MSTR": ["microstrategy", "strategy"],
    "MU": ["micron"],
    "NBIS": ["nebius"],
    "NFLX": ["netflix"],
    "NVDA": ["nvidia"],
    "NXPI": ["nxp"],
    "ODFL": ["old dominion"],
    "ORLY": ["oreilly", "o'reilly", "o reilly"],
    "PANW": ["palo alto networks", "palo alto"],
    "PAYX": ["paychex"],
    "PCAR": ["paccar", "kenworth", "peterbilt"],
    "PDD": ["pdd", "temu", "pinduoduo"],
    "PEP": ["pepsico", "pepsi"],
    "PLTR": ["palantir"],
    "PYPL": ["paypal"],
    "QCOM": ["qualcomm", "snapdragon"],
    "REGN": ["regeneron"],
    "RKLB": ["rocket lab"],
    "ROP": ["roper"],
    "ROST": ["ross", "ross stores", "ross dress for less"],
    "SBUX": ["starbucks"],
    "SHOP": ["shopify"],
    "SNDK": ["sandisk"],
    "SNPS": ["synopsys"],
    "SPCX": ["spacex", "space x", "starlink"],
    "STX": ["seagate"],
    "TER": ["teradyne"],
    "TMUS": ["t-mobile", "tmobile", "t mobile"],
    "TRI": ["thomson reuters", "reuters"],
    "TSLA": ["tesla"],
    "TTWO": ["take-two", "take two", "taketwo", "rockstar games", "2k"],
    "TXN": ["texas instruments"],
    "VRTX": ["vertex", "vertex pharmaceuticals"],
    "WBD": ["warner bros", "warner brothers", "warner bros. discovery"],
    "WDAY": ["workday"],
    "WDC": ["western digital"],
    "WMT": ["walmart"],
    "XEL": ["xcel energy", "xcel"],
}

# Every ticker is also its own alias (VLMs sometimes answer with the ticker).
# Skip ambiguous/short tickers that collide with common words.
_TICKER_ALIAS_SKIP = {"APP", "ARM", "FAST", "COST", "META", "EA", "MU", "FANG", "DASH", "LITE"}

_PATTERNS = []
for _t, _als in ALIASES.items():
    als = set(_als)
    if _t not in _TICKER_ALIAS_SKIP:
        als.add(_t.lower())
    for _a in als:
        _PATTERNS.append((re.compile(r"(?<![a-z0-9])" + re.escape(_a) + r"(?![a-z0-9])"), _t, len(_a)))
# Longer aliases first so "palo alto networks" wins over "palo alto" at same pos.
_PATTERNS.sort(key=lambda x: -x[2])


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[\*_`#|:;()\[\]{}\"<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def link(text: str, top_k: int = 3) -> list[str]:
    """Return ranked ticker candidates found in free text (earliest mention first)."""
    norm = normalize(text)
    hits: dict[str, int] = {}
    for pat, ticker, _ in _PATTERNS:
        m = pat.search(norm)
        if m and (ticker not in hits or m.start() < hits[ticker]):
            hits.setdefault(ticker, m.start())
    ranked = sorted(hits, key=lambda t: hits[t])
    return ranked[:top_k]


if __name__ == "__main__":
    tests = [
        ("1. Apple 2. Samsung 3. LG", ["AAPL"]),
        ("This logo belongs to NVIDIA Corporation.", ["NVDA"]),
        ("Google, Alphabet, Microsoft", ["GOOGL", "MSFT"]),
        ("The company is Coca-Cola Europacific Partners", ["CCEP"]),
        ("T-Mobile US, Deutsche Telekom", ["TMUS"]),
        ("An armadillo with a fast metabolism", []),
        ("Palo Alto Networks", ["PANW"]),
        ("intuitive surgical", ["ISRG"]),
        ("Intuit", ["INTU"]),
    ]
    for text, expect in tests:
        got = link(text)
        status = "OK " if got[: len(expect)] == expect else "FAIL"
        print(f"{status} {text!r} -> {got}")
