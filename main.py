"""Smoke test: confirm both embedders load and produce sane vectors before building the full index.

Usage:
    python main.py
"""

from dotenv import load_dotenv

from embeddings import BGEEmbedder, BGEKoEmbedder, GPTEmbedder


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def main():
    load_dotenv()

    related = "애플의 최근 분기 매출과 영업이익은 얼마야?"
    unrelated = "테슬라 관련 최근 뉴스 알려줘"

    for embedder in (BGEEmbedder(), BGEKoEmbedder(), GPTEmbedder()):
        vecs = embedder.embed([related, unrelated])
        sim = cosine(vecs[0], vecs[1])
        print(f"{embedder.name}: dim={len(vecs[0])}, sample_cosine={sim:.4f}")


if __name__ == "__main__":
    main()
