"""네이버 검색 API(뉴스) 클라이언트 — 실시간 뉴스 후보 수집.

발급: https://developers.naver.com/apps/#/register (검색 API 사용 설정)
.env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 설정 필요.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openapi.naver.com/v1/search/news.json"
_TAG_RE = re.compile(r"</?b>")


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text or "").replace("&quot;", '"').replace("&amp;", "&").strip()


def search_news(query: str, display: int = 100, start: int = 1, sort: str = "date") -> list[dict]:
    """네이버 뉴스 검색 API를 1회 호출해 실시간 결과를 반환한다.

    display: 1~100 (호출당 최대), start: 1~1000, sort: "date"(최신순) | "sim"(정확도순)
    """
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 설정되어 있지 않습니다. "
            "https://developers.naver.com/apps/#/register 에서 발급 후 .env에 추가하세요."
        )

    resp = requests.get(
        API_URL,
        headers={"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret},
        params={"query": query, "display": display, "start": start, "sort": sort},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    articles = []
    for item in items:
        try:
            pub_date: datetime = parsedate_to_datetime(item["pubDate"])
        except (KeyError, TypeError, ValueError):
            continue
        articles.append(
            {
                "title": _clean(item.get("title", "")),
                "description": _clean(item.get("description", "")),
                "link": item.get("link", ""),
                "originallink": item.get("originallink") or item.get("link", ""),
                "pub_date": pub_date,
            }
        )
    return articles


def search_news_paged(query: str, sort: str = "date", max_results: int = 300) -> list[dict]:
    """start를 이어가며 여러 페이지를 실시간으로 수집한다 (최대 max_results건, API 상한 1000)."""
    all_articles: list[dict] = []
    start = 1
    while len(all_articles) < max_results and start <= 1000:
        batch = search_news(query, display=100, start=start, sort=sort)
        if not batch:
            break
        all_articles.extend(batch)
        start += 100
    return all_articles[:max_results]