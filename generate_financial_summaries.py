"""KOSPI200_output/kospi200_financials/*.md를 GPT로 요약해 Supabase financial_summaries 테이블에 적재한다.

financial_chunks는 티커당 여러 행(statement_type x period_type)으로 쪼개져 있어 요약 컬럼을 넣기
맞지 않으므로, 원본 마크다운(손익계산서+대차대조표+현금흐름표 전체)을 통째로 읽어 티커당 하나의
요약을 만들고 별도 테이블에 적재한다.

사전 작업:
    1. Supabase SQL Editor에서 sql/schema.sql 실행 (financial_summaries 테이블 추가분 포함)
    2. .env에 SUPABASE_URL / SUPABASE_SERVICE_KEY / OPENAI_API_KEY 설정

Usage:
    python generate_financial_summaries.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from generation import GPTGenerator

FINANCIALS_DIR = Path("KOSPI200_output/kospi200_financials")
FILENAME_RE = re.compile(r"^(.+)_financials\.md$")

INSTRUCTION = (
    "위 재무제표 원문(손익계산서/대차대조표/현금흐름표, 연간+분기)을 바탕으로 투자자가 참고할 만한 "
    "핵심 요약을 한국어 불릿 5~8줄로 작성하세요. 매출/이익 성장 추세, 수익성(마진) 변화, 재무 "
    "건전성(부채/자기자본), 현금흐름(영업/투자/재무, FCF) 특징, 종합 투자 시사점(강점/리스크)을 "
    "포함하세요. 컨텍스트에 없는 수치는 언급하지 마세요."
)


def main():
    load_dotenv()
    generator = GPTGenerator()
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    paths = sorted(FINANCIALS_DIR.glob("*_financials.md"))
    for i, path in enumerate(paths, 1):
        ticker = FILENAME_RE.match(path.name).group(1)
        content = path.read_text(encoding="utf-8")
        summary = generator.generate(INSTRUCTION, content)
        client.table("financial_summaries").upsert({"ticker": ticker, "summary": summary}).execute()
        print(f"[{i}/{len(paths)}] {ticker} 요약 완료")

    print(f"완료: {len(paths)}개 재무제표 요약 -> Supabase.financial_summaries")


if __name__ == "__main__":
    main()
