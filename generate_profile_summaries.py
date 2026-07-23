"""KOSPI200_output/kospi200_profiles/*.md를 GPT로 요약해 Supabase company_profile_chunks.summary 컬럼에 채운다.

사전 작업:
    1. Supabase SQL Editor에서 sql/schema.sql 실행 (summary 컬럼 추가분 포함)
    2. .env에 SUPABASE_URL / SUPABASE_SERVICE_KEY / OPENAI_API_KEY 설정
    3. build_index_profiles.py로 company_profile_chunks에 행이 이미 적재되어 있어야 함 (update 대상)

Usage:
    python generate_profile_summaries.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from generation import GPTGenerator

PROFILE_DIR = Path("KOSPI200_output/kospi200_profiles")
FILENAME_RE = re.compile(r"^(.+)_profile\.md$")

INSTRUCTION = (
    "위 기업 프로필(섹터, 사업 개요, 주요 임원)을 바탕으로 투자자가 참고할 만한 핵심 인사이트를 "
    "한국어 불릿 5~8줄로 요약하세요. 사업 모델과 핵심 경쟁력, 소속 산업/섹터 특징, 경영진 관련 "
    "특이사항(있는 경우), 종합 투자 시사점을 포함하세요. 컨텍스트에 없는 내용은 추측하지 마세요."
)


def main():
    load_dotenv()
    generator = GPTGenerator()
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    paths = sorted(PROFILE_DIR.glob("*_profile.md"))
    for i, path in enumerate(paths, 1):
        ticker = FILENAME_RE.match(path.name).group(1)
        content = path.read_text(encoding="utf-8")
        summary = generator.generate(INSTRUCTION, content)
        client.table("company_profile_chunks").update({"summary": summary}).eq("id", ticker).execute()
        print(f"[{i}/{len(paths)}] {ticker} 요약 완료")

    print(f"완료: {len(paths)}개 프로필 요약 -> Supabase.company_profile_chunks.summary")


if __name__ == "__main__":
    main()
