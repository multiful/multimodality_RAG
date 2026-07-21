# 로고 이미지 수집기

NASDAQ-100 로고 데이터셋(`logos/`) 수집 스크립트. 종목당 20장 목표.

- 소스: Clearbit 공식 로고 → DuckDuckGo → Bing (제목에 브랜드명 필수, 위키류 제외)
- 중복 제거: MD5(완전 동일) + dHash 지각해시(해밍거리 ≤2, 리사이즈본 제거)
- 필터: 로고 모음/변천사/배너류 제목 패턴 차단, HTML 응답·3KB 미만 파일 폐기

## 실행 (Windows)

`run_collect.bat` 더블클릭 — 저장소 루트에 `logos/티커_브랜드/` 폴더와 `collect_log.txt`가 생성된다.

주의: 실행 시 기존 `logos/` 내용을 전부 지우고 클린 스타트한다.
