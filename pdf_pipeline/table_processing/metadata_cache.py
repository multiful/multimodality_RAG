"""[19] Redis 캐싱 인터페이스 — 사용자 요청: 추출한 메타데이터를 Redis에 캐싱.

로컬에 Redis 서버가 없어 검증 단계에서는 fakeredis(순수 파이썬 인메모리 목업, redis-py와 동일
API)를 쓴다 — get_client(use_fake=False)로 바꾸면 실제 Redis 서버에 그대로 연결되는 코드다.
테이블뿐 아니라 이미지/텍스트에서 뽑힐 메타데이터도 같은 키 스키마를 공유하도록 설계
(source 필드로 table/image/text 구분) — 팀원 파트 연결 시 그대로 재사용 가능.

키 스키마:
  pdf:{pdf_id}:metadata               -> 이 PDF에서 추출된 모든 구조화 레코드(JSON 배열, 전체)
  pdf:{pdf_id}:field:{canonical_key}  -> 특정 canonical field로 색인된 레코드만(빠른 조회용)
"""

import json


def get_client(use_fake: bool = True):
    if use_fake:
        import fakeredis
        return fakeredis.FakeStrictRedis(decode_responses=True)
    import redis
    return redis.Redis(host="localhost", port=6379, decode_responses=True)


def cache_metadata(client, pdf_id: str, records: list):
    """records: [{"source": "table"|"image"|"text", "canonical_field": str|None,
                  "raw_label": str, "value": ..., "page": int, ...}, ...]"""
    full_key = f"pdf:{pdf_id}:metadata"
    existing = json.loads(client.get(full_key) or "[]")
    existing.extend(records)
    client.set(full_key, json.dumps(existing, ensure_ascii=False))

    for rec in records:
        cf = rec.get("canonical_field")
        if not cf:
            continue
        field_key = f"pdf:{pdf_id}:field:{cf}"
        existing_field = json.loads(client.get(field_key) or "[]")
        existing_field.append(rec)
        client.set(field_key, json.dumps(existing_field, ensure_ascii=False))


def get_all_metadata(client, pdf_id: str) -> list:
    return json.loads(client.get(f"pdf:{pdf_id}:metadata") or "[]")


def get_field(client, pdf_id: str, canonical_key: str) -> list:
    return json.loads(client.get(f"pdf:{pdf_id}:field:{canonical_key}") or "[]")
