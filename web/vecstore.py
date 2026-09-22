"""Chroma 컬렉션 대체 - Supabase(Postgres+pgvector)의 rag_chunks 를 exp02 가 쓰는 Chroma 컬렉션과 같은 모양으로 감싼다.

exp02 가 Chroma 에 부르는 것은 세 가지뿐이다:
    collection.get(include=["metadatas"])                       (모든 청크의 메타데이터 - 알려진 모델명 목록)
    collection.query(query_embeddings, n_results, where)        (벡터 검색)
    collection.get(ids=[...], include=["documents","metadatas"])  (id 로 청크 조회 - Chroma 는 요청 순서가 아니라 컬렉션 내부 순번(seq) 순으로 돌려준다)
결과 모양(ids/documents/metadatas/distances, query 는 리스트의 리스트)과 거리 정의(Chroma 기본 = 제곱 L2)를 똑같이 맞춘다.
where 는 Chroma 문법의 일부만 지원한다: {"key": v}, {"key": {"$eq"|"$in": ...}}, {"$and": [...]}, {"$or": [...]}.
"""
from __future__ import annotations

import json

from . import rdb

_COLUMN_KEYS = {"product_model": "product_model", "content_type": "content_type"}   # 인덱스/컬럼으로 빼 둔 메타데이터 키


def _field(key: str) -> str:
    return _COLUMN_KEYS.get(key) or f"metadata->>'{key.replace(chr(39), '')}'"


def _where_sql(where: dict | None) -> tuple[str, list]:
    if not where:
        return "TRUE", []
    if "$and" in where or "$or" in where:
        op, items = ("AND", where["$and"]) if "$and" in where else ("OR", where["$or"])
        parts = [_where_sql(w) for w in items]
        return "(" + f" {op} ".join(p[0] for p in parts) + ")", [x for p in parts for x in p[1]]
    clauses, params = [], []
    for key, cond in where.items():
        f = _field(key)
        if isinstance(cond, dict):
            (op, val), = cond.items()
            if op == "$eq":
                clauses.append(f"{f} = %s"); params.append(str(val))
            elif op == "$in":
                clauses.append(f"{f} = ANY(%s)"); params.append([str(v) for v in val])
            else:
                raise ValueError(f"지원하지 않는 where 연산자: {op}")
        else:
            clauses.append(f"{f} = %s"); params.append(str(cond))
    return "(" + " AND ".join(clauses) + ")", params


def _vec(embedding) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class PgCollection:
    name = "appliance_manuals"
    metadata = None

    def count(self) -> int:
        return rdb.one("SELECT count(*) AS n FROM rag_chunks")["n"]

    def get(self, ids: list[str] | None = None, where: dict | None = None, include: list[str] | None = None,
            limit: int | None = None, offset: int | None = None) -> dict:
        include = include or ["metadatas", "documents"]
        w, params = _where_sql(where)
        if ids is not None:
            w, params = f"({w}) AND id = ANY(%s)", params + [list(ids)]
        cols = "id" + (", text" if "documents" in include else "") + (", metadata" if "metadatas" in include else "") \
            + (", embedding::text AS embedding" if "embeddings" in include else "")
        sql = f"SELECT {cols} FROM rag_chunks WHERE {w} ORDER BY seq, id" + (f" LIMIT {int(limit)}" if limit else "") + (f" OFFSET {int(offset)}" if offset else "")
        with rdb.pool().connection() as con:
            rows = con.execute(sql, params).fetchall()
        out: dict = {"ids": [r["id"] for r in rows]}
        if "documents" in include:
            out["documents"] = [r["text"] for r in rows]
        if "metadatas" in include:
            out["metadatas"] = [r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"]) for r in rows]
        if "embeddings" in include:
            out["embeddings"] = [json.loads(r["embedding"]) for r in rows]
        return out

    def query(self, query_embeddings: list, n_results: int = 10, where: dict | None = None, include: list[str] | None = None) -> dict:
        w, wparams = _where_sql(where)
        ids, docs, metas, dists = [], [], [], []
        with rdb.pool().connection() as con:
            for emb in query_embeddings:
                q = _vec(emb)
                rows = con.execute(
                    f"SELECT id, text, metadata, power(embedding <-> %s::vector, 2) AS distance FROM rag_chunks "
                    f"WHERE {w} ORDER BY embedding <-> %s::vector LIMIT %s", [q] + wparams + [q, int(n_results)]).fetchall()
                ids.append([r["id"] for r in rows]); docs.append([r["text"] for r in rows])
                metas.append([r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"]) for r in rows])
                dists.append([float(r["distance"]) for r in rows])
        return {"ids": ids, "documents": docs, "metadatas": metas, "distances": dists}
