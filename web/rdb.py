"""RDB(매뉴얼·에러코드·기능 표) 조회 - 기본은 data/appliance.sqlite, DATABASE_URL 이 postgres 면 Supabase(Postgres)를 읽는다.

SQL 은 SQLite 문법(? 자리표시자)으로 한 번만 쓰고, 모델 패턴 매칭만 {MODEL_MATCH} 자리표시자로 쓴다:
    SELECT has FROM model_features WHERE ... AND {MODEL_MATCH}   -> params 마지막에 모델명
Postgres 에서는 자리표시자를 %s 로, GLOB 를 LIKE 로 바꿔서 실행한다. 결과는 항상 dict 목록.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQLITE_PATH = ROOT / "data" / "appliance.sqlite"


def database_url() -> str:
    return os.getenv("DATABASE_URL", "")


def use_postgres() -> bool:
    return database_url().startswith(("postgres://", "postgresql://", "postgresql+"))


def pg_dsn() -> str:
    """SQLAlchemy 식 'postgresql+psycopg://' 도, Supabase 가 주는 'postgres://' 도 psycopg 가 읽는 형태로."""
    url = database_url()
    url = re.sub(r"^postgresql\+\w+://", "postgresql://", url)
    return re.sub(r"^postgres://", "postgresql://", url)


_MODEL_MATCH = {
    "sqlite": "? GLOB replace(model_pattern,'*','?')",
    "pg": "%s LIKE replace(model_pattern,'*','%%')",
}
_QMARK_OUTSIDE_QUOTES = re.compile(r"\?(?=(?:[^']*'[^']*')*[^']*$)")

_pool = None
_pool_lock = threading.Lock()
_sqlite_local = threading.local()


def pool():
    """Postgres 연결 풀(읽기 전용, autocommit). Supabase 의 pgbouncer(transaction 모드)와 호환되게 prepared statement 는 끈다."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
                _pool = ConnectionPool(pg_dsn(), min_size=1, max_size=6, open=True,
                                       kwargs={"row_factory": dict_row, "autocommit": True, "prepare_threshold": None})
    return _pool


def _sqlite():
    con = getattr(_sqlite_local, "con", None)
    if con is None:
        con = sqlite3.connect(str(SQLITE_PATH))
        con.row_factory = sqlite3.Row
        _sqlite_local.con = con
    return con


def rows(sql: str, params=()) -> list[dict]:
    if use_postgres():
        q = _QMARK_OUTSIDE_QUOTES.sub("%s", sql.replace("{MODEL_MATCH}", _MODEL_MATCH["pg"]))
        with pool().connection() as con:
            return [dict(r) for r in con.execute(q, tuple(params)).fetchall()]
    q = sql.replace("{MODEL_MATCH}", _MODEL_MATCH["sqlite"])
    return [dict(r) for r in _sqlite().execute(q, tuple(params)).fetchall()]


def one(sql: str, params=()) -> dict | None:
    r = rows(sql, params)
    return r[0] if r else None


def feature_vocab(category: str, brand: str, db_path=None) -> list[str]:
    """siyeon.rag.understand.feature_vocab 과 같은 결과(그 브랜드·제품군 기능 표의 기능 이름) - Postgres 모드에서 그 함수를 대체한다.
    정렬은 파이썬에서(코드포인트 순 = SQLite BINARY) - Postgres 의 로케일 정렬은 한글 순서가 달라서 SQL ORDER BY 를 쓰지 않는다."""
    return sorted(r["feature"] for r in rows("SELECT DISTINCT feature FROM model_features WHERE category=? AND brand=?", (category, brand)))
