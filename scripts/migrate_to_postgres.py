"""로컬 데이터(SQLite RDB + 앱 DB + Chroma 벡터 + PDF)를 공유 DB(Supabase = Postgres + pgvector + Storage)로 옮긴다.

필요한 환경변수(.env - 채팅/저장소에 절대 올리지 않는다):
    DATABASE_URL          Supabase 대시보드 > Connect > 'Session pooler' 또는 'Direct connection' 문자열(비밀번호 포함)
    SUPABASE_URL          https://<project-ref>.supabase.co                (--storage 일 때만)
    SUPABASE_SERVICE_KEY  Project Settings > API > service_role 키          (--storage 일 때만, 서버 전용 비밀)

사용법:
    python scripts/migrate_to_postgres.py --all              # 스키마 + RDB + 앱 데이터 + 벡터 + PDF + RLS
    python scripts/migrate_to_postgres.py --schema --rdb --vectors
    python scripts/migrate_to_postgres.py --storage          # PDF 만 (이미 올라간 파일은 건너뜀)

안전장치: RDB/벡터는 원본에서 다시 만들 수 있는 파생 데이터라 비우고 다시 채운다. 앱 데이터(users/대화)는 대상이 비어 있을 때만
복사한다 - 이미 팀원들이 쓰고 있는 공유 DB 를 덮어쓰지 않는다. 여러 번 실행해도 안전하다.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

RDB_TABLES = ["brands", "categories", "manuals", "manual_models", "model_features", "feature_aliases", "error_entries", "error_codes"]
APP_TABLES = ["users", "user_appliances", "conversations", "messages"]     # 외래키 순서
ALL_TABLES = RDB_TABLES + ["rag_chunks"] + APP_TABLES
BATCH = 400


def connect():
    import psycopg
    from web import rdb
    if not rdb.use_postgres():
        sys.exit("DATABASE_URL 이 postgres 주소가 아닙니다 (.env 확인). 로컬 SQLite 를 건드리지 않으려고 중단합니다.")
    host = urlparse(rdb.pg_dsn()).hostname
    print(f"대상 DB: {host}")
    return psycopg.connect(rdb.pg_dsn(), autocommit=False, prepare_threshold=None)


def apply_schema(con) -> None:
    con.execute((ROOT / "supabase" / "schema.sql").read_text(encoding="utf-8"))
    con.commit()
    print("스키마 적용 완료")


def copy_rdb(con) -> None:
    src = sqlite3.connect(str(ROOT / "data" / "appliance.sqlite"))
    src.row_factory = sqlite3.Row
    con.execute("TRUNCATE " + ", ".join(reversed(RDB_TABLES)) + " CASCADE")
    for t in RDB_TABLES:
        rows = src.execute(f"SELECT * FROM {t}").fetchall()
        if not rows:
            continue
        cols = rows[0].keys()
        with con.cursor() as cur:
            cur.executemany(f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})", [tuple(r) for r in rows])
        print(f"  RDB {t}: {len(rows)}행")
    con.commit()


def copy_app(con) -> None:
    """users/user_appliances/conversations/messages - 대상이 비어 있을 때만. 테이블은 서버가 쓰는 SQLAlchemy 모델로 만든다."""
    from sqlalchemy import JSON, DateTime, text
    from web.db import Base, engine
    Base.metadata.create_all(engine)
    src = sqlite3.connect(str(ROOT / "data" / "app.sqlite"))
    src.row_factory = sqlite3.Row
    with engine.begin() as db:
        if db.execute(text("SELECT count(*) FROM users")).scalar() > 0:
            print("  앱 데이터: 대상에 이미 사용자가 있어 건너뜀 (공유 DB 덮어쓰기 방지)")
            return
        for name in APP_TABLES:
            table = Base.metadata.tables[name]
            rows = []
            for r in src.execute(f"SELECT * FROM {name}").fetchall():
                d = dict(r)
                for col in table.columns:
                    v = d.get(col.name)
                    if isinstance(col.type, JSON) and isinstance(v, str):
                        d[col.name] = json.loads(v)
                    elif isinstance(col.type, DateTime) and isinstance(v, str):
                        d[col.name] = datetime.fromisoformat(v)
                rows.append(d)
            if rows:
                db.execute(table.insert(), rows)
            if "id" in table.columns:                       # 시퀀스를 최대 id 뒤로 - 안 하면 다음 INSERT 가 id 충돌
                db.execute(text(f"SELECT setval(pg_get_serial_sequence('{name}', 'id'), COALESCE((SELECT MAX(id) FROM {name}), 1))"))
            print(f"  앱 {name}: {len(rows)}행")


def copy_vectors(con) -> None:
    import chromadb
    col = chromadb.PersistentClient(path=str(ROOT / os.getenv("CHROMA_DIR", "chroma_db"))).get_collection("appliance_manuals")
    total = col.count()
    con.execute("TRUNCATE rag_chunks")
    done, t0 = 0, time.time()
    while done < total:
        b = col.get(include=["documents", "metadatas", "embeddings"], limit=BATCH, offset=done)
        if not b["ids"]:
            break
        rows = [(i, m.get("product_model", ""), m.get("content_type"), d, json.dumps(m, ensure_ascii=False),
                 "[" + ",".join(repr(float(x)) for x in e) + "]", done + k)
                for k, (i, d, m, e) in enumerate(zip(b["ids"], b["documents"], b["metadatas"], b["embeddings"]))]
        with con.cursor() as cur:
            cur.executemany("INSERT INTO rag_chunks (id, product_model, content_type, text, metadata, embedding, seq) "
                            "VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector, %s)", rows)
        con.commit()
        done += len(rows)
        print(f"  벡터 {done}/{total} ({time.time() - t0:.0f}초)", flush=True)
    con.execute("ANALYZE rag_chunks")
    con.commit()


def upload_pdfs() -> None:
    url, key = os.getenv("SUPABASE_URL", "").rstrip("/"), os.getenv("SUPABASE_SERVICE_KEY", "")
    if not (url and key):
        sys.exit("--storage 는 SUPABASE_URL 과 SUPABASE_SERVICE_KEY 가 필요합니다 (.env)")
    bucket = os.getenv("SUPABASE_BUCKET_MANUALS", "manuals")
    from web.manual_pages import storage_headers
    auth = storage_headers(key)
    r = requests.post(f"{url}/storage/v1/bucket", headers=auth, json={"id": bucket, "name": bucket, "public": False}, timeout=30)
    print(f"  버킷 '{bucket}' (비공개): {'생성' if r.status_code == 200 else '이미 있음/응답 ' + str(r.status_code)}")
    files = sorted((ROOT / "data").glob("*/*/*.pdf"))
    sent = skipped = 0
    for i, f in enumerate(files, 1):
        brand, category = f.parent.parent.name, f.parent.name
        r = requests.post(f"{url}/storage/v1/object/{bucket}/{brand}/{category}/{f.name}",
                          headers={**auth, "Content-Type": "application/pdf", "x-upsert": "false"}, data=f.read_bytes(), timeout=600)
        if r.status_code == 200:
            sent += 1
        elif "exist" in r.text.lower() or "duplicate" in r.text.lower():
            skipped += 1
        else:
            print(f"  ! {f.name}: {r.status_code} {r.text[:120]}")
        if i % 10 == 0 or i == len(files):
            print(f"  PDF {i}/{len(files)} (업로드 {sent}, 이미 있음 {skipped})", flush=True)


def enable_rls(con) -> None:
    """Supabase 는 public 스키마 테이블을 anon 키로 API 에 노출한다. RLS 를 켜고 정책을 안 만들면 API 로는 아무것도 못 읽는다
    (users.password_hash 보호). 서버는 postgres 역할로 직접 접속해서 RLS 를 우회한다."""
    for t in ALL_TABLES:
        if con.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0]:
            con.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
    con.commit()
    print("RLS 활성화 완료 (API 직접 접근 차단)")


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ("schema", "rdb", "app", "vectors", "storage", "rls", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if not any(vars(a).values()):
        ap.print_help()
        return
    if a.all:
        a.schema = a.rdb = a.app = a.vectors = a.storage = a.rls = True
    if a.schema or a.rdb or a.app or a.vectors or a.rls:
        con = connect()
        if a.schema:
            apply_schema(con)
        if a.rdb:
            copy_rdb(con)
        if a.vectors:
            copy_vectors(con)
        if a.app:
            copy_app(con)
        if a.rls:
            enable_rls(con)
        con.close()
    if a.storage:
        upload_pdfs()
    print("완료")


if __name__ == "__main__":
    main()
