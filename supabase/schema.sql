-- ask-my-appliance 공유 DB 스키마 (Supabase / Postgres + pgvector)
-- 실행: python scripts/migrate_to_postgres.py  (이 파일을 먼저 적용한다. 여러 번 실행해도 안전)
--
-- 구성
--   1) RDB  - 매뉴얼/모델/기능 표/에러코드   (원본: data/appliance.sqlite, 읽기 전용 참조 데이터)
--   2) 벡터 - rag_chunks                     (원본: chroma_db 컬렉션 appliance_manuals, 6105청크 x 1024차원)
--   3) 앱   - users/user_appliances/conversations/messages 는 서버가 시작할 때 SQLAlchemy 가 만든다(web/db.py)

create extension if not exists vector;

-- ── 1) RDB ────────────────────────────────────────────────────────────────
create table if not exists brands (
    brand    text primary key,          -- 'lg' | 'samsung'
    name_ko  text not null
);

create table if not exists categories (
    category text primary key,          -- 'aircon' | 'fridge' | 'washer'
    name_ko  text not null
);

create table if not exists manuals (
    manual_id   text primary key,       -- 'AC_FQ18GC1EHN' (= PDF 파일명 stem = 청크 doc_id)
    brand       text not null references brands,
    category    text not null references categories,
    file_path   text not null,
    md5         text not null,
    page_count  integer,
    has_feature_table integer not null default 0,
    product_type text
);
create index if not exists ix_manuals_md5 on manuals (md5);

create table if not exists manual_models (
    manual_id     text not null references manuals,
    model_pattern text not null,
    source        text not null check (source in ('filename', 'appendix_table', 'manual')),
    primary key (manual_id, model_pattern)
);

create table if not exists model_features (
    brand         text not null references brands,
    category      text not null references categories,
    model_pattern text not null,
    feature       text not null,
    has           integer not null check (has in (0, 1)),
    manual_id     text not null references manuals,
    primary key (brand, category, model_pattern, feature)
);

create table if not exists feature_aliases (
    feature  text not null,
    alias    text not null,
    primary key (feature, alias)
);

create table if not exists error_entries (
    entry_id    integer primary key,
    brand       text not null references brands,
    category    text not null references categories,
    title       text not null,
    summary     text,
    content     text not null,
    is_code     integer not null check (is_code in (0, 1)),
    source_file text not null,
    chunk_id    text
);
create index if not exists ix_error_entries_scope on error_entries (brand, category);

create table if not exists error_codes (
    brand        text not null references brands,
    category     text not null references categories,
    code_norm    text not null,
    code_display text not null,
    kind         text not null check (kind in ('primary', 'variant')),
    entry_id     integer not null references error_entries,
    primary key (brand, category, code_norm, entry_id)
);
create index if not exists ix_error_codes_lookup on error_codes (brand, category, code_norm);

-- ── 2) 벡터 ───────────────────────────────────────────────────────────────
-- 검색은 항상 product_model 로 좁힌 뒤(모델당 ~100청크) 정확 검색(순차 스캔)한다 - 근사 인덱스(HNSW)를 안 써서
-- Chroma 와 결과가 똑같고, 필터가 걸린 HNSW 검색이 후보를 덜 돌려주는 pgvector 의 알려진 문제도 없다.
create table if not exists rag_chunks (
    id            text primary key,
    product_model text not null,
    content_type  text,
    text          text not null,
    metadata      jsonb not null,       -- Chroma 메타데이터 전체(brand, category, section_title, page, ...)
    embedding     vector(1024) not null,
    seq           integer not null default 0   -- Chroma 컬렉션의 내부 순번 - get(ids) 가 이 순서로 돌려줘서 리랭킹 후보 순서가 같아야 결과가 같다
);
alter table rag_chunks add column if not exists seq integer not null default 0;
create index if not exists ix_rag_chunks_model on rag_chunks (product_model);
