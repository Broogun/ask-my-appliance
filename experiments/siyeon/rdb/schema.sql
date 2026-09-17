-- ============================================================================
-- ask-my-appliance 구조화 데이터 스키마 (SQLite)
--
-- 벡터 검색(Chroma)이 맡지 않는 것들을 RDB로:
--   1) 에러코드 조회      — "CH05 떠요"는 키 조회 문제. 임베딩+리랭커(3.6초) 대신 ms 단위, 정확도 100%
--   2) 모델 ↔ 매뉴얼 매핑  — 파일명 모델과 매뉴얼 안 모델 패턴이 다른 경우가 있음 (AC_FQ18GC1EHN.pdf 표엔 GC2/GG1만)
--   3) 모델별 기능 유무    — 점수(임베딩 거리·리랭커 확률)로는 못 가르던 "이 모델엔 클린봇 없음"을 룰로 판정
--   4) 등록 가전          — 웹에서 사용자가 등록한 가전 → 검색 필터(doc_id)와 에러코드 조회 범위(brand, category)의 출처
--
-- DB 파일: data/appliance.sqlite (data/는 git 제외. build_db.py가 JSON+PDF에서 매번 재생성)
-- 브랜드/제품군 코드는 data/ 폴더명과 동일: brand ∈ {lg, samsung}, category ∈ {aircon, fridge, washer}
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- 0. 코드표 (작지만 오타 방지용 FK 대상)
-- ----------------------------------------------------------------------------
CREATE TABLE brands (
    brand    TEXT PRIMARY KEY,          -- 'lg' | 'samsung'
    name_ko  TEXT NOT NULL              -- 'LG' | '삼성'
);

CREATE TABLE categories (
    category TEXT PRIMARY KEY,          -- 'aircon' | 'fridge' | 'washer'
    name_ko  TEXT NOT NULL              -- '에어컨' | '냉장고' | '세탁기'
);

-- ----------------------------------------------------------------------------
-- 1. 매뉴얼 (PDF 1개 = 1행). manual_id는 벡터 인덱스 청크의 doc_id와 같은 값.
-- ----------------------------------------------------------------------------
CREATE TABLE manuals (
    manual_id   TEXT PRIMARY KEY,       -- 'AC_FQ18GC1EHN'  (= 파일명 stem = 청크 doc_id)
    brand       TEXT NOT NULL REFERENCES brands,
    category    TEXT NOT NULL REFERENCES categories,
    file_path   TEXT NOT NULL,          -- 'data/lg/aircon/AC_FQ18GC1EHN.pdf'
    md5         TEXT NOT NULL,          -- 같은 md5 = 같은 매뉴얼 (FQ18GN7BKN == FQ25GN9BKN, SQ07GA3WBN == SQ07GJ1WEN)
    page_count  INTEGER,
    has_feature_table INTEGER NOT NULL DEFAULT 0,  -- 부록 '모델별 기능 확인하기' 표 유무 (SQ 벽걸이형은 없음)
    product_type TEXT                   -- 평가 체계(pipeline.evaluate.MODEL_LIST) 8분류: 에어컨/냉장고/김치냉장고/세탁기/건조기/세탁건조기/스타일러
                                        -- category(폴더 3분류)는 에러코드 JSON·RDB 조회 범위 기준, product_type은 표시·보고서 기준
);
CREATE INDEX ix_manuals_md5 ON manuals (md5);

-- 모델 ↔ 매뉴얼 (N:1). 한 매뉴얼이 모델 패밀리 여러 개를 커버한다.
-- model_pattern 표기: 정확한 모델명('FQ18GC1EHN') 또는 LG 와일드카드('FQ**GC2***', * = 정확히 한 글자).
-- 조회 시 SQLite GLOB 사용: replace(model_pattern, '*', '?') → '?'가 한 글자 와일드카드.
CREATE TABLE manual_models (
    manual_id     TEXT NOT NULL REFERENCES manuals,
    model_pattern TEXT NOT NULL,
    source        TEXT NOT NULL CHECK (source IN ('filename', 'appendix_table', 'manual')),
                                        -- filename: 파일명에서 / appendix_table: 부록 표에서 / manual: 사람이 직접 추가
    PRIMARY KEY (manual_id, model_pattern)
);

-- ----------------------------------------------------------------------------
-- 2. 모델별 기능 유무 (부록 '모델별 기능 확인하기' 표)
--    행 없음 = "모름"(검색으로 폴백), has=0 = "없음"(검색 생략하고 없다고 답함), has=1 = "있음"
-- ----------------------------------------------------------------------------
CREATE TABLE model_features (
    brand         TEXT NOT NULL REFERENCES brands,
    category      TEXT NOT NULL REFERENCES categories,
    model_pattern TEXT NOT NULL,        -- 'FQ**FN9***' (표의 첫 열 그대로. 'FQ**FN9*** FQ**FN7***'처럼 두 개면 두 행으로 분리)
    feature       TEXT NOT NULL,        -- 표 헤더 정규화: '클린봇', 'UVnano', '공기청정', '음성인식', '활동부재', '좌우 토출구 덮개', ...
    has           INTEGER NOT NULL CHECK (has IN (0, 1)),   -- O → 1, X → 0
    manual_id     TEXT NOT NULL REFERENCES manuals,          -- 어느 매뉴얼 표에서 왔는지 (출처 추적)
    PRIMARY KEY (brand, category, model_pattern, feature)
);

-- 질문에서 기능명을 잡기 위한 별칭. '자동 청소' → 클린봇, 'UV 살균' → UVnano
CREATE TABLE feature_aliases (
    feature  TEXT NOT NULL,
    alias    TEXT NOT NULL,             -- 소문자·공백 제거 후 비교 ('자동청소', 'uv살균')
    PRIMARY KEY (feature, alias)
);

-- ----------------------------------------------------------------------------
-- 3. 에러코드
--    error_entries: JSON sections 1건 = 1행 (제목·본문). 벡터 인덱스에도 같은 내용이 청크로 들어가 있으므로 chunk_id로 연결.
--    error_codes  : 항목 하나에 코드 여러 개 (CH05 / E0 / CH53). 조회 키는 여기.
-- ----------------------------------------------------------------------------
CREATE TABLE error_entries (
    entry_id    INTEGER PRIMARY KEY,
    brand       TEXT NOT NULL REFERENCES brands,
    category    TEXT NOT NULL REFERENCES categories,
    title       TEXT NOT NULL,          -- 원문: 'CH05 / E0 / CH53 (통신 에러)'
    summary     TEXT,                   -- 괄호 안: '통신 에러'
    content     TEXT NOT NULL,          -- 원인·조치 본문
    is_code     INTEGER NOT NULL CHECK (is_code IN (0, 1)),
                                        -- 0 = 코드가 아닌 항목 ('딩동딩동 경보음', '온도 표시 깜빡임'). 코드 조회 대상 아님, 벡터 검색으로만
    source_file TEXT NOT NULL,          -- 'data/lg_aircon_errors.json'
    chunk_id    TEXT                    -- 벡터 인덱스 청크 id ('LG-AC-COMMON_에러코드_CH05_..._7dd05894'). 없으면 NULL
);
CREATE INDEX ix_error_entries_scope ON error_entries (brand, category);

CREATE TABLE error_codes (
    brand        TEXT NOT NULL REFERENCES brands,
    category     TEXT NOT NULL REFERENCES categories,
    code_norm    TEXT NOT NULL,         -- 정규화 키 (아래 규칙). 'CH05'→'CH5', 'Er(E) FF'→'ERFF'와 'EFF'와 'FF' 세 행
    code_display TEXT NOT NULL,         -- 사람이 보는 표기 'CH05'
    kind         TEXT NOT NULL CHECK (kind IN ('primary', 'variant')),
                                        -- primary: 제목 앞부분의 코드 / variant: 괄호 안 '제품에 따라 P4/P6/P7/P8/CH34' 같은 부수 언급
    entry_id     INTEGER NOT NULL REFERENCES error_entries,
    PRIMARY KEY (brand, category, code_norm, entry_id)
    -- 같은 코드가 항목 둘에 걸칠 수 있음 (LG 에어컨 F4: CH38 냉매 부족 + CH90/91 시운전 보호) → 둘 다 반환
);
CREATE INDEX ix_error_codes_lookup ON error_codes (brand, category, code_norm);

-- code_norm 정규화 규칙 (build_db.py 와 질문 파서가 같은 함수를 써야 함):
--   1) 대문자, 영숫자 외 제거          'ch 05' → 'CH05',  'Er(E) FF' → 괄호 대안 전개 후 'ERFF', 'EFF'
--   2) 숫자부 앞 0 제거                'CH05' → 'CH5',  'C101' → 'C101',  'E0' → 'E0' (0 하나는 유지)
--   3) LG 냉장고 'Er FF'/'E FF' 류는 접두어 뗀 'FF'도 별칭으로 추가 (사용자는 보통 'FF 떠요'라고 침)
--   4) 조회 시 혼동 문자 변형도 함께 조회: O↔0, I↔1, S↔5, B↔8  ('0E' 입력 → 'OE'도 시도)
--   ※ 정규화 키만으로는 브랜드·제품군을 넘나들며 충돌함 (FF: LG 세탁기 '동결' vs LG 냉장고 '팬 모터')
--     → 조회는 반드시 등록 가전의 (brand, category)로 범위를 좁힌다.

-- ----------------------------------------------------------------------------
-- 4. 등록 가전 (웹). 검색 필터와 에러코드 조회 범위의 출처.
--    users 테이블은 웹 프레임워크(장고/FastAPI 등)가 소유 — 여기선 user_id만 참조.
-- ----------------------------------------------------------------------------
CREATE TABLE user_appliances (
    appliance_id INTEGER PRIMARY KEY,
    user_id      TEXT NOT NULL,
    brand        TEXT NOT NULL REFERENCES brands,
    category     TEXT NOT NULL REFERENCES categories,
    model        TEXT NOT NULL,         -- 사용자 입력 모델명, 대문자·공백 제거 저장 ('FQ18GC1EHN')
    nickname     TEXT,                  -- '거실 에어컨'
    manual_id    TEXT REFERENCES manuals,
                                        -- 등록 시점에 manual_models로 해석한 결과. NULL = 매뉴얼 없음
                                        -- (이때 챗봇은 brand+category 공통 문서(에러코드)만으로 답하고 그 사실을 알림)
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX ix_user_appliances_user ON user_appliances (user_id);

-- ----------------------------------------------------------------------------
-- 5. (선택) 청크 메타데이터 미러 + 전문검색
--    벡터 인덱스의 청크를 RDB에도 두면 (a) 답변에 "어느 매뉴얼 몇 페이지" 출처 표시 (b) FTS5로 제목·증상 첫 줄만 대상으로 한
--    필드 제한 키워드 검색(BM25 내장) 이 가능. 3단계 실험(필드 제한 BM25)에서 쓸 예정. 지금은 만들지 않아도 됨.
-- ----------------------------------------------------------------------------
CREATE TABLE chunks (
    chunk_id    TEXT PRIMARY KEY,       -- 청크 id (md5 기반, 재실행해도 동일)
    manual_id   TEXT NOT NULL,          -- manuals.manual_id 또는 'LG-AC-COMMON' 같은 공통 문서 id (FK 없음)
    brand       TEXT NOT NULL,
    category    TEXT NOT NULL,
    section     TEXT NOT NULL,
    subsection  TEXT NOT NULL,
    tag         TEXT NOT NULL,          -- howto | symptom | safety | diagnosis | generic | error_code
    page_start  INTEGER,
    body        TEXT NOT NULL
);
CREATE INDEX ix_chunks_manual ON chunks (manual_id, tag);

-- 제목 + 첫 줄(증상 문장)만 색인. 본문 전체를 넣으면 흔한 단어 잡음으로 정확도가 떨어지는 걸 실측했음.
-- trigram: 형태소 분석 없이 한국어 부분 문자열 매칭. ('시원하게' ↔ '시원하' 3글자 공유)
CREATE VIRTUAL TABLE chunks_fts USING fts5 (
    chunk_id UNINDEXED,
    subsection,
    first_line,
    tokenize = 'trigram'
);

-- ============================================================================
-- 조회 예시 (라우터가 실제로 날리는 쿼리)
-- ============================================================================

-- [A] 등록 시 모델 → 매뉴얼 해석. 정확 일치 우선, 없으면 패턴(GLOB) 일치.
--     :model = 'FQ16FV6EDN'
-- SELECT m.manual_id
--   FROM manual_models mm JOIN manuals m USING (manual_id)
--  WHERE m.brand = :brand AND m.category = :category
--    AND (mm.model_pattern = :model OR :model GLOB replace(mm.model_pattern, '*', '?'))
--  ORDER BY (mm.model_pattern = :model) DESC, mm.source = 'filename' DESC
--  LIMIT 1;

-- [B] 에러코드 조회. :codes = 질문에서 뽑아 정규화한 코드 + 혼동 문자 변형들
-- SELECT e.entry_id, e.title, e.summary, e.content, e.chunk_id, c.code_display, c.kind
--   FROM error_codes c JOIN error_entries e USING (entry_id)
--  WHERE c.brand = :brand AND c.category = :category
--    AND c.code_norm IN (:code1, :code2, ...)
--  ORDER BY c.kind = 'primary' DESC;

-- [C] 기능 유무. 결과 없음 → 모름(검색 폴백), has=0 → "이 모델엔 없는 기능", has=1 → 있음(검색 진행)
-- SELECT f.feature, f.has, f.model_pattern
--   FROM model_features f
--  WHERE f.brand = :brand AND f.category = :category AND f.feature = :feature
--    AND :model GLOB replace(f.model_pattern, '*', '?');

-- [D] 사용자의 가전 → 검색 필터용 doc_id 목록 (챗봇 세션 시작 시 1회)
-- SELECT appliance_id, nickname, brand, category, model, manual_id
--   FROM user_appliances WHERE user_id = :user_id;
