"""data/appliance.sqlite 생성 스크립트.

  python experiments/siyeon/rdb/build_db.py            # 기본: data/appliance.sqlite 재생성
  python experiments/siyeon/rdb/build_db.py --check    # 생성 후 에러코드 파싱 결과를 표로 출력 (검수용)

입력:
  data/{brand}/{category}/*.pdf          → manuals, manual_models, model_features (부록 '모델별 기능' 표, LG만 있음)
  data/{brand}_{category}_errors.json    → error_entries, error_codes
스키마: experiments/siyeon/rdb/schema.sql  (테이블 설명·조회 예시는 그쪽에)

코드 정규화(normalize_code)는 질문 파서(retrieval.py)와 같은 함수를 써야 하므로 여기서 import해서 쓴다:
  sys.path.insert(0, "experiments/siyeon/rdb"); from build_db import normalize_code, code_variants
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = Path(__file__).with_name("schema.sql")
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "appliance.sqlite"

BRANDS = {"lg": "LG", "samsung": "삼성"}
CATEGORIES = {"aircon": "에어컨", "fridge": "냉장고", "washer": "세탁기"}

# 질문에서 기능명을 잡기 위한 별칭 (feature_aliases). 비교 전에 소문자·공백 제거.
# 키는 부록 표 헤더를 정규화한 feature 이름과 같아야 한다 (normalize_feature 참고).
FEATURE_ALIASES = {
    "클린봇":        ["클린봇", "클린 봇", "자동청소", "필터자동청소", "자동필터청소"],   # '로봇청소'는 본문의 "로봇청소기"(인체감지 오작동 안내)에 걸려 클린봇 없는 모델을 있는 걸로 오판 → 제외
    "UVnano":       ["uvnano", "유브이나노", "uv나노", "uv살균", "자외선살균", "팬살균"],
    "공기청정":      ["공기청정", "공청", "미세먼지", "청정기능"],
    "종합청정도":    ["종합청정도", "청정도표시", "미세먼지표시", "공기질표시"],
    "음성인식":      ["음성인식", "음성명령", "말로", "목소리로", "음성으로"],
    "음성안내":      ["음성안내", "말로알려", "음성으로알려"],
    "활동부재":      ["활동부재", "부재감지", "사람없으면", "사람감지", "인체감지"],
    "스마트가이드":   ["스마트가이드"],
    "좌우토출구덮개": ["좌우토출구덮개", "토출구덮개", "토출구커버"],
    "알러지케어집진필터": ["알러지케어", "알러지필터", "집진필터"],
    "극세필터":      ["극세필터"],
    "먼지통":        ["먼지통", "먼지비움"],          # 표 헤더엔 없지만(클린봇 부속) 질문엔 자주 나옴 → 클린봇 유무로 판정하게 매핑
    # 세탁기 (표 없음 → feature_available 2차 판정: 설명서에 아래 단어가 0회면 '없는 기능')
    "세제자동투입":   ["세제자동투입", "자동투입", "세제자동", "자동세제", "ezdispense", "이지디스펜스"],
}
# 표에 없는 기능명을 표에 있는 기능으로 연결 (먼지통은 클린봇이 있어야 존재)
FEATURE_IMPLIED_BY = {"먼지통": "클린봇"}


# ── 코드 정규화 (질문 파서와 공유) ───────────────────────────────────────────
# 질문/제목에서 코드로 볼 토큰: 영문 1~3자 + 숫자 0~3자 (CH05, E0, C101, FF, dE1, 1E, 5C, tCL)
CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,3}\s?-?\d{0,3}|\d[A-Za-z]{1,2})(?![A-Za-z0-9])")
# 질문에서 "코드 질문"으로 판단할 최소 조건: 영문+숫자 조합 또는 대문자 2~3자 코드 (FF, UE, IE, dE)
QUERY_CODE_RE = re.compile(r"(?<![A-Za-z0-9가-힣])(?:[A-Za-z]{1,3}\s?-?\d{1,3}|\d[A-Za-z]{1,2}|[A-Z][A-Za-z]{1,2}|[a-z][A-Z]{1,2})(?![A-Za-z0-9])")   # 마지막 대안: dE, tE, dH, rF 처럼 소문자로 시작하는 LG 코드(2026-09-22)
CONFUSABLE = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S", "B": "8", "8": "B"}


def normalize_code(token: str) -> str:
    """'ch 05' → 'CH5', 'C101' → 'C101', 'E0' → 'E0', 'Er FF' → 'ERFF'. 영숫자만 남기고 숫자부 앞 0 제거."""
    s = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    m = re.fullmatch(r"([A-Z]*)(\d+)([A-Z]*)", s)
    if m and m.group(2):
        s = m.group(1) + str(int(m.group(2))) + m.group(3)
    return s


def code_variants(token: str) -> list[str]:
    """정규화 키 + 혼동 문자(O/0, I/1, S/5, B/8) 한 글자 치환 변형. 조회 시 IN (...) 에 넣는다."""
    base = normalize_code(token)
    out = {base}
    for i, ch in enumerate(base):
        if ch in CONFUSABLE:
            out.add(base[:i] + CONFUSABLE[ch] + base[i + 1:])
    return sorted(out)


def extract_query_codes(query: str) -> list[str]:
    """질문에서 코드 후보를 뽑는다. 'CH05 에러가 떴어요' → ['CH05'], '0E 뜨는데' → ['0E'], 'FF 떠요' → ['FF']"""
    return [m.group(0) for m in QUERY_CODE_RE.finditer(query)]


# ── 에러코드 JSON 제목 파싱 ───────────────────────────────────────────────────
def split_title(title: str) -> tuple[str, str]:
    """'CH05 / E0 / CH53 (통신 에러)' → ('CH05 / E0 / CH53', '통신 에러').  요약은 제목 끝의 괄호."""
    m = re.fullmatch(r"(.*?)\s*\(([^()]*)\)\s*$", title.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (title.strip(), "")


def expand_paren_alternatives(text: str) -> list[str]:
    """'Er(E) FF' → ['Er FF', 'E FF'].  괄호 안이 대안 표기일 때만 (LG 냉장고 표기)."""
    m = re.search(r"(\w+)\((\w+)\)", text)
    if not m:
        return [text]
    a, b = m.group(1), m.group(2)
    return [text.replace(m.group(0), a), text.replace(m.group(0), b)]


def parse_codes(codes_part: str) -> list[tuple[str, str]]:
    """코드 부분 문자열 → [(code_norm, code_display)]. 코드가 아닌 문장이면 빈 리스트.

    'C101/E101, C102/E102' → C101, E101, C102, E102
    'Er(E) FF / Er(E) rF'  → ERFF, EFF, FF, ERRF, ERF, RF   (접두어 뗀 'FF'도: 사용자는 'FF 떠요'라고 침)
    '온도 표시 깜빡임 + 냉기 없음' → []
    """
    out: list[tuple[str, str]] = []
    for piece in re.split(r"\s*[/,]\s*", codes_part):
        piece = piece.strip()
        if not piece:
            continue
        for alt in expand_paren_alternatives(piece):
            alt = alt.strip()
            # 'Er FF' / 'E FF' 처럼 접두어 + 코드 → 둘 다, 그리고 코드 단독
            m = re.fullmatch(r"(Er|E)\s+([A-Za-z0-9]{1,4})", alt)
            tokens = [alt.replace(" ", ""), m.group(2)] if m else [alt]
            for tok in tokens:
                if not re.fullmatch(r"[A-Za-z]{0,3}\d{0,3}[A-Za-z]{0,3}", tok.replace(" ", "")) or not re.search(r"[A-Za-z0-9]", tok):
                    continue  # 문장/한글 → 코드 아님
                if not re.fullmatch(r"[A-Za-z0-9 ]+", tok):
                    continue
                out.append((normalize_code(tok), tok.replace(" ", "")))
    # 중복 제거 (순서 유지)
    seen, uniq = set(), []
    for k, d in out:
        if k and k not in seen:
            seen.add(k); uniq.append((k, d))
    return uniq


def variant_codes_in_summary(summary: str) -> list[tuple[str, str]]:
    """요약 괄호 안의 부수 코드: '제품에 따라 P4/P6/P7/P8/CH34' → P4, P6, P7, P8, CH34"""
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9가-힣])([A-Za-z]{1,3}\d{1,3})(?![A-Za-z0-9])", summary):
        out.append((normalize_code(m.group(1)), m.group(1)))
    return out


def chunk_id_for_error(common_doc_id: str, title: str, content: str) -> str:
    """노트북 make_chunk()와 같은 규칙으로 청크 id를 만든다 (벡터 인덱스 ↔ RDB 연결용)."""
    digest = hashlib.md5(content.strip().encode("utf-8")).hexdigest()[:8]
    return re.sub(r"\s+", "_", f"{common_doc_id}_에러코드_{title.strip()}_{digest}")


# ── PDF: 매뉴얼 / 모델 패턴 / 기능 표 ─────────────────────────────────────────
def normalize_feature(header: str) -> str:
    """표 헤더 → feature 키. '알러지 케어 집진 필터' → '알러지케어집진필터', 'UVnano' 그대로(대소문자 유지)."""
    return re.sub(r"\s+", "", header.strip())


def model_from_filename(stem: str) -> str:
    """'AC_FQ18GC1EHN' → 'FQ18GC1EHN', 'RF_GC-B40BSCQ' → 'GC-B40BSCQ'"""
    return stem.split("_", 1)[1] if "_" in stem else stem


def extract_feature_tables(doc) -> list[tuple[list[str], list[list[str]]]]:
    """부록의 '모델명 | 기능 …' 표를 (헤더, 행들) 목록으로. 페이지가 넘어가며 헤더가 반복되면 이어붙인다."""
    pages = sorted({pg - 1 for lvl, title, pg in doc.get_toc() if "모델별" in title})
    tables: list[tuple[list[str], list[list[str]]]] = []
    for pi in pages:
        for t in doc[pi].find_tables().tables:
            rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in t.extract()]
            if not rows or not rows[0] or "모델명" not in rows[0][0]:
                continue
            header, body = None, []
            for r in rows:
                if r[0].startswith("모델명"):          # 헤더 (페이지 넘김으로 반복될 수 있음)
                    if header and header != r and body:
                        tables.append((header, body)); body = []
                    header = r
                elif header:
                    body.append(r)
            if header and body:
                # 같은 헤더의 표가 직전에 있으면 이어붙임 (페이지 분할)
                if tables and tables[-1][0] == header:
                    tables[-1][1].extend(body)
                else:
                    tables.append((header, body))
    return tables


# ── 빌드 ─────────────────────────────────────────────────────────────────────
def build(db_path: Path = DB_PATH, data_dir: Path = DATA_DIR, verbose: bool = True) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.executemany("INSERT INTO brands VALUES (?, ?)", BRANDS.items())
    con.executemany("INSERT INTO categories VALUES (?, ?)", CATEGORIES.items())

    # 1) 매뉴얼 + 모델 패턴 + 기능 표
    try:   # 평가 체계의 8분류 (없으면 폴더 3분류의 한글명)
        sys.path.insert(0, str(ROOT))
        from pipeline.evaluate import MODEL_LIST
        ptype = {model: cat for brand, cat, model in MODEL_LIST}
    except Exception:
        ptype = {}
    n_manuals = n_features = 0
    for brand in BRANDS:
        for category in CATEGORIES:
            for pdf in sorted((data_dir / brand / category).glob("*.pdf")):
                doc = pymupdf.open(str(pdf))
                tables = extract_feature_tables(doc)
                model_raw = model_from_filename(pdf.stem)
                con.execute(
                    "INSERT INTO manuals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (pdf.stem, brand, category, pdf.relative_to(ROOT).as_posix(),
                     hashlib.md5(pdf.read_bytes()).hexdigest(), len(doc), int(bool(tables)),
                     ptype.get(model_raw, CATEGORIES.get(category))),
                )
                model = model_raw.upper()
                con.execute("INSERT OR IGNORE INTO manual_models VALUES (?, ?, 'filename')", (pdf.stem, model))
                if "-" in model:  # 사용자가 하이픈 없이 칠 수 있음
                    con.execute("INSERT OR IGNORE INTO manual_models VALUES (?, ?, 'filename')", (pdf.stem, model.replace("-", "")))
                for header, rows in tables:
                    features = [normalize_feature(h) for h in header[1:]]
                    for r in rows:
                        for pattern in r[0].split():           # 'FQ**FN9*** FQ**FN7***' → 두 행
                            pattern = pattern.upper()
                            con.execute("INSERT OR IGNORE INTO manual_models VALUES (?, ?, 'appendix_table')", (pdf.stem, pattern))
                            for feat, val in zip(features, r[1:]):
                                if val.strip().upper() in ("O", "X"):
                                    con.execute(
                                        "INSERT OR REPLACE INTO model_features VALUES (?, ?, ?, ?, ?, ?)",
                                        (brand, category, pattern, feat, int(val.strip().upper() == "O"), pdf.stem),
                                    )
                                    n_features += 1
                doc.close()
                n_manuals += 1
    for feat, aliases in FEATURE_ALIASES.items():
        for a in aliases:
            con.execute("INSERT OR IGNORE INTO feature_aliases VALUES (?, ?)", (feat, re.sub(r"\s+", "", a).lower()))

    # 2) 에러코드
    n_entries = n_codes = 0
    for jf in sorted(data_dir.glob("*_errors.json")):
        brand, category = jf.stem.replace("_errors", "").split("_", 1)
        for product in json.loads(jf.read_text(encoding="utf-8")):
            common_doc_id = product["product_model"].replace("ERRORCODE", "COMMON")   # 'LG-AC-COMMON' (노트북과 동일)
            for sec in product["sections"]:
                codes_part, summary = split_title(sec["title"])
                primary = parse_codes(codes_part)
                cur = con.execute(
                    "INSERT INTO error_entries (brand, category, title, summary, content, is_code, source_file, chunk_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (brand, category, sec["title"].strip(), summary, sec["content"].strip(), int(bool(primary)),
                     jf.relative_to(ROOT).as_posix(), chunk_id_for_error(common_doc_id, sec["title"], sec["content"])),
                )
                entry_id = cur.lastrowid
                n_entries += 1
                for kind, codes in (("primary", primary), ("variant", variant_codes_in_summary(summary))):
                    for norm, disp in codes:
                        con.execute("INSERT OR IGNORE INTO error_codes VALUES (?, ?, ?, ?, ?, ?)",
                                    (brand, category, norm, disp, kind, entry_id))
                        n_codes += 1
    con.commit()
    if verbose:
        print(f"{db_path} 생성: 매뉴얼 {n_manuals}, 기능 {n_features}, 에러 항목 {n_entries}, 코드 별칭 {n_codes}")
    return con


def print_check(con: sqlite3.Connection) -> None:
    """검수용: 항목마다 어떤 코드로 파싱됐는지 한 줄씩."""
    rows = con.execute(
        "SELECT e.brand, e.category, e.title, e.is_code, "
        "       group_concat(CASE WHEN c.kind='primary' THEN c.code_norm END, ' '), "
        "       group_concat(CASE WHEN c.kind='variant' THEN c.code_norm END, ' ') "
        "FROM error_entries e LEFT JOIN error_codes c USING (entry_id) GROUP BY e.entry_id ORDER BY e.brand, e.category, e.entry_id"
    ).fetchall()
    for brand, cat, title, is_code, prim, var in rows:
        flag = "" if is_code else "  [코드 아님 → 벡터 검색만]"
        print(f"{brand:8s}{cat:7s} {title[:52]:52s} → {prim or '-'}" + (f"  (+{var})" if var else "") + flag)
    print("\n기능 표:")
    for r in con.execute("SELECT manual_id, count(DISTINCT model_pattern), count(*) FROM model_features GROUP BY manual_id"):
        print(f"  {r[0]}: 패턴 {r[1]}개, 셀 {r[2]}개")
    print("\n모델 패턴이 파일명 모델과 안 맞는 매뉴얼 (표의 패턴이 파일명 모델을 커버하지 않음):")
    for (mid,) in con.execute("SELECT manual_id FROM manuals WHERE has_feature_table = 1"):
        model = model_from_filename(mid).upper()
        ok = con.execute("SELECT 1 FROM manual_models WHERE manual_id=? AND source='appendix_table' AND ? GLOB replace(model_pattern,'*','?')", (mid, model)).fetchone()
        if not ok:
            pats = [r[0] for r in con.execute("SELECT model_pattern FROM manual_models WHERE manual_id=? AND source='appendix_table'", (mid,))]
            print(f"  {mid}: 파일명 모델 {model} ∉ {pats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="생성 후 파싱 결과 검수 출력")
    args = ap.parse_args()
    con = build()
    if args.check:
        print_check(con)
