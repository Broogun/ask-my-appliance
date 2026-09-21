"""검색 엔진을 박형건 exp02(청킹+벡터검색+listwise 리랭킹)로 교체한 버전.

배경: 실사용 로그 감사 중(2026-09-21) 세탁기 "탈수 시 모터소리+안돌아감" 질문에서
시연님 RRF 파이프라인(기본 n_candidates=8)이 실제 정답 청크(본문 벡터검색 5위)를
놓치는 걸 확인했다 - 본문에선 순위가 괜찮은데 예상질문 쪽 매칭이 안 돼서 RRF
합산 점수가 밀린 것. 같은 질문을 exp02 파이프라인(700자 청킹이 짧은 증상 항목
여러 개를 한 청크로 묶어서 저장 + top_k=40으로 풀이 넓음)으로 돌리면 정답을
바로 잡아왔다.

한 사례로 전체를 바꾸는 건 위험하다는 걸 이번 세션 내내 확인했으므로(청킹
오버랩 실험 3종 전부 실패 등), RDB(에러코드/모델매핑/기능유무)와 "등록 가전
→ manual_id" 해석 흐름은 시연님 것을 그대로 쓰고, **벡터검색+리랭킹 단계만**
exp02로 교체한다. RDB는 시연님의 무거운 LGAirconRetriever(임베딩+리랭커+
예상질문 캐시 전부 로딩)를 통째로 인스턴스화하지 않고, SQLite만 직접 열어서
가볍게 재구현했다 - 벡터 부분을 안 쓸 거면서 그 무거운 로딩을 또 할 필요가 없다.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "siyeon" / "rdb"))

from build_db import FEATURE_IMPLIED_BY, code_variants, extract_query_codes  # noqa: E402
from siyeon.rag.retrieval import CONTEXT_CUTOFF, RetrievalResult, apply_cutoff  # noqa: E402
import 박형건_exp02 as exp02  # noqa: E402

DB_PATH = ROOT / "data" / "appliance.sqlite"

_db: sqlite3.Connection | None = None


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _db.row_factory = sqlite3.Row
    return _db


def appliance_for(manual_id: str) -> dict:
    """시연님 LGAirconRetriever.appliance_for()와 동일 - manual_id는 이미 등록
    시점에 검증된 값이라 known_doc_ids 재확인 없이 바로 신뢰한다."""
    row = _get_db().execute(
        "SELECT brand, category, product_type FROM manuals WHERE manual_id = ?", (manual_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"manuals 테이블에 없는 매뉴얼: {manual_id}")
    return {
        "brand": row["brand"], "category": row["category"], "product_type": row["product_type"],
        "model": manual_id.split("_", 1)[1].upper() if "_" in manual_id else manual_id,
        "manual_id": manual_id,
    }


def lookup_error_codes(brand: str, category: str, query: str) -> list[dict]:
    """시연님 LGAirconRetriever.lookup_error_codes()와 동일 - 정규화된 코드로 정확 매칭."""
    variants = sorted({v for tok in extract_query_codes(query) for v in code_variants(tok)})
    if not variants:
        return []
    rows = _get_db().execute(
        f"SELECT e.entry_id, e.brand, e.category, e.title, e.summary, e.content, e.chunk_id, "
        f"c.code_display, c.kind FROM error_codes c JOIN error_entries e USING (entry_id) "
        f"WHERE c.brand = ? AND c.category = ? AND c.code_norm IN ({','.join('?' * len(variants))}) "
        f"ORDER BY c.kind = 'primary' DESC, e.entry_id", (brand, category, *variants),
    ).fetchall()
    seen, out = set(), []
    for r in rows:
        if r["entry_id"] not in seen:
            seen.add(r["entry_id"])
            out.append(dict(r))
    return out


def feature_available(appliance: dict, feature: str) -> bool | None:
    """시연님 LGAirconRetriever.feature_available()과 동일."""
    target = FEATURE_IMPLIED_BY.get(feature, feature)
    row = _get_db().execute(
        "SELECT has FROM model_features WHERE brand=? AND category=? AND feature=? "
        "AND ? GLOB replace(model_pattern,'*','?')",
        (appliance["brand"], appliance["category"], target, appliance.get("model") or ""),
    ).fetchone()
    return None if row is None else bool(row["has"])


def _error_entry_as_chunk(e: dict) -> dict:
    """시연님 _error_entry_as_chunk()의 폴백 경로(RDB 원문 그대로 사용) - exp02엔
    시연님 chunk_by_id가 없어 원본 청크 대조는 생략, RDB 내용만으로 구성해도
    정확도는 동일하다(에러코드 조회는 RDB가 곧 정답)."""
    label = f"에러코드 ({e.get('brand', '')} {e.get('category', '')})".strip()
    return {
        "id": e["chunk_id"] or f"rdb_error_{e['entry_id']}", "doc_id": "RDB", "section": label,
        "subsection": e["title"], "page_start": None, "page_end": None, "tag": "error_code",
        "body": e["content"], "text": f"{label} > {e['title']}\n{e['content']}", "source": "rdb",
    }


_SAFETY_KEYWORDS = ("화재", "감전", "가스", "누전", "상해나 사망", "인체에 해롭", "스파크")


def _translate_chunk(c: dict, manual_id: str) -> dict:
    """exp02 후보({"id","text","distance","metadata":{"section_title","page"}})를
    시연님 chunk 형식({"id","doc_id","section","subsection","tag","page_start",
    "page_end","body","text"})으로 변환 - sources_of()/build_web_context()가
    그대로 재사용된다."""
    meta = c.get("metadata", {})
    heading = meta.get("section_title", "")
    page = meta.get("page")
    body = c["text"]
    prefix = f"[{heading}] "
    if body.startswith(prefix):
        body = body[len(prefix):]
    # tag는 exp02 청킹에 없는 필드라 추정해야 한다 - 처음엔 섹션 제목에 "안전"이
    # 있는지만 봤는데, exp02 청킹은 제목이 중복될 때만 조상 경로("안전을 위해
    # 주의하기 > 경고 > ...")를 붙이므로 "전원 플러그나 전원선을 다룰 때"처럼
    # 유일한 제목이면 상위 챕터명이 사라져 놓치는 걸 실측 확인함(2026-09-21,
    # "젖은 손으로 플러그" 질문에서 안전 경고가 안 붙는 회귀). 제목 대신 본문
    # 자체에 실제 위험 경고 문구(화재/감전/가스/누전 등)가 있는지로 판별 -
    # 청킹 구조와 무관하게 원문에 그 단어가 있으면 항상 잡힌다.
    if any(kw in body for kw in _SAFETY_KEYWORDS):
        tag = "safety"
    elif "고장" in heading or "확인하기" in heading:
        tag = "symptom"
    else:
        tag = "howto"
    return {
        "id": c["id"], "doc_id": manual_id, "section": heading, "subsection": "",
        "tag": tag, "page_start": page, "page_end": page, "body": body, "text": c["text"],
    }


def retrieve(query: str, manual_id: str | None = None, top_k: int = 3, feature: str | None = None,
             appliance: dict | None = None) -> RetrievalResult:
    """rag_service.retrieve()가 그대로 호출할 수 있는 형태 - ① RDB 에러코드
    → ② 기능유무 표 → ③ exp02 벡터검색+listwise 리랭킹, 순서와 의미는 시연님
    retrieve()와 동일하다."""
    if appliance is None and manual_id:
        appliance = appliance_for(manual_id)

    if appliance:
        entries = lookup_error_codes(appliance["brand"], appliance["category"], query)
        if entries:
            hits = [(_error_entry_as_chunk(e), 0.0) for e in entries[:top_k]]
            return RetrievalResult(hits=hits, route="error_code", used=apply_cutoff(hits), appliance=appliance)

        if feature and manual_id and feature_available(appliance, feature) is False:
            body = f"이 모델({appliance['model']})에는 '{feature}' 기능이 없습니다. (모델별 기능 표 기준)"
            chunk = {
                "id": f"feature_absent_{feature}", "doc_id": manual_id, "section": "기능 없음",
                "subsection": feature, "page_start": None, "page_end": None, "tag": "feature_absent",
                "body": body, "text": body,
            }
            return RetrievalResult(hits=[(chunk, 1.0)], route="feature_absent", used=[(chunk, 1.0)], appliance=appliance)

    product_model = manual_id.split("_", 1)[1] if manual_id and "_" in manual_id else None
    state = exp02._load_state()
    where = {"product_model": product_model} if product_model else None
    found = exp02._vector_search(query, state, top_k=40, where=where)
    picked = exp02._llm_rerank([query], found, state, top_k=top_k)

    # picked는 순위 정보만 있고 점수가 없다(web/rag_listwise.py와 같은 이유) -
    # CONTEXT_CUTOFF(0.9)보다 확실히 낮은 값을 순서대로 매긴다.
    hits = [(_translate_chunk(c, manual_id or ""), 0.1 + i * 0.05) for i, c in enumerate(picked)]
    used = apply_cutoff(hits, CONTEXT_CUTOFF)
    route = "vector" if used else "none"
    return RetrievalResult(hits=hits, route=route, used=used, appliance=appliance)
