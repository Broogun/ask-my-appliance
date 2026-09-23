"""검색 엔진을 exp02(청킹+벡터검색+listwise 리랭킹)로 교체한 버전.

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

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "siyeon" / "rdb"))

from build_db import FEATURE_IMPLIED_BY, code_variants, extract_query_codes  # noqa: E402
from siyeon.rag.retrieval import CONTEXT_CUTOFF, RetrievalResult, apply_cutoff  # noqa: E402
from . import rdb  # noqa: E402
from .manual_pages import locate_pages  # noqa: E402
import exp02_retrieval as exp02  # noqa: E402



def appliance_for(manual_id: str) -> dict:
    """시연님 LGAirconRetriever.appliance_for()와 동일 - manual_id는 이미 등록
    시점에 검증된 값이라 known_doc_ids 재확인 없이 바로 신뢰한다."""
    row = rdb.one("SELECT brand, category, product_type FROM manuals WHERE manual_id = ?", (manual_id,))
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
    rows = rdb.rows(
        f"SELECT e.entry_id, e.brand, e.category, e.title, e.summary, e.content, e.chunk_id, "
        f"c.code_display, c.kind FROM error_codes c JOIN error_entries e USING (entry_id) "
        f"WHERE c.brand = ? AND c.category = ? AND c.code_norm IN ({','.join('?' * len(variants))}) "
        f"ORDER BY c.kind = 'primary' DESC, e.entry_id", (brand, category, *variants))
    seen, out = set(), []
    for r in rows:
        if r["entry_id"] not in seen:
            seen.add(r["entry_id"])
            out.append(r)
    return out


def feature_available(appliance: dict, feature: str) -> bool | None:
    """시연님 LGAirconRetriever.feature_available()과 동일."""
    target = FEATURE_IMPLIED_BY.get(feature, feature)
    row = rdb.one("SELECT has FROM model_features WHERE brand=? AND category=? AND feature=? AND {MODEL_MATCH}",
                  (appliance["brand"], appliance["category"], target, appliance.get("model") or ""))
    return None if row is None else bool(row["has"])


_ERROR_IMG_INDEX = ROOT / "data" / "error_images" / "index.json"


def _wanted_variants(appliance: dict | None) -> set[str] | None:
    """LG 세탁기 에러코드 페이지는 드럼/통돌이가 따로다 - 통돌이 모델명은 T로 시작한다(T17J4EFNTX, TR16MV6 등)."""
    if appliance and appliance.get("brand") == "lg" and appliance.get("category") == "washer":
        return {"top"} if (appliance.get("model") or "").upper().startswith("T") else {"drum"}
    return None


def _error_images(chunk_id: str | None, appliance: dict | None = None) -> list[dict]:
    """scripts/crawl_error_images.py 가 받아 둔 제조사 고객지원 페이지 사진(없으면 빈 목록)."""
    if not chunk_id or not _ERROR_IMG_INDEX.exists():
        return []
    try:
        index = json.loads(_ERROR_IMG_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    want = _wanted_variants(appliance)
    return [{"url": f"/api/error-images/{i['file']}", "alt": i.get("alt", ""), "role": i.get("role", "body"),
             "page": i.get("page"), "credit": i.get("credit")}
            for i in index.get(chunk_id, [])
            if i.get("file") and not (want and i.get("variant") in ("drum", "top") and i["variant"] not in want)]


def _error_entry_as_chunk(e: dict, appliance: dict | None = None) -> dict:
    """시연님 _error_entry_as_chunk()의 폴백 경로(RDB 원문 그대로 사용) - exp02엔
    시연님 chunk_by_id가 없어 원본 청크 대조는 생략, RDB 내용만으로 구성해도
    정확도는 동일하다(에러코드 조회는 RDB가 곧 정답)."""
    label = f"에러코드 ({e.get('brand', '')} {e.get('category', '')})".strip()
    return {
        "id": e["chunk_id"] or f"rdb_error_{e['entry_id']}", "doc_id": "RDB", "section": label,
        "subsection": e["title"], "page_start": None, "page_end": None, "tag": "error_code",
        "body": e["content"], "text": f"{label} > {e['title']}\n{e['content']}", "source": "rdb",
        "images": _error_images(e.get("chunk_id"), appliance),
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
    page_start, page_end = locate_pages(manual_id, body, page)   # 메타데이터 page는 섹션 시작 쪽이라 1~5쪽 앞을 가리키는 경우가 31%
    return {
        "id": c["id"], "doc_id": manual_id, "section": heading, "subsection": "",
        "tag": tag, "page_start": page_start, "page_end": page_end, "body": body, "text": c["text"],
        "is_structural": bool(c.get("_structural")),
    }


def retrieve(query: str, manual_id: str | None = None, top_k: int = 5, feature: str | None = None,
             appliance: dict | None = None) -> RetrievalResult:
    """rag_service.retrieve()가 그대로 호출할 수 있는 형태 - ① RDB 에러코드
    → ② 기능유무 표 → ③ exp02 벡터검색+listwise 리랭킹, 순서와 의미는 시연님
    retrieve()와 동일하다."""
    if appliance is None and manual_id:
        appliance = appliance_for(manual_id)

    if appliance:
        entries = lookup_error_codes(appliance["brand"], appliance["category"], query)
        if entries:
            hits = [(_error_entry_as_chunk(e, appliance), 0.0) for e in entries[:top_k]]
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
    # _strip_known_models 연결을 시도했다가 되돌림(2026-09-23) - exp02.my_answer()에서 검증된
    # 픽스(예전 pointwise CrossEncoder 기준)라 여기(listwise LLM 리랭커) 연결도 시도했는데,
    # 검증된 10개 모델 한정 오프라인 A/B(rerank_strip_ab_test_v3.py)에서는 71.2%->74.0%로
    # 오히려 개선됐지만, 실제 round1 60개 모델 전체 재측정에서는 SUCCESS가 92.3%->86~87%로
    # 떨어졌다 - 검증 안 된 나머지 50개 모델에서 부작용이 있는 것으로 추정. 좁은 범위 오프라인
    # 지표보다 실제 결과를 우선해 되돌림. 연결하려면 60개 모델 전체 기준 검증이 먼저 필요하다.
    found = exp02._vector_search(query, state, top_k=40, where=where)
    picked = exp02._llm_rerank([query], found, state, top_k=top_k)

    # 구조적 챕터 매칭(exp02.gather_candidates와 같은 로직, 원래 실험 하니스에만 연결돼 있었는데
    # 벡터검색+리랭킹만 이식할 때 빠졌던 걸 실사용 로그 감사로 발견함) - "에러 메시지 확인하는
    # 방법"처럼 "에러"라는 단어가 원문에 없어 벡터검색이 놓치는 질문은, 모델별로 미리 인덱싱해 둔
    # 대표 트러블슈팅/패널 챕터(고장신고 전 확인하기, LG ThinQ로 고장 진단하기 등)가 정답이다.
    # find_troubleshooting_chapter/find_panel_chapter는 내부에서 이미 자체 리랭킹을 하므로(그
    # 챕터 안의 세부 청크만 추리는 것), 위 vector 리랭킹 결과와 다시 합쳐 리랭킹하지 않고 앞에
    # 붙이기만 한다 - exp02._load_state()가 이미 인덱스를 만들어 둬서 추가 계산 비용도 없고,
    # 패턴이 안 걸리는 보통 질문엔 LLM 호출이 전혀 추가되지 않는다(2026-09-22, round1 300문항
    # 실패 23건 중 13건이 이 유형이었음. 10/13은 이 인덱스에 챕터가 있었고, 아래 마킹까지
    # 붙인 뒤 재현 테스트로 10/10 확인함(단, 그중 1건은 해당 모델 설명서에 애초에 화면 에러
    # 코드 표시 자체가 없어 "설명서에 없다"는 답이 맞는 정답 케이스). 나머지 3건은 처음엔
    # "삼성 PDF 목차 손상"이라 짐작했는데 틀렸다 - 실제로는 exp02._load_state()의
    # _TS_TITLES exact-match가 놓친 두 가지 제목 변형 패턴이었고, 인덱싱 로직을 고쳐서
    # 마저 잡았다(experiments/exp02_retrieval.py, 상세 경위는 docs/05-evaluation.md 5.3절).
    if product_model:
        structural: list[dict] = []
        if exp02.is_generic_troubleshooting_question(query):
            structural += exp02.find_troubleshooting_chapter(query, product_model, state, top_k=top_k)
        if exp02.is_panel_setting_question(query):
            structural += exp02.find_panel_chapter(query, product_model, state, top_k=top_k)
        if structural:
            # WEB_SYSTEM_PROMPT 규칙 2("근거 제목과 질문을 비교해서 고르라")가 여기서는
            # 역효과를 낸다 - 삼성 등 모델은 이 챕터의 실제 하위 제목이 "고장신고 전
            # 확인하기"처럼 뭉뚱그려져 있어서(LG처럼 "LG ThinQ로 고장 진단하기" 같은
            # 구체적 부제목이 없음), 제목만 보고는 "에러 코드는 어디서 확인해" 질문과
            # 안 맞는다고 LLM이 오판해 통째로 거절하는 걸 실측 확인함(2026-09-22, 10건
            # 재현 테스트 중 4건 - SQ09GK1WEN/DV90TA040TE/DF60R8300WG/DF90H24R5C, 내용은
            # 맞는데 제목 매칭 규칙 때문에 거절). 안전 경고와 같은 이유로 - "이 근거가
            # 맞다"는 판단을 LLM의 제목-비교 휴리스틱에 맡기지 않고, 사전 검증된 구조적
            # 매칭 결과라는 걸 _structural 플래그로 표시해서 넘긴다(build_web_context가
            # 이 플래그를 보고 제목 표시 + "점검문자/코드 표는 곧 화면 표시 코드다"라는
            # 해석 규칙을 함께 준다 - 표 제목만 있고 "화면에 표시됩니다" 연결 문장이
            # 청킹 경계에서 잘린 경우까지 커버해야 4건이 전부 잡혔다).
            for c in structural:
                c["_structural"] = True
            seen_ids = {c["id"] for c in structural}
            picked = (structural + [c for c in picked if c["id"] not in seen_ids])[:top_k]

    # picked는 순위 정보만 있고 점수가 없다(web/rag_listwise.py와 같은 이유) -
    # CONTEXT_CUTOFF(0.9)보다 확실히 낮은 값을 순서대로 매긴다.
    hits = [(_translate_chunk(c, manual_id or ""), 0.1 + i * 0.05) for i, c in enumerate(picked)]
    used = apply_cutoff(hits, CONTEXT_CUTOFF)
    route = "vector" if used else "none"
    return RetrievalResult(hits=hits, route=route, used=used, appliance=appliance)
