"""exp02 파이프라인을 web/rag_service.py(김시연)와 같은 4개 함수 계약으로
감싼 어댑터. 웹앱에 통째로 꽂아 넣을 때 이 파일 하나만 옮기면 된다.

    und = understand_query(query, appliance)   # exp02엔 별도 "질문 이해" 단계가 없어 패스스루
    res = retrieve(query, manual_id, und.feature)
    stream_answer(query, res, history)          # 최근 대화까지 반영해 스트리밍
    sources_of(res)                             # 출처 패널 데이터

통합 방법(web/README.md 형식 그대로): web/routers/*.py의
`from ..rag_service import ...`를 `from ..rag_service_exp02_adapter import ...`로
바꾸면 이 파이프라인으로 통째로 전환된다. 이 파일은 일부러 자체완결형으로
작성했다 - RetrievalResult/Understanding을 siyeon.rag에서 import하지 않고
같은 모양으로 직접 재정의해서, 어느 저장소/브랜치에 두든 exp02_retrieval.py
하나만 옆에 있으면 동작한다.

⚠ 알려진 통합 갭 (2026-09-18, NOTES_ARCHITECTURE.md 참고 - 다음에 다듬을 지점):
1. exp02는 "질문 문자열 안에 모델 코드가 있다"고 가정하고 모델을 감지한다
   (팀 공용 MODEL_QUESTIONS는 항상 그럼). 웹은 이미 등록된 가전의 manual_id를
   따로 넘겨주므로, 여기서는 manual_id를 exp02.known_models와 매칭해 질문
   앞에 모델 코드를 붙여서 넘긴다(_resolve_model 참고) - exp02 내부 코드를
   고치는 대신 입력 형식을 맞추는 방식. manual_id가 여러 모델이 공유하는
   공용 설명서 이름이라 모델 목록에 정확히/부분적으로도 안 걸리면(PDF명≠
   모델명 문제, 역할분담 논의 참고) 필터 없는 전체 검색으로 대체된다 - 이
   경우 다른 모델 내용이 섞일 위험이 있어 개선이 필요하다.
2. exp02의 확신 게이트(_is_relevant_enough)는 후보 전체를 하나로 합쳐 관련
   있는지 한 번만 판단한다(전부 보여주거나 전부 보류) - 청크 단위로 컷하는
   구조가 아니라서, 여기서는 게이트를 통과하면 candidates 전체를 그대로
   RetrievalResult.used로 채운다. feature_absent 라우트는 exp02에 대응하는
   기능이 없어 절대 나오지 않는다.
3. 멀티턴은 여기서만 새로 추가함 - exp02.SYSTEM_PROMPT(팀 채점 기준, 수정
   금지)는 그대로 쓰고, 최근 대화(history)를 그 앞뒤에 messages로만 추가해서
   스트리밍한다(my_answer()엔 없는 기능. 검색 자체는 재작성하지 않는다 -
   siyeon 쪽도 같은 이유로 질문 재작성 단계를 뺐다고 되어 있음).
4. exp02._load_state()는 GPU에 임베딩 모델을 올린다 - 같은 프로세스에 다른
   팀원 리트리버가 이미 떠 있으면 VRAM을 나눠 써야 한다. 이 어댑터로
   "통째로 교체"할 때만 쓰고, 두 리트리버를 동시에 띄우는 용도로는 안 만듦.
"""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import importlib.util

_spec = importlib.util.spec_from_file_location("exp02_retrieval", ROOT / "exp02_retrieval.py")
exp02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp02)

SUPPORT = {  # 고객센터 연결 버튼 (브랜드별) - web/rag_service.py와 동일한 상수
    "lg": {"name": "LG전자 고객센터", "phone": "1544-7777", "url": "https://www.lge.co.kr/support"},
    "samsung": {"name": "삼성전자 서비스", "phone": "1588-3366", "url": "https://www.samsung.com/sec/support/"},
}
HISTORY_MESSAGES = 8  # 답변 LLM에 그대로 넣는 최근 메시지 수


@dataclass
class Understanding:
    query: str
    feature: str | None = None
    other_product: str | None = None
    source: str = "llm"  # llm | cache | passthrough(exp02엔 이 단계가 없어 항상 passthrough)


@dataclass
class RetrievalResult:
    hits: list                      # [(chunk_dict, distance)] 전체 후보, 거리 오름차순은 보장 안 함(exp02는 리랭킹 순서)
    route: str                      # "error_code" | "vector" | "none" (exp02엔 feature_absent 없음)
    used: list = field(default_factory=list)   # 확신 게이트를 통과해 LLM에 넘어간 조각(전부 또는 없음)
    appliance: dict | None = None

    @property
    def grounded(self) -> bool:
        return bool(self.used)

    def candidates(self) -> list[dict]:
        """팀 채점 형식 [{"id", "text", "distance"}] - my_answer()의 반환 계약과 동일."""
        return [{"id": c["id"], "text": c["text"], "distance": round(float(d), 4)} for c, d in self.hits]


_state: dict | None = None
_lock = threading.Lock()


def get_retriever() -> dict:
    """첫 호출 때 exp02._load_state() 실행(임베딩 모델·Chroma·에러코드 RDB 로딩, GPU 기준
    수십초). 이후 재사용."""
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = exp02._load_state()
    return _state


def is_ready() -> bool:
    return _state is not None


def warm_in_background() -> None:
    """서버 시작 직후 호출 - 사용자가 채팅 화면에 도달하기 전에 로딩을 시작한다."""
    if _state is None:
        threading.Thread(target=get_retriever, name="rag-warmup-exp02", daemon=True).start()


def manual_pdf_path(doc_id: str) -> Path | None:
    """doc_id(PDF 파일명 stem) -> data/{brand}/{category}/{doc_id}.pdf."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id):
        return None
    return next(iter((ROOT.parent / "data").glob(f"*/*/{doc_id}.pdf")), None)


def _resolve_model(manual_id: str, state: dict) -> str | None:
    """manual_id(웹의 등록 가전 식별자)를 exp02의 product_model 코드로 매핑.

    정확히 일치하면 그대로 쓰고, 아니면 known_models 중 서로 포함 관계인 걸
    찾는다(부분 일치) - 통합 갭 1번, 파일 상단 설명 참고."""
    known = state.get("known_models", [])
    if manual_id in known:
        return manual_id
    for m in known:
        if m in manual_id or manual_id in m:
            return m
    return None


def understand_query(query: str, appliance) -> Understanding:
    """exp02엔 기능명 정규화·다른 제품 감지 단계가 없다 - 항상 패스스루로 반환한다.
    질문 자체는 retrieve()의 gather_candidates가 그대로 처리한다."""
    return Understanding(query=query, feature=None, other_product=None, source="passthrough")


def retrieve(query: str, manual_id: str, feature: str | None = None, top_k: int = 3) -> RetrievalResult:
    state = get_retriever()
    model = _resolve_model(manual_id, state)
    # exp02는 질문 문자열 안의 모델 코드로 where 필터를 결정한다(통합 갭 1번) -
    # 모델을 찾았으면 질문 앞에 붙여서 넘긴다. _strip_known_models가 실제 검색
    # 시점에 다시 떼어내므로 검색어 자체는 오염되지 않는다.
    effective_query = f"{model} {query}" if model else query

    candidates = exp02.gather_candidates(effective_query, state)
    hits = [(c, c["distance"]) for c in candidates]

    if not candidates:
        return RetrievalResult(hits=[], route="none", used=[], appliance={"manual_id": manual_id, "model": model})

    context = exp02._format_context_with_pages(candidates)
    min_distance = min(c["distance"] for c in candidates)
    grounded = exp02._is_relevant_enough(query, context, min_distance)

    route = "error_code" if any(c["id"].startswith("rdb_") for c in candidates) else "vector"
    if not grounded:
        route = "none"

    return RetrievalResult(
        hits=hits, route=route, used=(hits if grounded else []),
        appliance={"manual_id": manual_id, "model": model},
    )


def sources_of(res: RetrievalResult) -> list[dict]:
    """출처 패널용 - exp02 청크엔 section/subsection 분리가 없어 section_title
    하나를 제목으로 쓴다."""
    out = []
    for c, d in res.used:
        meta = c.get("metadata", {})
        title = meta.get("section_title") or meta.get("product_model") or c["id"]
        tag = "오류코드 안내" if c["id"].startswith("rdb_") else "사용설명서"
        doc_id = meta.get("product_model", "")
        out.append({
            "title": title, "snippet": c["text"][:220], "body": c["text"], "tag": tag,
            "doc_id": doc_id, "distance": round(float(d), 4), "chunk_id": c["id"],
            "page_start": meta.get("page"), "page_end": meta.get("page"),
            "pdf_url": f"/api/manuals/{doc_id}/pdf" if doc_id and manual_pdf_path(doc_id) else None,
        })
    return out


def stream_answer(query: str, res: RetrievalResult, history: list[dict] | None = None):
    """exp02.SYSTEM_PROMPT(팀 채점 기준, 수정 금지)는 그대로 쓰고 최근 대화만
    앞뒤에 얹어 스트리밍 - my_answer()엔 없는 멀티턴을 이 어댑터에서만 추가."""
    if not res.used:
        yield exp02.NO_ANSWER_MESSAGE
        return

    candidates = [c for c, _ in res.used]
    context = exp02._format_context_with_pages(candidates)
    messages = [{"role": "system", "content": exp02.SYSTEM_PROMPT}]
    for m in (history or [])[-HISTORY_MESSAGES:]:
        messages.append({"role": m["role"], "content": m["content"][:800]})
    messages.append({"role": "user", "content": f"[문서]\n{context}\n\n[질문]\n{query}"})

    stream = exp02.OAI.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

    citation = exp02._page_citation(candidates)
    if citation:
        yield citation
