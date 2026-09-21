"""검색·답변 서비스 — experiments/siyeon/rag 를 그대로 import 해서 검색기 1개를 상주시킨다. 라우터는 이 파일의 함수만 부른다.

    und = understand_query(query, appliance)          # LLM 1회: 기능명 정규화("자동청소"→클린봇) · 다른 제품 감지   (understand.py)
    res = retrieve(query, manual_id, und.feature)      # SQLite 조회(에러코드·기능 표) + 벡터 검색                    (retrieval.py)
    stream_answer(query, res, history)                 # 웹 전용 프롬프트 + 최근 4턴 그대로 + 근거 → 답변 스트리밍
    sources_of(res)                                    # 출처 패널 데이터

멀티턴: 검색은 질문 원문으로만 하고, 답변 LLM의 messages에 최근 대화를 그대로 넣는다 (LangChain RunnableWithMessageHistory와 같은 방식).
질문을 다시 쓰는 단계는 새 증상을 직전 질문의 후속으로 오판해 합쳐 버려서 뺐다 (2026-09-18).
RAG를 통째로 바꾸려면 위 4개 함수의 입출력만 지키면 된다 (web/README.md).
"""
from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from siyeon.rag.retrieval import CONTEXT_CUTOFF, LGAirconRetriever, RetrievalResult, appliance_label  # noqa: E402
from siyeon.rag.understand import Understanding, understand  # noqa: E402
from .rag_listwise import ListwiseRetriever  # noqa: E402

SUPPORT = {   # 고객센터 연결 버튼 (브랜드별)
    "lg":      {"name": "LG전자 고객센터", "phone": "1544-7777", "url": "https://www.lge.co.kr/support"},
    "samsung": {"name": "삼성전자 서비스", "phone": "1588-3366", "url": "https://www.samsung.com/sec/support/"},
}
ANSWER_MODEL = "gpt-4o-mini"
ANSWER_TEMPERATURE = 0.2
HISTORY_MESSAGES = 8      # 답변 LLM에 그대로 넣는 최근 메시지 수 (4턴). "그래도 안 돼요" 같은 후속은 LLM이 대화 맥락으로 답한다

# 웹 전용 시스템 프롬프트 — 팀 채점용 SYSTEM_PROMPT(pipeline/evaluate.py)는 채점 비교 기준이라 그대로 두고, 서비스는 이걸 쓴다.
# 실사용 로그에서 확인된 답변 결함 3종을 겨눈다:
#   A. 근거 3개 중 3번째가 정답인데 "설명서에 없습니다"라고 포기  → 규칙 2
#   B. 다른 증상 조각의 조치를 가져다 자신 있게 오답              → 규칙 2·3
#   C. 설명서 문장을 바꿔 쓰다 뜻이 뒤집힘("높게 설정돼 있는지 확인… 높여보세요") → 규칙 4
WEB_SYSTEM_PROMPT = """당신은 사용자가 등록한 가전제품의 사용설명서를 대신 읽어주는 안내 챗봇입니다. 아래 규칙을 반드시 지킵니다.

1. 답은 [근거]에 적힌 설명서 내용만으로 합니다. 설명서 밖의 일반 상식, 다른 제품의 통상적인 방법, 추정한 수치는 쓰지 않습니다.
2. [근거]가 여러 개면 먼저 각 근거의 제목(증상: … / 소제목)과 질문을 비교해 질문에 맞는 근거를 고르세요. 맞는 근거가 하나라도 있으면
   그것으로 답합니다 — 다른 근거가 무관하다는 이유로 "설명서에 없다"고 하지 마세요. 반대로 맞는 근거가 하나도 없으면
   "설명서에서 해당 내용을 찾지 못했습니다"라고만 하고, 실제로 받은 근거의 제목이 있을 때만 그 이름을 짧게 언급합니다
   ([근거] 없음이면 항목 이름을 지어내지 않습니다).
3. 고른 근거의 조치만 안내합니다. 다른 증상 근거의 조치를 섞어 넣지 마세요.
4. 설명서 문장의 뜻을 바꾸지 마세요. 특히 높이다/낮추다, 켜다/끄다, 열다/닫다 같은 방향, 버튼 이름, 수치·시간·온도는 설명서 그대로 옮깁니다.
   여러 모델 공용 설명서라 값이 여러 개면 스스로 고르지 말고 모델별로 나열합니다.
5. 형식: 첫 줄에 결론 한 문장 → 확인·조치를 설명서 순서대로 번호 목록. 증상·고장 질문이면 마지막에 "그래도 해결되지 않으면
   고객센터에 문의하세요" 한 줄을 붙이고, 사용법·기능 유무·설명서에 없는 내용 답변에는 붙이지 않습니다.
   전체 5~10줄. 각 단계는 설명서의 표현을 살려 구체적으로 (예: "리모컨의 밝기 버튼을 누를 때마다 ON/OFF로 바뀝니다").
6. 화재·감전·연기·타는 냄새·가스처럼 안전과 관련된 질문이면 첫 줄에 사용 중지와 전원 차단(플러그 분리)을 먼저 안내합니다.
7. 이전 대화가 있으면 그 맥락을 반영합니다. 이미 안내한 내용은 반복하지 말고 새로 물은 것에만 답하고, 사용자의 상황 진술
   ("건전지는 새로 갈았어요", "그래도 안 돼요")은 그 항목을 이미 확인한 것으로 보고 다음 확인 사항으로 넘어갑니다.
   단, 질문이 새 증상·새 주제면 이전 대화와 섞지 말고 그 질문에만 답합니다.
8. 존댓말, 간결하게. 인사말·사족 없이 바로 답합니다."""

_retriever: LGAirconRetriever | None = None
_lock = threading.Lock()


def get_retriever() -> LGAirconRetriever:
    """첫 호출 때 로딩(임베딩 모델·인덕스, GPU 기준 ~50초). 이후 재사용.

    ListwiseRetriever(rag_listwise.py) - CrossEncoder 리랭킹을 listwise LLM
    리랭킹으로 교체한 버전. CrossEncoder를 더 이상 안 써서 그 모델 로딩/워밍업이
    필요 없다(첫 질문 응답 시 gpt-4o-mini 호출로 자연히 워밍업됨)."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                from siyeon.rag.chunking_lg import build_all_chunks
                from siyeon.rag.chunking_samsung import build_samsung_chunks
                _retriever = ListwiseRetriever(
                    build_all_chunks() + build_samsung_chunks(), verbose=False, use_reranker=False,
                )
    return _retriever


def is_ready() -> bool:
    return _retriever is not None


def warm_in_background() -> None:
    """서버 시작 직후 호출 — 사용자가 채팅 화면에 도달하기 전에 로딩을 시작한다."""
    if _retriever is None:
        threading.Thread(target=get_retriever, name="rag-warmup", daemon=True).start()


def appliance_dict(a) -> dict:
    """DB의 user_appliances 행 → understand/retrieve가 쓰는 dict"""
    return {"brand": a.brand, "category": a.category, "product_type": a.product_type, "model": a.model, "manual_id": a.manual_id}


def understand_query(query: str, appliance) -> Understanding:
    """LLM 1회 — 기능명 정규화·다른 제품 감지. 실패하면 판단 없는 Understanding(둘 다 None)."""
    return understand(query, appliance_dict(appliance))


def retrieve(query: str, manual_id: str, feature: str | None = None, top_k: int = 3) -> RetrievalResult:
    return get_retriever().retrieve(query, doc_id=manual_id, top_k=top_k, context_cutoff=CONTEXT_CUTOFF, feature=feature)


def manual_pdf_path(doc_id: str) -> Path | None:
    """doc_id(PDF 파일명 stem) → data/{brand}/{category}/{doc_id}.pdf. 에러코드 공통 문서('LG-AC-COMMON')는 None."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id) or doc_id.endswith("-COMMON"):
        return None
    return next(iter((ROOT / "data").glob(f"*/*/{doc_id}.pdf")), None)


def _title_and_body(c: dict) -> tuple[str, str]:
    """증상 조각은 본문 첫 줄이 '증상: …' — 그걸 제목으로 떼어낸다 (LLM이 어느 근거가 어느 증상인지 바로 보게)."""
    body = c["body"]
    path = " › ".join(x for x in (c.get("section"), c.get("subsection")) if x)
    if body.startswith("증상:"):
        first, _, rest = body.partition("\n")
        return f"{path} — {first.strip()}", rest.strip()
    return path, body


def sources_of(res: RetrievalResult) -> list[dict]:
    """출처 패널용. LLM에 실제로 넘어간(used) 조각만. page_start/page_end: 청킹의 assign_pages. body: 전문(모달)."""
    out = []
    for c, d in res.used:
        tag = {"error_code": "오류코드 안내", "symptom": "고장 신고 전 확인", "safety": "안전 주의사항",
               "feature_absent": "기능 유무"}.get(c.get("tag"), "사용설명서")
        title, _ = _title_and_body(c)
        doc_id = c.get("doc_id", "")
        out.append({"title": title or doc_id, "snippet": c["body"][:220], "body": c["body"], "tag": tag, "doc_id": doc_id,
                    "distance": round(float(d), 3), "chunk_id": c.get("id", ""),
                    "page_start": c.get("page_start"), "page_end": c.get("page_end"),
                    "pdf_url": f"/api/manuals/{doc_id}/pdf" if manual_pdf_path(doc_id) else None})
    return out


def build_web_context(res: RetrievalResult) -> str:
    """답변 LLM의 user 메시지 — 등록 가전, 번호 매긴 근거(제목 분리), 경로별 안내."""
    head = f"[사용자 가전] {appliance_label(res.appliance)}"
    if not res.used:
        return head + "\n\n[근거] 없음 — 이 모델의 설명서에서 질문과 관련 있는 내용을 찾지 못했습니다. 설명서에 없다고 간단히 안내하고 추측으로 답하지 마세요."
    if res.route == "feature_absent":
        return head + f"\n\n[근거] {res.used[0][0]['body']}\n(모델별 기능 표 기준 확정 정보입니다. 이 모델에 그 기능이 없다고 분명히 답하세요.)"
    parts = []
    for i, (c, _) in enumerate(res.used, 1):
        title, body = _title_and_body(c)
        src = "오류코드 표" if c.get("source") == "rdb" else "설명서"
        parts.append(f"[근거 {i}] ({src}) {title}\n{body}")
    return head + "\n\n" + "\n\n".join(parts)


def stream_answer(query: str, res: RetrievalResult, history: list[dict] | None = None):
    """gpt-4o-mini 스트리밍. history = 최근 메시지 [{role, content}]를 그대로 messages에 넣는다."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    for m in (history or [])[-HISTORY_MESSAGES:]:
        messages.append({"role": m["role"], "content": m["content"][:800]})
    messages.append({"role": "user", "content": f"{build_web_context(res)}\n\n[질문]\n{query}"})
    stream = client.chat.completions.create(model=ANSWER_MODEL, temperature=ANSWER_TEMPERATURE, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
