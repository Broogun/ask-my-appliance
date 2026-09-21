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

from siyeon.rag.retrieval import CONTEXT_CUTOFF, RetrievalResult, appliance_label  # noqa: E402
from siyeon.rag.understand import Understanding, understand  # noqa: E402
from . import rag_myretriever  # noqa: E402
from .manual_pages import manual_pdf_path  # noqa: E402,F401  (라우터들이 여기서 import)
import 박형건_exp02 as exp02  # noqa: E402

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
   길이는 고정하지 않습니다 - [근거]에 실제로 담긴 내용만큼 씁니다. [근거]에 원인이 서로 다른 확인 항목이 여러 개
   있으면(예: 근거 안에 "세탁물이 치우쳤나요?"/"양이 적나요?"/"분류를 안 했나요?"처럼 별개의 하위 증상이 나열돼
   있으면) 그걸 하나로 뭉뚱그리지 말고 각각 번호를 따로 매겨서, 항목마다 "왜 그런지" 이유를 한 줄 덧붙이세요.
   아래 두 예시를 비교하세요 - 왼쪽처럼 뭉뚱그리지 말고 오른쪽처럼 [근거]에 있는 항목 수만큼 풀어 쓰세요:
   나쁜 예(항목을 하나로 뭉침):
     "탈수가 안 되는 원인은 세탁물이 치우쳐서입니다.
     1. 세탁물을 고르게 펴주세요.
     2. 종류별로 나누어 탈수하세요."
   좋은 예(근거에 있는 항목 수만큼, 이유 포함):
     "탈수 시 소음이 크고 잘 안 도는 건 대부분 세탁물 무게가 한쪽으로 쏠려서입니다.
     1. 세탁물이 한쪽으로 치우쳤나요? 치우치면 소음·진동이 커져 자동으로 회전수를 낮추므로 탈수가 덜 될 수
        있습니다. 세탁물을 고르게 펴서 한 번 더 탈수하세요.
     2. 세탁물 양이 너무 적나요? 양이 적으면 세탁물이 한쪽으로 뭉쳐 같은 문제가 생깁니다. 세탁물 몇 개를
        더 모아 탈수하세요.
     3. 세탁물을 종류별로 나누지 않았나요? 종류가 섞이면 무게 불균형이 커집니다. 종류별로 나누어 탈수하세요."
   근거가 하나뿐이거나 정말 단순한 질문(사용법 한 단계 등)이면 짧게 답해도 됩니다 - 위 규칙은 "근거에 있는 내용을
   누락하지 말라"는 뜻이지 "무조건 길게 쓰라"는 뜻이 아닙니다. 근거에 실제로 나열된 예시(구체적 물건 이름, 수치
   등)가 있으면 그 예시도 같이 적으세요 - 근거에 없는 예시를 지어내지는 마세요. 각 단계는 설명서의 표현을 살려
   구체적으로 (예: "리모컨의 밝기 버튼을 누를 때마다 ON/OFF로 바뀝니다").
6. "사용 중지와 전원 차단(플러그 분리)을 먼저 하세요" 같은 안전 경고 문구는 당신이 쓰지 않습니다 - 시스템이
   필요할 때 답변 앞에 자동으로 붙입니다. 질문이나 근거가 얼마나 심각하게 들리든(소음이 심하다, 잠을 못 잔다 등)
   당신은 이 문구를 절대 스스로 만들어 붙이지 말고, 바로 확인 사항/조치 목록부터 시작하세요.
7. 이전 대화가 있으면 그 맥락을 반영합니다. 이미 안내한 내용은 반복하지 말고 새로 물은 것에만 답하고, 사용자의 상황 진술
   ("건전지는 새로 갈았어요", "그래도 안 돼요")은 그 항목을 이미 확인한 것으로 보고 다음 확인 사항으로 넘어갑니다.
   단, 질문이 새 증상·새 주제면 이전 대화와 섞지 말고 그 질문에만 답합니다.
8. 존댓말, 간결하게. 인사말·사족 없이 바로 답합니다."""

_retriever: dict | None = None
_lock = threading.Lock()


def get_retriever() -> dict:
    """첫 호출 때 로딩(exp02 임베딩 모델, GPU 기준 ~1분). 이후 재사용.

    2026-09-21 검색 엔진을 exp02(청킹+벡터검색+listwise 리랭킹, rag_myretriever.py)로
    교체 - 실사용 로그 감사 중 시연님 RRF 파이프라인(n_candidates=8)이 실제 정답
    청크를 놓치는 사례(세탁기 탈수 소음 질문)를 발견했고, 같은 질문을 exp02로
    돌리면 바로 잡아왔다. RDB(에러코드/모델매핑/기능유무)는 시연님 것을 그대로
    쓴다(rag_myretriever.py가 SQLite만 가볍게 재사용 - 시연님의 무거운
    LGAirconRetriever는 이제 안 씀. CrossEncoder/expected-question 인덕스 로딩이
    전부 필요 없어져서 로딩 자체도 이전보다 가벼워짐)."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = exp02._load_state()
    return _retriever


def is_ready() -> bool:
    return _retriever is not None


def embed_texts(texts: list[str]):
    """답변 그림 매칭의 의미 유사도 보조 단계용 - 검색에 쓰는 것과 같은 로컬 임베딩 모델(정규화된 벡터)."""
    return exp02._load_state()["embed_model"].encode(texts, normalize_embeddings=True, show_progress_bar=False)


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


def retrieve(query: str, manual_id: str, feature: str | None = None, top_k: int = 5) -> RetrievalResult:
    get_retriever()  # exp02 임베딩 모델 워밍업 보장(최초 호출 시 로딩)
    return rag_myretriever.retrieve(query, manual_id=manual_id, top_k=top_k, feature=feature)


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
                    "pdf_url": f"/api/manuals/{doc_id}/pdf" if manual_pdf_path(doc_id) else None,
                    "images": c.get("images") or []})   # 에러코드 항목에 연결된 제조사 사진(있을 때만)
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
        # 청킹 단계에서 "안전을 위해 주의하기" 챕터는 tag="safety"로 이미 구조적으로
        # 표시돼 있다(chunking_lg.py/chunking_samsung.py) - 질문 문구만 보고 LLM이
        # "이게 위험 상황인가"를 매번 판단하게 하면 "소음"류 무해한 질문에도 안전
        # 경고를 잘못 붙이는 오탐이 실제로 발생함(실측 확인, 2026-09-21). 안전 경고를
        # 붙일지 여부는 실제로 안전 챕터가 검색됐는지로 결정론적으로 표시해준다.
        tag_mark = "[⚠ 안전 경고 챕터] " if c.get("tag") == "safety" else ""
        parts.append(f"[근거 {i}] ({src}) {tag_mark}{title}\n{body}")
    return head + "\n\n" + "\n\n".join(parts)


SAFETY_WARNING_LINE = "사용 중지와 전원 차단(플러그 분리)을 먼저 하세요.\n\n"

# LLM이 규칙 6(안전 경고 문구를 스스로 만들지 말 것)을 무시하고 첫 문장에
# 이 문구를 또 만들어내는 걸 반복 확인함(2026-09-21, 마커+명시적 금지 지시
# 3차례 시도 후에도 재발 - 같은 질문 4회 연속 재현에서 후보 선정은 완전히
# 동일했는데도 생성 답변에만 등장한 걸로 봐서 검색/코드 문제가 아니라 생성
# 단계에서 LLM이 규칙을 어기는 것으로 확정). 프롬프트로 더 못 이기므로 생성된
# 첫 문장을 코드로 검사해서, 진짜 안전 챕터가 없는데 이 패턴이 나오면 잘라낸다.
_SAFETY_PHRASE_RE = re.compile(r"전원\s*(을|를)?\s*(차단|끄|뽑)|사용\s*(을|를)?\s*중지|플러그\s*(를|을)?\s*(뽑|분리)")

# 경고 줄은 "지금 사용자 상황이 위험하니 멈추라"는 지시라서, 근거 청크에 '화재/감전' 같은
# 단어가 있다는 것만으로는 부족하다 - 300문항 baseline에서 그 기준이었더니 청소/필터/냉방
# 같은 무해한 질문의 47%(141/300)에 붙었다(2026-09-21). 질문 자체에 위험 신호가 있을
# 때만 붙인다. '불이 안 켜져요'(표시등)는 위험이 아니라서 '불이 나/났'만 잡는다.
_HAZARD_QUERY_RE = re.compile(
    r"타는|탄\s*내|연기|불꽃|불이\s*(나|났|붙)|불나|화재|발화|스파크|폭발|과열|"
    r"감전|찌릿|누전|합선|"
    r"(가스|냉매)\s*(냄새|샌|새|누출)|"
    r"젖은|물이\s*(들어|튀|닿|샜|새)|침수|물에\s*(젖|잠)|"
    r"(전원선|전원\s*코드|플러그)[^.?!]{0,8}(손상|벗겨|녹|뜨겁|뜨거|피복)"
)


def needs_safety_warning(query: str, res: RetrievalResult) -> bool:
    """근거에 안전 챕터가 있고(매뉴얼이 뒷받침) + 질문에 위험 신호가 있을 때만."""
    return any(c.get("tag") == "safety" for c, _ in res.used) and bool(_HAZARD_QUERY_RE.search(query))


def stream_answer(query: str, res: RetrievalResult, history: list[dict] | None = None):
    """gpt-4o-mini 스트리밍. history = 최근 메시지 [{role, content}]를 그대로 messages에 넣는다.

    안전 경고 문구는 프롬프트 지시로 LLM에게 맡기지 않고 코드가 직접 붙인다 -
    "[⚠ 안전 경고 챕터]" 마커를 컨텍스트에 넣고 규칙으로 지시해도(마커 있을 때만
    붙여라/절대 자체 판단하지 마라를 3차례 다르게 시도), "소음" 같은 무해한 증상
    질문에도 LLM이 자체적으로 안전 경고를 계속 붙이는 걸 반복 확인함(2026-09-21) -
    실제 안전 챕터(tag=="safety")가 검색됐을 때만 앞에 붙이고, LLM 자체 생성분의
    첫 문장도 같은 패턴이면 코드로 잘라낸다(아래 _SAFETY_PHRASE_RE)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    for m in (history or [])[-HISTORY_MESSAGES:]:
        messages.append({"role": m["role"], "content": m["content"][:800]})
    messages.append({"role": "user", "content": f"{build_web_context(res)}\n\n[질문]\n{query}"})

    is_safety = needs_safety_warning(query, res)
    if is_safety:
        yield SAFETY_WARNING_LINE

    stream = client.chat.completions.create(model=ANSWER_MODEL, temperature=ANSWER_TEMPERATURE, messages=messages, stream=True)
    buf, checked = "", False
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if checked:
            yield delta
            continue
        buf += delta
        m = re.search(r"[.!?。]\s*|\n", buf)
        if m is None and len(buf) < 80:
            continue  # 첫 문장이 끝나거나 80자 넘을 때까지 계속 버퍼링
        checked = True
        if m and not is_safety and _SAFETY_PHRASE_RE.search(buf[:m.end()]):
            buf = buf[m.end():].lstrip()  # 규칙 위반 문장 통째로 삭제
        if buf:
            yield buf
