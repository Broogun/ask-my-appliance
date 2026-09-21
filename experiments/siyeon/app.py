"""검색·답변을 눈으로 확인하는 Streamlit 챗봇 (실험용 GUI).

실행 (프로젝트 루트에서):
    streamlit run experiments/siyeon/app.py

왼쪽에서 제조사 → 카테고리 → 모델을 고르면 "그 가전을 등록한 사용자"가 되고, 질문하면
`retrieve()` 라우터가 어느 경로로 답을 찾았는지(에러코드 DB 조회 / 기능 없음 룰 / 벡터 검색)와
LLM에 실제로 넘어간 근거 조각까지 그대로 보여준다.

처음 실행이면 임베딩 모델·리랭커 로딩에 1~2분 걸리고, 그 뒤로는 캐시된다(질문당 3~6초, 리랭커가 CPU라서).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
load_dotenv(ROOT / ".env")

from pipeline.evaluate import SYSTEM_PROMPT  # noqa: E402  (팀 공용 프롬프트, 수정 금지)
from siyeon.rag.chunking_lg import build_all_chunks  # noqa: E402
from siyeon.rag.chunking_samsung import build_samsung_chunks  # noqa: E402
from siyeon.rag.retrieval import CONTEXT_CUTOFF, LGAirconRetriever, RetrievalResult, apply_cutoff, build_context  # noqa: E402
from siyeon.rag.understand import understand  # noqa: E402

BRAND_KO = {"lg": "LG", "samsung": "삼성"}
DB_PATH = ROOT / "data" / "appliance.sqlite"
TESTSET = Path(__file__).with_name("testset.json")

st.set_page_config(page_title="가전 설명서 챗봇 (실험용)", page_icon="🔧", layout="wide")


# ── 준비 (앱 실행 중 한 번만) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="인덱스·모델 로딩 중 (처음 한 번만, 1~2분)…")
def load_retriever() -> LGAirconRetriever:
    chunks = build_all_chunks() + build_samsung_chunks()
    r = LGAirconRetriever(chunks, verbose=False)
    # 리랭커는 기본적으로 첫 사용 때 로딩된다(6~7초). 여기서 미리 깨워 첫 질문이 느려지지 않게 한다.
    r.reranker.predict([("워밍업", "리랭커 모델을 미리 로딩합니다.")])
    return r


@st.cache_data
def load_models() -> list[dict]:
    """매뉴얼 목록 = 드롭다운 원본. (brand, product_type, model, manual_id)"""
    if not DB_PATH.exists():
        st.error(f"{DB_PATH} 가 없습니다. 먼저 `python experiments/siyeon/rdb/build_db.py` 를 실행하세요.")
        st.stop()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT m.brand, m.product_type, m.manual_id,
                  (SELECT mm.model_pattern FROM manual_models mm
                    WHERE mm.manual_id = m.manual_id AND mm.source = 'filename'
                    ORDER BY length(mm.model_pattern) DESC LIMIT 1) AS model
             FROM manuals m ORDER BY m.brand, m.product_type, model"""
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@st.cache_data
def load_examples() -> dict[str, list[str]]:
    """자체 테스트셋을 모델별 예시 질문으로."""
    if not TESTSET.exists():
        return {}
    out: dict[str, list[str]] = {}
    for c in json.loads(TESTSET.read_text(encoding="utf-8"))["cases"]:
        if c.get("doc_id"):
            out.setdefault(c["doc_id"], []).append(c["query"])
    return out


def answer_with_llm(query: str, res: RetrievalResult) -> str:
    """검색된 근거로 gpt-4o-mini 답변. 프롬프트 구성은 retrieval.build_context (실험 스크립트와 동일)."""
    from openai import OpenAI

    context = build_context(res.used, res.appliance)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"[검색된 문서]\n{context}\n\n[질문]\n{query}"}],
        stream=True,
    )
    return st.write_stream(chunk.choices[0].delta.content or "" for chunk in resp)


ROUTE_BADGE = {
    "error_code":     ("DB 조회 — 에러코드 표에서 바로 찾음", "green"),
    "feature_absent": ("DB 조회 — 이 모델에 없는 기능 (모델별 기능 표 / 매뉴얼 언급 0회)", "orange"),
    "vector":         ("벡터 검색 + 리랭커", "blue"),
    "none":           ("근거 없음 — 설명서에서 찾지 못함", "gray"),
    "followup":       ("후속 질문 — 직전 답변의 근거로 이어서 답함", "teal"),
    "base":           ("기본 벡터 검색 (리랭커 없음)", "gray"),
    "dense":          ("벡터 검색 + 리랭커 (이해·DB 조회 없음)", "blue"),
}


def badge_of(res: RetrievalResult, mode: str) -> tuple[str, str]:
    if not res.hits:
        return "검색 결과 없음", "gray"
    key = res.route if mode == "v3" else mode
    label, color = ROUTE_BADGE.get(key, (key, "gray"))
    if mode == "v3" and not res.grounded and res.route == "vector":
        label, color = ROUTE_BADGE["none"]
    return label, color


# ── 사이드바: 가전 선택 + 옵션 ────────────────────────────────────────────────
models = load_models()
examples = load_examples()

with st.sidebar:
    st.header("등록한 가전")
    brands = sorted({m["brand"] for m in models}, key=lambda b: b != "lg")
    brand = st.selectbox("제조사", brands, format_func=lambda b: BRAND_KO.get(b, b))

    cats = sorted({m["product_type"] for m in models if m["brand"] == brand})
    category = st.selectbox("카테고리", cats)

    cand = [m for m in models if m["brand"] == brand and m["product_type"] == category]
    picked = st.selectbox("모델명", cand, format_func=lambda m: m["model"])
    manual_id = picked["manual_id"]
    st.caption(f"매뉴얼 문서: `{manual_id}`")

    st.divider()
    st.header("검색 옵션")
    MODES = {
        "① 최종 — 기능명·다른 제품 판정(LLM) + DB 조회 + 검색 + 리랭커": "v3",
        "② 검색 + 리랭커 (이해·DB 조회 없음)": "dense",
        "③ 기본 검색만 (가장 단순)": "base",
    }
    mode_label = st.radio("검색 방식", list(MODES), help="같은 질문을 방식만 바꿔 던져보면 각 단계가 무엇을 개선했는지 보입니다.")
    mode = MODES[mode_label]
    with st.expander("세 방식이 뭐가 달라요?"):
        st.markdown(
            """세 방식은 **포함 관계**입니다. ③에 기능을 더한 게 ②, ②에 더한 게 ①.

**③ 기본 검색만** — 질문과 설명서 조각을 각각 숫자(벡터)로 바꿔서 가까운 것 3개를 고릅니다.
빠르지만(0.1초) `CH05 에러` 같은 질문에 엉뚱한 조각이 1등으로 옵니다.

**② + 리랭커** — 여기에 세 가지를 더합니다.
- 조각마다 미리 만들어 둔 **"예상 질문"** 과도 비교 (`말로 조작` ↔ `음성인식`처럼 말이 달라도 찾음)
- 질문에 코드가 있으면 **키워드 검색**도 함께
- 후보 8개를 **리랭커**가 질문과 나란히 정독해서 3개만 선별
→ 느려지지만(3~6초) 정확도가 크게 오릅니다 (자체 테스트 top-1 83% → 88%).

**① 최종** — 여기에 **LLM 질문 이해 1회**(후속 질문을 완결 문장으로 재작성, "자동청소"→클린봇 같은 기능명 정규화, 다른 제품 감지)와
**검색 전에 DB를 먼저 조회하는 2단계**를 더합니다.
- 질문에 **에러코드**(CH05, UE …)가 있으면 SQLite에서 바로 꺼냅니다 → 0초, 정확
- 질문에 **기능명**(클린봇 …)이 있으면 "이 모델에 그 기능이 있나" 표를 봅니다 → 없으면 검색 없이 "없는 기능"이라 답
→ 실제 서비스에 쓸 방식이고 기본값입니다 (top-1 90%).

답변 아래 배지에 이번 질문이 ①~③ 중 **어느 경로로 처리됐는지** 표시됩니다."""
        )
    top_k = st.slider("근거 조각 수 (top_k)", 1, 8, 3)
    with st.container(border=True):
        st.caption("⚡ 속도 조절 — 리랭커(CPU)가 검색 시간의 대부분입니다")
        n_cand = st.select_slider("리랭커가 검토할 후보 수", [3, 5, 8], value=8,
                                  help="8개가 가장 정확했고(자체 테스트 기준), 5개면 1.5초쯤 빨라집니다.")
        rr_len = st.select_slider("리랭커 입력 길이", [256, 384], value=384,
                                  help="256으로 줄이면 긴 조각의 뒷부분을 덜 읽는 대신 빨라집니다.")
        st.caption({(8, 384): "약 3.8초", (5, 384): "약 2.3초", (8, 256): "약 2.3초",
                    (5, 256): "약 1.5초", (3, 384): "약 1.1초", (3, 256): "약 1.0초"}.get((n_cand, rr_len), ""))
    use_cut = st.checkbox(f"관련도 낮은 근거 제외 (distance ≥ {CONTEXT_CUTOFF})", value=True,
                          help="끄면 관련 없는 조각도 LLM에 넘어가 엉뚱한 근거로 답할 수 있습니다.")
    use_llm = st.checkbox("LLM 답변 생성", value=True, help="끄면 검색 결과만 봅니다 (빠르고 API 비용 없음).")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

retriever = load_retriever()

# ── 본문 ─────────────────────────────────────────────────────────────────────
st.title("🔧 가전 설명서 챗봇")
st.caption(f"현재 가전: **{BRAND_KO.get(brand, brand)} {category} {picked['model']}** — "
           f"이 모델의 설명서 조각과 해당 브랜드·제품군 에러코드만 검색합니다.")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)

# 이 모델의 예시 질문 (자체 테스트셋에서)
if examples.get(manual_id):
    with st.expander(f"예시 질문 {len(examples[manual_id])}개 (자체 테스트셋)"):
        cols = st.columns(2)
        for i, q in enumerate(examples[manual_id]):
            if cols[i % 2].button(q, key=f"ex_{manual_id}_{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            if msg.get("target"):
                st.caption(msg["target"])
        else:
            st.markdown(msg["content"])
            if msg.get("hits") is not None:
                badge, color = msg["badge"]
                st.caption(f":{color}[{badge}]  ·  {msg['elapsed']:.1f}초  ·  근거 {len(msg['hits'])}개"
                           f"{' (LLM 전달 ' + str(msg['n_used']) + '개)' if msg.get('n_used') is not None else ''}")
                with st.expander("검색된 근거 조각 보기"):
                    for rank, (c, d) in enumerate(msg["hits"], start=1):
                        used = "✅ LLM 전달" if rank <= msg.get("n_used", 0) or not use_cut else "⬜ 컷 제외"
                        page = (f" · 📄 p.{c['page_start']}" + (f"–{c['page_end']}" if c.get("page_end") != c.get("page_start") else "")
                                if c.get("page_start") else "")
                        st.markdown(f"**{rank}. `{d:.3f}`** · {used} · `{c['doc_id']}`{page} · "
                                    f"{'🗄️ RDB' if c.get('source') == 'rdb' else '🔍 벡터'} · `{c.get('tag', '')}`")
                        st.markdown(f"**{c['section']} › {c['subsection']}**")
                        st.text(c["body"][:1200] + ("…" if len(c["body"]) > 1200 else ""))
                        st.divider()

query = st.chat_input("질문을 입력하세요 (예: 필터 청소는 얼마나 자주 해요?)") or st.session_state.pending
st.session_state.pending = None

if query:
    target = f"{BRAND_KO.get(brand, brand)} {category} {picked['model']}"
    st.session_state.messages.append({"role": "user", "content": query, "target": f"대상: {target}"})
    with st.chat_message("user"):
        st.markdown(query)
        st.caption(f"대상: {target}")

    with st.chat_message("assistant"):
        t0 = time.time()
        retriever.reranker.max_length = rr_len
        cutoff = CONTEXT_CUTOFF if use_cut else None
        # 웹(chat_router)과 같은 경로: LLM 질문 이해(기능명 정규화·다른 제품 감지) → 라우터. 검색은 질문 원문으로
        appl = retriever.appliance_for(manual_id)
        und = None
        with st.spinner("검색 중…"):
            if mode == "v3":
                und = understand(query, appl)
                res = retriever.retrieve(query, doc_id=manual_id, top_k=top_k, n_candidates=n_cand, context_cutoff=cutoff, feature=und.feature)
            else:
                hits = (retriever.search_dense(query, manual_id, top_k=top_k, n_candidates=n_cand) if mode == "dense"
                        else retriever.search_baseline(query, doc_id=manual_id, top_k=top_k))
                res = RetrievalResult(hits=hits, route="vector", used=apply_cutoff(hits, cutoff), appliance=appl)
        if und is not None and und.feature:
            st.caption(f"기능 질문으로 이해: {und.feature}")
        if und is not None and und.other_product:
            st.warning(f'"{und.other_product}"는 등록된 제품이 아니라서 설명서 근거로 답하기 어려워요. (웹에서는 여기서 등록 안내로 갈라집니다)')
        t_search = time.time() - t0
        hits, used = res.hits, res.used

        if use_llm:
            if not os.getenv("OPENAI_API_KEY"):
                answer = "`.env`에 `OPENAI_API_KEY`가 없어서 답변을 생성하지 못했습니다. 검색 결과만 아래에서 확인하세요."
                st.warning(answer)
            else:
                try:
                    answer = answer_with_llm(query, res)
                except Exception as e:
                    answer = f"LLM 호출 실패: {type(e).__name__} — {e}"
                    st.error(answer)
        else:
            answer = "_(LLM 답변 생성이 꺼져 있습니다. 아래 근거 조각만 확인하세요.)_"
            st.markdown(answer)

        elapsed = time.time() - t0
        badge, color = badge_of(res, mode)
        st.caption(f":{color}[{badge}]  ·  검색 {t_search:.1f}초 + 답변 {elapsed - t_search:.1f}초 = **{elapsed:.1f}초**"
                   f"  ·  근거 {len(hits)}개 (LLM 전달 {len(used)}개)")
        with st.expander("검색된 근거 조각 보기", expanded=not use_llm):
            for rank, (c, d) in enumerate(hits, start=1):
                mark = "✅ LLM 전달" if (c, d) in used else "⬜ 컷 제외"
                st.markdown(f"**{rank}. `{d:.3f}`** · {mark} · `{c['doc_id']}` · "
                            f"{'🗄️ RDB' if c.get('source') == 'rdb' else '🔍 벡터'} · `{c.get('tag', '')}`")
                st.markdown(f"**{c['section']} › {c['subsection']}**")
                st.text(c["body"][:1200] + ("…" if len(c["body"]) > 1200 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "hits": hits, "n_used": len(used),
        "badge": (badge, color), "elapsed": elapsed,
    })
