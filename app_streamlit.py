# -*- coding: utf-8 -*-
"""가전제품 RAG 데모용 Streamlit 채팅 앱.

사이드바에서 제품(카테고리 -> 모델)을 선택하면, 그 모델 문서로 검색 범위를
제한한 채팅창이 뜬다. 실제 답변이 얼마나 자연스럽게/이쁘게 나오는지 눈으로
확인하기 위한 데모용이라, 근거 문서(거리/섹션)와 참고 그림도 같이 보여준다.

실행: streamlit run app_streamlit.py
"""
import time

import streamlit as st

import config
from pipeline.embed_store import get_collection
from pipeline.query_engine import answer

st.set_page_config(page_title="가전제품 AI 상담", page_icon="🔧", layout="wide")


@st.cache_data(show_spinner=False)
def load_products() -> dict[str, list[tuple[str, str]]]:
    """카테고리별 (모델명, 제품명) 목록. Chroma 메타데이터에서 직접 뽑는다
    (get_known_models()는 모델명만 주고 카테고리/제품명은 안 줘서 여기선 직접 조회)."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    by_category: dict[str, dict[str, str]] = {}
    for m in result["metadatas"]:
        model, category, name = m.get("product_model"), m.get("category"), m.get("product_name")
        if not model or not category:
            continue
        by_category.setdefault(category, {})[model] = name or model
    return {
        cat: sorted(models.items(), key=lambda kv: kv[0])
        for cat, models in sorted(by_category.items())
    }


PRODUCTS = load_products()

with st.sidebar:
    st.header("제품 선택")
    if not PRODUCTS:
        st.error("벡터DB에 적재된 문서가 없습니다. 먼저 ingest 스크립트를 실행하세요.")
        st.stop()

    category = st.selectbox("카테고리", list(PRODUCTS.keys()))
    model_options = PRODUCTS[category]
    labels = [f"{model} ({name})" for model, name in model_options]
    idx = st.selectbox(
        "모델", range(len(model_options)), format_func=lambda i: labels[i]
    )
    selected_model, selected_name = model_options[idx]

    st.caption(f"선택된 제품: **{selected_name}**")
    st.divider()
    st.caption(f"임베딩: {config.EMBEDDING_BACKEND} ({config.LOCAL_EMBEDDING_MODEL if config.EMBEDDING_BACKEND == 'local' else config.EMBEDDING_MODEL})")
    st.caption(f"답변 생성: {config.CHAT_BACKEND} ({config.OLLAMA_MODEL if config.CHAT_BACKEND == 'ollama' else (config.LOCAL_CHAT_MODEL if config.CHAT_BACKEND == 'local' else config.CHAT_MODEL)})")
    st.caption("로컬 모델 특성상 답변까지 수십 초 걸릴 수 있습니다.")

    if st.button("대화 초기화"):
        st.session_state.pop(f"messages_{selected_model}", None)
        st.rerun()

st.title("🔧 가전제품 사용·문제해결 상담")
st.caption(f"현재 상담 제품: {selected_name} ({selected_model})")

msg_key = f"messages_{selected_model}"
if msg_key not in st.session_state:
    st.session_state[msg_key] = []
messages = st.session_state[msg_key]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("candidates"):
            with st.expander("근거 문서 보기"):
                for c in msg["candidates"]:
                    meta = c["metadata"]
                    st.markdown(
                        f"- **{meta.get('section_title', '(제목 없음)')}** "
                        f"· 유사도 거리 {c['distance']:.3f} · {meta.get('source', 'vector')}"
                    )
            figures = msg.get("figures") or []
            if figures:
                st.caption("관련 그림")
                cols = st.columns(min(len(figures), 4))
                for i, fig_path in enumerate(figures[:4]):
                    try:
                        cols[i % len(cols)].image(fig_path, use_container_width=True)
                    except Exception:
                        pass

query = st.chat_input("궁금한 점을 입력하세요 (예: 탈수가 안 돼요, F4 에러 떠요)")
if query:
    messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중... (로컬 모델이라 다소 시간이 걸립니다)"):
            t0 = time.time()
            result = answer(query, top_k=3, model_filter=selected_model)
            elapsed = time.time() - t0
        st.markdown(result["answer"])
        st.caption(f"⏱ {elapsed:.1f}초 · 검색 방식: {result['source']}")

        figures = []
        if result["source"] == "vector":
            for c in result["candidates"]:
                for f in c["metadata"].get("figures", []) or []:
                    if f not in figures:
                        figures.append(f)
            with st.expander("근거 문서 보기"):
                for c in result["candidates"]:
                    meta = c["metadata"]
                    st.markdown(
                        f"- **{meta.get('section_title', '(제목 없음)')}** "
                        f"· 유사도 거리 {c['distance']:.3f}"
                    )
            if figures:
                st.caption("관련 그림")
                cols = st.columns(min(len(figures), 4))
                for i, fig_path in enumerate(figures[:4]):
                    try:
                        cols[i % len(cols)].image(fig_path, use_container_width=True)
                    except Exception:
                        pass
        else:
            with st.expander("근거 문서 보기 (RDB 정확 조회)"):
                for c in result["candidates"]:
                    st.markdown(f"- **{c['metadata'].get('section_title')}** (코드: {c['metadata'].get('aliases')})")

    messages.append({
        "role": "assistant",
        "content": result["answer"],
        "candidates": result["candidates"],
        "figures": figures,
    })
