"""박형건 exp01 — TOC 기반 청킹 + BGE-m3-ko(GPU) + Chroma + 에러코드 RDB +
질문 재구성(decomposition) + 리랭커 결합 검색.

`experiments/rag_tutorial.ipynb`에서 단계별로(TOC 파싱 -> 중복 헤딩 처리 ->
문제해결 챕터 강제 세분화 -> TOC 없는 구형 PDF 페이지 폴백 -> GPU 임베딩 ->
RDB 구축 -> decomposition+리랭커 결합) 검증한 파이프라인을 그대로 옮긴 파일이다.
벡터 DB(`chroma_db/`)와 에러코드 DB(`data/error_codes.db`)는 노트북에서 이미
구축해 둔 걸 그대로 읽기만 한다 (재구축 안 함 - 60개 PDF GPU 임베딩은 노트북에서
한 번만 해도 충분).

사용법:
  python experiments/박형건_exp01.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import chromadb
import requests
import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.harness.evaluate import SYSTEM_PROMPT, run_and_save  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"
ERROR_DB_PATH = ROOT / "data" / "error_codes.db"

# ⚠️ 임시: OpenAI 프로젝트 지출 한도(spend limit) 초과로 로컬(Ollama)을 쓰고 있다.
# 한도가 풀리면 generate_local() 호출을 OpenAI(gpt-4o-mini) 호출로 되돌릴 것 -
# 통일 사항(gpt-4o-mini)은 최종 제출 실험에서는 반드시 지켜야 한다 (CLAUDE.md 참고).
# SYSTEM_PROMPT는 pipeline.evaluate에서 그대로 가져다 쓰고 있어 수정하지 않았다.
LOCAL_LLM_MODEL = "qwen3.5:9b"

# ── 실험 메타 정보 ────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "박형건_exp01"

NOTES = (
    "PDF 내장 TOC로 재귀 청킹(중복 헤딩은 조상 경로 접두어로 구분, 문제해결/고장진단 "
    "챕터는 키워드 매칭으로 max_depth 무시하고 강제 세분화) + TOC 자체가 없는 구형 "
    "PDF는 다단 레이아웃이라 안전하게 페이지 단위 폴백. 임베딩은 dragonkue/BGE-m3-ko"
    "(한국어 검색 벤치마크로 파인튜닝된 BGE-M3, GPU) 하나로 단일 Chroma 컬렉션에 "
    "brand/category/product_model 메타데이터와 함께 적재. 에러코드는 벡터 유사도보다"
    " 정확 매칭이 신뢰도가 높아 별도 SQLite RDB로 분리. 검색 시 질문에서 에러코드 "
    "토큰을 뽑아 RDB부터 확인하고, 항상 로컬 LLM으로 질문을 주제별 서브 질문으로 "
    "재구성(decomposition)한 뒤 서브 질문별 벡터 검색 결과를 합쳐서 "
    "bge-reranker-v2-m3-ko cross-encoder로 서브 질문 기준 재정렬한다."
)

STRATEGY = {
    "chunking": "TOC 재귀 분할(max_depth=2) + 중복 헤딩 조상경로 접두어 + 문제해결 키워드 강제 세분화 + 700자/15% 오버랩 슬라이싱, TOC 없는 문서는 페이지 단위 폴백",
    "chunking_reason": "매뉴얼이 이미 TOC로 의미 단위 구조화돼 있어 고정길이 분할보다 유리. 문제해결 챕터는 하위 항목이 뭉쳐 나오는 걸 실제 확인해서 강제 세분화 추가. TOC 없는 일부 구형 PDF는 인쇄된 목차가 다단 레이아웃이라 좌표 기반 파싱도 불안정해 정보 손실 없는 페이지 폴백을 선택.",
    "embedding": "dragonkue/BGE-m3-ko (GPU)",
    "embedding_reason": "BGE-M3(다국어)를 한국어 검색 벤치마크(Ko-StrategyQA, AutoRAGRetrieval, MIRACLRetrieval)로 파인튜닝한 로컬/무료 모델. GPU(RTX 4070)로 60개 PDF 5,700여 청크를 1분 내 임베딩 가능해서 반복 실험에 부담이 없음.",
    "db": "ChromaDB 단일 컬렉션(appliance_manuals) + SQLite 에러코드 RDB(error_solutions/error_codes)",
    "db_reason": "브랜드/카테고리를 특정하지 않는 질문도 많아서(COMMON_QUESTIONS 참고) 컬렉션을 6개로 쪼개는 대신 단일 컬렉션 + 메타데이터 필터를 선택. 에러코드는 근사 매칭(임베딩)보다 정확 매칭이 훨씬 신뢰도가 높아 벡터 DB와 별도로 관계형 스키마로 분리.",
    "retrieval": "질문에서 에러코드처럼 보이는 토큰을 뽑아 RDB 정확 매칭 -> 로컬 LLM으로 질문 재구성(decomposition, 복합 주제면 서브 질문으로 분리) -> 서브 질문별 벡터 검색(top_k=5) -> id 기준 중복 제거 -> bge-reranker-v2-m3-ko로 서브 질문 기준 재정렬해서 최종 top_k=3",
    "retrieval_reason": "에러코드 조회는 정확도가 생명이라 임베딩 유사도만으로는 부족해 RDB 정확 매칭을 우선함. 'UE 오류랑 필터 청소 방법 같이 알려줘' 같은 복합 질문은 임베딩 하나로 두 주제를 동시에 못 담아서 decomposition으로 주제를 분리. bi-encoder 검색만으로는 상위권 정밀도가 부족해 cross-encoder 리랭커로 보정 - 단, 리랭커를 원본 복합 질문 그대로 돌리면 변별력이 없어지는 걸 확인해서 서브 질문 단위로 재정렬하도록 조정함.",
}


# ════════════════════════════════════════════════════════════════════════════
# 모델/DB 로딩 (my_answer가 22번 호출되는 동안 한 번만 로드하도록 지연 초기화)
# ════════════════════════════════════════════════════════════════════════════

_state: dict = {}


def _load_state() -> dict:
    if _state:
        return _state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[박형건_exp01] 임베딩/리랭커 device: {device}")

    _state["embed_model"] = SentenceTransformer("dragonkue/BGE-m3-ko", device=device)
    _state["reranker"] = CrossEncoder("dragonkue/bge-reranker-v2-m3-ko", device=device)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _state["collection"] = client.get_collection("appliance_manuals")

    conn = sqlite3.connect(str(ERROR_DB_PATH))
    _state["conn"] = conn
    _state["all_codes"] = {
        row[0].upper(): row[0]
        for row in conn.execute("SELECT DISTINCT code FROM error_codes").fetchall()
    }
    return _state


def generate_local(system_prompt: str, user_message: str, model: str = LOCAL_LLM_MODEL) -> str:
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "think": False,  # think=True면 응답이 훨씬 느려짐 (노트북에서 확인함)
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ════════════════════════════════════════════════════════════════════════════
# RDB(에러코드) 조회
# ════════════════════════════════════════════════════════════════════════════

def find_error_codes_in_query(query: str, state: dict) -> list[str]:
    """질문에서 에러코드로 보이는 토큰(영문+숫자, 2자 이상)을 뽑아 RDB에 존재하는 것만 반환."""
    tokens = re.findall(r"[A-Za-z0-9()]{2,}", query)
    found = []
    for t in tokens:
        code = state["all_codes"].get(t.upper())
        if code and code not in found:
            found.append(code)
    return found


def lookup_code(conn: sqlite3.Connection, code: str) -> list[dict]:
    """코드 하나로 관련 조치법을 전부 조회한다 (브랜드가 달라도 같은 코드면 다 나옴)."""
    rows = conn.execute(
        """
        SELECT s.solution_id, s.brand, s.category, s.title, s.content
        FROM error_codes c JOIN error_solutions s ON c.solution_id = s.solution_id
        WHERE c.code = ?
        """,
        (code,),
    ).fetchall()
    return [
        {"solution_id": r[0], "brand": r[1], "category": r[2], "title": r[3], "content": r[4]}
        for r in rows
    ]


# ════════════════════════════════════════════════════════════════════════════
# 벡터 검색 + 질문 재구성(decomposition) + 리랭커
# ════════════════════════════════════════════════════════════════════════════

DECOMPOSE_SYSTEM_PROMPT = (
    "너는 사용자 질문을 검색하기 좋은 형태로 정리하는 어시스턴트야. "
    "질문에 서로 다른 주제(예: 오류코드 문의 + 사용법 문의)가 섞여 있으면 "
    "각 주제를 독립된 질문으로 분리해. 하나의 주제면 원래 질문 그대로 하나만 담아. "
    "반드시 JSON 배열만 출력해. 다른 설명이나 마크다운 없이 [\"질문1\", \"질문2\"] 형식만."
)


def decompose_query(query: str) -> list[str]:
    """복합 질문이면 주제별로 쪼개서 리스트로 반환, 아니면 [query] 그대로."""
    raw = generate_local(DECOMPOSE_SYSTEM_PROMPT, query)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [query]
    try:
        parts = json.loads(match.group(0))
        parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        return parts if parts else [query]
    except json.JSONDecodeError:
        return [query]


def _vector_search(query: str, state: dict, top_k: int, where: dict | None) -> list[dict]:
    query_emb = state["embed_model"].encode([query], normalize_embeddings=True).tolist()
    result = state["collection"].query(query_embeddings=query_emb, n_results=top_k, where=where)
    candidates = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        candidates.append({"text": doc, "metadata": meta, "distance": dist})
    return candidates


def _rerank(sub_queries: list[str], candidates: list[dict], state: dict, top_k: int) -> list[dict]:
    """cross-encoder로 각 후보를 서브 질문들과 비교하고, 후보당 최고점(max)으로 재정렬.

    원본(복합) 질문 그대로 rerank하면 cross-encoder가 복합 문장에는 거의 다 0점에
    가까운 점수를 매겨서 변별력이 없다 (노트북에서 실제로 확인함). 분해된 서브
    질문 각각과 비교해서 최고점을 쓰면 제대로 구분된다.
    """
    if not candidates:
        return candidates
    reranker = state["reranker"]
    for c in candidates:
        sub_scores = reranker.predict([[sq, c["text"]] for sq in sub_queries])
        c["rerank_score"] = float(max(sub_scores))
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]


def search_v2(
    query: str,
    state: dict,
    top_k: int = 3,
    where: dict | None = None,
    candidates_per_subquery: int = 5,
) -> list[dict]:
    """decomposition -> 서브 질문별 벡터 검색 -> 중복 제거 -> 서브 질문 기준 리랭커로 최종 top_k."""
    sub_queries = decompose_query(query)

    merged: dict[str, dict] = {}
    for sub_q in sub_queries:
        for c in _vector_search(sub_q, state, top_k=candidates_per_subquery, where=where):
            chunk_id = (
                c["metadata"]["product_model"] + "_" + c["metadata"]["section_title"] + str(c["metadata"]["page"])
            )
            if chunk_id not in merged or c["distance"] < merged[chunk_id]["distance"]:
                merged[chunk_id] = c

    return _rerank(sub_queries, list(merged.values()), state, top_k=top_k)


# ════════════════════════════════════════════════════════════════════════════
# ★ my_answer() — RDB + 벡터 검색(decomposition+리랭커) 결합
# ════════════════════════════════════════════════════════════════════════════

def my_answer(query: str) -> dict:
    state = _load_state()
    candidates: list[dict] = []

    for code in find_error_codes_in_query(query, state):
        for r in lookup_code(state["conn"], code):
            candidates.append({
                "id": f"rdb_{r['solution_id']}",
                "text": f"[{r['brand']}/{r['category']}] {r['title']}\n{r['content']}",
                "distance": 0.0,
            })

    for c in search_v2(query, state, top_k=3):
        candidates.append({
            "id": f"{c['metadata']['product_model']}_{c['metadata']['page']}",
            "text": c["text"],
            "distance": c["distance"],
        })

    context = "\n\n".join(c["text"] for c in candidates)
    answer = generate_local(SYSTEM_PROMPT, f"[문서]\n{context}\n\n[질문]\n{query}")

    return {"answer": answer, "candidates": candidates}


if __name__ == "__main__":
    run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)
