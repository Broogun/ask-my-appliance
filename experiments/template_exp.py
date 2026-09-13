"""RAG 실험 템플릿.

사용법:
  1. 이 파일을 복사한다.
  2. 파일명을 {이름}_exp{번호}.py 로 바꾼다.  예: minsoo_exp01.py
  3. EXPERIMENT_NAME, NOTES 를 채운다.
  4. my_chunk() 와 my_embed() 두 함수만 구현한다.
  5. python experiments/{이름}_exp01.py 로 실행한다.
  6. results/{이름}_exp01.json 이 저장된다.

실험 변수:
  - my_chunk()  : 청킹 방식 (고정크기 / 문단 / 슬라이딩 등 자유롭게)
  - my_embed()  : 임베딩 모델 (OpenAI / bge-m3 등 자유롭게)

고정값 (모든 팀원 동일):
  - 질문 셋     : COMMON_QUESTIONS (5개)
  - LLM         : gpt-4o-mini
  - 답변 형식   : 한국어, 근거 문서 기반
  - 평가 방식   : 키워드 히트율, 평균 거리, 응답 시간
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# ── 설정 ─────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
OAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# PDF 경로 — 팀 공유 드라이브에서 받은 LG PDF 폴더
PDF_DIR = Path(__file__).parent.parent / "data" / "lg"

EXPERIMENT_NAME = "template_baseline"   # ← 본인 이름 + 번호로 변경
NOTES = "베이스라인"                     # ← 무엇을 바꿨는지 한 줄 메모

# ── 공통 테스트 질문 (수정 금지) ──────────────────────────────────────────────
COMMON_QUESTIONS = [
    ("에어컨 UE 오류가 뭐야?",              "UE",   "에러코드 단순 조회"),
    ("에어컨 필터 청소 방법 알려줘",         "필터", "일반 사용법"),
    ("세탁기 탈수가 너무 시끄러워",          "탈수", "증상 기반"),
    ("냉장고 온도를 어떻게 설정해?",         "온도", "설정/조작"),
    ("UE 오류랑 필터 청소 방법 같이 알려줘", "필터", "복합 질문"),
]


# ════════════════════════════════════════════════════════════════════════════
# ★ 여기만 구현하면 됩니다 ★
# ════════════════════════════════════════════════════════════════════════════

def my_chunk(pdf_path: str) -> list[str]:
    """PDF 파일 하나를 청크 리스트로 변환한다.

    rag_tutorial.ipynb 의 Part 1 을 참고해서 본인 방식으로 구현하세요.
    반환값은 문자열 리스트여야 합니다.
    """
    # 예시: 고정 크기 청킹 (바꿔보세요)
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)

    chunk_size = 500    # ← 바꿔보세요
    overlap    = 50     # ← 바꿔보세요

    chunks = []
    start = 0
    while start < len(full_text):
        chunks.append(full_text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def my_embed(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 임베딩 벡터 리스트로 변환한다.

    rag_tutorial.ipynb 의 Part 2 를 참고해서 본인 방식으로 구현하세요.
    반환값은 float 리스트의 리스트여야 합니다.
    """
    # 예시: OpenAI (바꿔보세요 — bge-m3 로 교체 가능)
    resp = OAI.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


# ════════════════════════════════════════════════════════════════════════════
# 아래는 공통 코드 — 수정 불필요
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 검색된 문서 내용만 근거로 자연스럽고 친절하게 답변해. "
    "문서에 없는 내용을 지어내면 안 돼. "
    "안전에 관련될 수 있는 내용(감전, 화재, 가스 누출 등)이면 "
    "전원을 끄고 서비스센터에 문의하라고 안내해."
)


def _build_index(pdf_dir: Path) -> chromadb.Collection:
    """PDF 디렉토리 전체를 청킹·임베딩해서 ChromaDB 인메모리 컬렉션으로 반환."""
    db = chromadb.Client()
    col = db.create_collection("exp")

    pdf_files = list(pdf_dir.rglob("*.pdf"))
    print(f"[인덱스] PDF {len(pdf_files)}개 처리 중...")

    all_chunks, all_ids, all_metas = [], [], []
    for pdf_path in pdf_files:
        chunks = my_chunk(str(pdf_path))
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{pdf_path.stem}_{i}")
            all_metas.append({"source": pdf_path.stem})

    # 배치 임베딩
    batch = 50
    for i in range(0, len(all_chunks), batch):
        vecs = my_embed(all_chunks[i:i + batch])
        col.add(
            ids=all_ids[i:i + batch],
            embeddings=vecs,
            documents=all_chunks[i:i + batch],
            metadatas=all_metas[i:i + batch],
        )

    print(f"[인덱스] 총 {col.count()}개 청크 저장 완료")
    return col


def _search(col: chromadb.Collection, query: str, top_k: int = 3) -> list[dict]:
    [q_vec] = my_embed([query])
    res = col.query(query_embeddings=[q_vec], n_results=top_k)
    return [
        {"text": doc, "distance": dist, "id": id_}
        for doc, dist, id_ in zip(
            res["documents"][0], res["distances"][0], res["ids"][0]
        )
    ]


def _generate(query: str, candidates: list[dict]) -> str:
    context = "\n".join(f"- {c['text']}" for c in candidates)
    resp = OAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[검색된 문서]\n{context}\n\n[사용자 질문]\n{query}"},
        ],
    )
    return resp.choices[0].message.content


def _run(col: chromadb.Collection) -> dict:
    results = []
    for question, keyword, description in COMMON_QUESTIONS:
        start = time.time()
        candidates = _search(col, question)
        answer = _generate(question, candidates)
        elapsed = time.time() - start

        all_text = answer + " ".join(c["text"] for c in candidates)
        keyword_hit = keyword in all_text
        distances = [c["distance"] for c in candidates]
        avg_dist = sum(distances) / len(distances) if distances else 1.0

        mark = "O" if keyword_hit else "X"
        print(f"[{mark}] {description}: {question}")
        print(f"    elapsed={elapsed:.1f}s | avg_dist={avg_dist:.4f}")

        results.append({
            "question": question,
            "description": description,
            "keyword_hit": keyword_hit,
            "elapsed_sec": round(elapsed, 2),
            "avg_distance": round(avg_dist, 4),
            "answer": answer,
            "candidates": [{"id": c["id"], "distance": c["distance"], "text": c["text"][:200]} for c in candidates],
        })
    return results


if __name__ == "__main__":
    col = _build_index(PDF_DIR)

    print(f"\n{'='*55}")
    print(f"  실험: {EXPERIMENT_NAME}")
    print(f"  메모: {NOTES}")
    print(f"{'='*55}\n")

    results = _run(col)

    hit_rate = sum(r["keyword_hit"] for r in results) / len(results)
    avg_dist = sum(r["avg_distance"] for r in results) / len(results)
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)

    print(f"\n{'='*55}")
    print(f"  키워드 히트율  : {hit_rate:.0%}")
    print(f"  평균 거리      : {avg_dist:.4f}")
    print(f"  평균 응답 시간 : {avg_time:.1f}s")
    print(f"{'='*55}")

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "notes": NOTES,
        "summary": {
            "keyword_hit_rate": hit_rate,
            "avg_distance": avg_dist,
            "avg_elapsed_sec": avg_time,
        },
        "results": results,
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")
