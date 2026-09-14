"""RAG 실험 템플릿.

사용법:
  1. 이 파일을 복사한다: cp template_exp.py {이름}_exp01.py
  2. EXPERIMENT_NAME, NOTES, STRATEGY 를 채운다.
  3. my_answer() 함수 안에 본인 RAG 전 과정을 구현한다.
  4. python experiments/{이름}_exp01.py 로 실행한다.

통일 사항 (모든 팀원 동일):
  - 데이터   : data/lg/ + data/samsung/ — LG·삼성 PDF 전체
  - LLM      : gpt-4o-mini (SYSTEM_PROMPT 수정 금지)
  - 평가 질문: COMMON_QUESTIONS (수정 금지)
  - 반환 형식: {"answer": str, "candidates": list[dict]}

자유 사항 (본인이 결정):
  - 청킹 방식, 청크 크기, 오버랩
  - 임베딩 모델 (OpenAI / bge-m3 / 기타)
  - 벡터 DB (ChromaDB / FAISS / numpy 코사인 등)
  - 검색 전략 (top_k, 필터, 중복 제거 등)

COMMON_QUESTIONS/SYSTEM_PROMPT(통일 기준)와 채점/보고서 저장 로직은
pipeline/evaluate.py에 공용으로 모아뒀다 - 실험마다 똑같이 복붙하지 않고
import해서 쓴다. 이 파일에는 본인이 실제로 다르게 구현하는 my_answer()만 남는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.evaluate import SYSTEM_PROMPT, run_and_save  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")
OAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

PDF_DIR = Path(__file__).parent.parent / "data"  # lg/ + samsung/ 전체

# ── 실험 메타 정보 ────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "template_baseline"   # {이름}_exp{번호} 형식 권장

NOTES = "설명을 여기 적으세요"          # 이번 실험에서 무엇을 시도했는지

STRATEGY = {
    "chunking":         "고정 크기 500자, 오버랩 50",      # 어떤 방식?
    "chunking_reason":  "베이스라인으로 가장 단순한 방식",  # 왜 선택?
    "embedding":        "OpenAI text-embedding-3-small",
    "embedding_reason": "API로 빠르게 테스트",
    "db":               "ChromaDB 인메모리",
    "db_reason":        "별도 설치 없이 바로 사용 가능",
    "retrieval":        "top_k=3, 코사인 유사도",
    "retrieval_reason": "기본 설정",
}


# ════════════════════════════════════════════════════════════════════════════
# ★ 여기만 구현하면 됩니다 ★
# ════════════════════════════════════════════════════════════════════════════

def my_answer(query: str) -> dict:
    """사용자 질문을 받아 RAG 답변을 반환한다.

    청킹, 임베딩, DB 구성, 검색, LLM 호출까지 전부 이 함수 안에서 자유롭게 구현.
    단, 반환 형식은 반드시 지켜야 한다:

    Returns:
        {
            "answer": str,           # LLM이 생성한 최종 답변
            "candidates": [          # 검색된 근거 청크 목록
                {
                    "id":       str,
                    "text":     str,
                    "distance": float,   # 낮을수록 유사 (없으면 0.0)
                },
                ...
            ],
        }
    """
    # ── 아래 예시 코드를 지우고 본인 방식으로 구현하세요 ──────────────────

    import fitz
    import chromadb

    # 1. 청킹
    all_chunks, all_ids = [], []
    for pdf_path in PDF_DIR.rglob("*.pdf"):
        doc = fitz.open(str(pdf_path))
        text = "".join(page.get_text() for page in doc)
        size, overlap = 500, 50
        start = 0
        while start < len(text):
            chunk = text[start:start + size].strip()
            if chunk:
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}_{len(all_chunks)}")
            start += size - overlap

    # 2. 임베딩 + DB 저장
    db = chromadb.Client()
    try:
        col = db.get_collection("exp")
    except Exception:
        col = db.create_collection("exp")
        batch = 50
        for i in range(0, len(all_chunks), batch):
            resp = OAI.embeddings.create(
                model="text-embedding-3-small",
                input=all_chunks[i:i + batch],
            )
            col.add(
                ids=all_ids[i:i + batch],
                embeddings=[e.embedding for e in resp.data],
                documents=all_chunks[i:i + batch],
            )

    # 3. 검색
    [q_vec] = [e.embedding for e in OAI.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data]
    res = col.query(query_embeddings=[q_vec], n_results=3)
    candidates = [
        {"id": id_, "text": doc, "distance": dist}
        for id_, doc, dist in zip(
            res["ids"][0], res["documents"][0], res["distances"][0]
        )
    ]

    # 4. LLM 답변 (SYSTEM_PROMPT 수정 금지)
    context = "\n".join(f"- {c['text']}" for c in candidates)
    resp = OAI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[검색된 문서]\n{context}\n\n[사용자 질문]\n{query}"},
        ],
    )
    return {"answer": resp.choices[0].message.content, "candidates": candidates}


if __name__ == "__main__":
    run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)
