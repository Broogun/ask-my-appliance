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
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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

# ── 공통 평가 질문 (수정 금지) ────────────────────────────────────────────────
COMMON_QUESTIONS = [
    ("에어컨 UE 오류가 뭐야?",              "UE",   "에러코드 단순 조회 (LG)"),
    ("에어컨 필터 청소 방법 알려줘",         "필터", "일반 사용법"),
    ("세탁기 탈수가 너무 시끄러워",          "탈수", "증상 기반"),
    ("냉장고 온도를 어떻게 설정해?",         "온도", "설정/조작"),
    ("UE 오류랑 필터 청소 방법 같이 알려줘", "필터", "복합 질문"),
    ("삼성 에어컨 스스로 청소 기능은 어떻게 써?", "청소", "삼성 특화 기능"),
    ("삼성 냉장고에서 소음이 나는데 왜 그래?",    "소음", "삼성 증상 기반"),
]

# ── 공통 시스템 프롬프트 (수정 금지) ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 검색된 문서 내용만 근거로 자연스럽고 친절하게 답변해. "
    "문서에 없는 내용을 지어내면 안 돼. "
    "안전에 관련될 수 있는 내용(감전, 화재, 가스 누출 등)이면 "
    "전원을 끄고 서비스센터에 문의하라고 안내해."
)


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


# ════════════════════════════════════════════════════════════════════════════
# 아래는 평가 harness — 수정 불필요
# ════════════════════════════════════════════════════════════════════════════

def _run_eval() -> list[dict]:
    results = []
    for question, keyword, description in COMMON_QUESTIONS:
        start = time.time()
        try:
            out = my_answer(question)
        except Exception as e:
            out = {"answer": f"[ERROR] {e}", "candidates": []}
        elapsed = time.time() - start

        answer = out.get("answer", "")
        candidates = out.get("candidates", [])
        all_text = answer + " ".join(c.get("text", "") for c in candidates)
        keyword_hit = keyword in all_text
        distances = [c.get("distance", 0.0) for c in candidates]
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
            "candidates": [
                {"id": c.get("id",""), "distance": c.get("distance", 0.0), "text": c.get("text","")[:200]}
                for c in candidates
            ],
        })
    return results


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  실험명  : {EXPERIMENT_NAME}")
    print(f"  메모    : {NOTES}")
    print(f"{'='*55}\n")

    results = _run_eval()

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
        "strategy": STRATEGY,
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
