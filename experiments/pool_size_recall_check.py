"""POOL_SIZE(현재 40, 리랭킹 pool=35) 하이퍼파라미터를, 그 값을 고르는 데 쓰인 것과 같은
LLM 라벨링 97케이스가 아니라 real_query_eval.py의 18건(사람이 PDF를 직접 읽고 확인한
정답 페이지, 5.8)으로 교차 검증한다.

OpenAI 호출 없음 - 벡터 검색(로컬 SentenceTransformer + pgvector/Chroma) + 쪽 번호 보정
(locate_pages, PDF 텍스트 매칭)까지만 본다. 리랭킹·생성을 포함한 최종 답변 품질이 아니라
"pool 안에 정답 페이지가 있는가"(Recall@K)만 측정하는 거라 비용 없이 결론을 낼 수 있다.
page_hit() 판정 기준은 real_query_eval.py와 동일(±1쪽 공차)하게 맞춰서 5.8과 바로 비교 가능하게 했다.

사용법: python experiments/pool_size_recall_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import exp02_retrieval as exp02  # noqa: E402
from real_query_eval import CASES, RAW_QUESTIONS, page_hit  # noqa: E402
from web.rag_myretriever import _translate_chunk  # noqa: E402

KS = (5, 10, 20, 35, 40, 60, 100)


def main() -> None:
    state = exp02._load_state()
    hits_by_k = {k: 0 for k in KS}
    n = 0
    print(f"{'query_id':>8} {'manual_id':<20} {'gt_page':>7} {'첫 히트 순위':>10}")
    for rid, manual_id, gt_page, _must_not_claim, _answerable in CASES:
        query = RAW_QUESTIONS[rid]
        product_model = manual_id.split("_", 1)[1] if "_" in manual_id else None
        where = {"product_model": product_model} if product_model else None
        raw_cands = exp02._vector_search(query, state, top_k=max(KS), where=where)
        cands = [(_translate_chunk(c, manual_id), 0.0) for c in raw_cands]
        n += 1
        first_hit_rank = None
        for k in KS:
            if page_hit(cands[:k], gt_page):
                hits_by_k[k] += 1
                if first_hit_rank is None:
                    first_hit_rank = k
        print(f"{rid:>8} {manual_id:<20} {gt_page:>7} {str(first_hit_rank):>10}")

    print(f"\n총 {n}건 (사람이 PDF 직접 확인한 정답 페이지, 5.8)")
    print(f"{'K':>5} {'Recall@K':>10}")
    for k in KS:
        print(f"{k:>5} {hits_by_k[k]}/{n} = {hits_by_k[k]/n:.1%}")


if __name__ == "__main__":
    main()
