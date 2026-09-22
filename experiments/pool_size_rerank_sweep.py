"""_llm_rerank()의 1차 압축 폭(pool_size, 현재 프로덕션 값 35)을, 그 값을 조정하는 데
쓰인 것과 같은 순환·편향 있는 97문항이 아니라 독립적인 5.8의 20건(사람이 PDF를 직접
읽고 확인한 정답 페이지, real_query_eval.py)으로 실제 비교한다.

pool_size_recall_check.py는 벡터 검색 단계(리랭킹 이전)까지만 봤는데, 그건 "후보 안에
정답이 있는가"만 확인할 뿐, 최종적으로 사용자에게 나가는 top-5는 리랭킹까지 거친 결과다.
이 스크립트는 pool_size 후보(15/20/35)별로 실제 _llm_rerank()를 그대로 돌려서 최종
top-5 안에 정답 페이지가 남아있는지를 비교한다 - gpt-4o-mini 리랭킹 호출이 후보 수 x
20건만큼 들어간다(사전 승인, 2026-09-22).

사용법: python experiments/pool_size_rerank_sweep.py
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

POOL_CANDIDATES = (15, 20, 35)
TOP_K = 5


def rerank_with_pool(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
    """exp02._llm_rerank()와 완전히 동일한 로직 - pool 절단 크기만 인자로 뺐다."""
    if not candidates:
        return candidates
    if len(candidates) <= top_k:
        return candidates
    pool = sorted(candidates, key=lambda c: c["distance"])[:pool_size]
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}"
        for i, c in enumerate(pool)
    )
    raw = exp02.generate_openai(
        exp02.RERANK_JUDGE_PROMPT.format(top_k=top_k),
        f"[질문]\n{query}\n\n[후보]\n{listing}",
        temperature=0.0,
    )
    import re
    indices, seen, picked = [int(x) for x in re.findall(r"\d+", raw)], set(), []
    for i in indices:
        if i < len(pool) and i not in seen:
            seen.add(i)
            picked.append(pool[i])
        if len(picked) >= top_k:
            break
    return picked if picked else pool[:top_k]


def main() -> None:
    state = exp02._load_state()
    hits = {p: 0 for p in POOL_CANDIDATES}
    n = 0
    print(f"{'query_id':>8} {'manual_id':<20} " + " ".join(f"pool={p:>3}" for p in POOL_CANDIDATES))
    for rid, manual_id, gt_page, _must_not_claim, _answerable in CASES:
        query = RAW_QUESTIONS[rid]
        product_model = manual_id.split("_", 1)[1] if "_" in manual_id else None
        where = {"product_model": product_model} if product_model else None
        raw_cands = exp02._vector_search(query, state, top_k=40, where=where)
        n += 1
        row = []
        for pool_size in POOL_CANDIDATES:
            picked = rerank_with_pool(query, raw_cands, pool_size, TOP_K)
            translated = [(_translate_chunk(c, manual_id), 0.0) for c in picked]
            hit = page_hit(translated, gt_page)
            if hit:
                hits[pool_size] += 1
            row.append("O" if hit else "X")
        print(f"{rid:>8} {manual_id:<20} " + " ".join(f"{r:>7}" for r in row))

    print(f"\n총 {n}건 (5.8의 독립 검증셋, 리랭킹까지 포함한 최종 top-{TOP_K})")
    for pool_size in POOL_CANDIDATES:
        print(f"pool_size={pool_size:>3}: {hits[pool_size]}/{n} = {hits[pool_size]/n:.1%}")


if __name__ == "__main__":
    main()
