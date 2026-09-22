"""OpenAI 호출 없이 계산 가능한 검색 단계 지표(MRR, Recall@K, Precision@1).

배경: NOTES_ARCHITECTURE.md 14절 - 검색 단계 지표(특히 MRR)가 완전히 빠져있던
공백을 메운다. `decompose_query`(질문 재구성)와 `_llm_rerank`(최종 리랭킹)만
OpenAI를 쓰고, `_vector_search`/`_faq_search`(로컬 SentenceTransformer +
Chroma)는 API 호출이 전혀 없다 - 그래서 이 스크립트는 decompose/rerank를
건너뛰고 순수 임베딩 검색 단계만 측정한다. 오늘 RPD(일일 요청 한도)를 소진해서
OpenAI 콜이 막힌 상태에서도 실행 가능.

측정 대상: experiments/results/faq_ground_truth_labels.json 의 note=="labeled"
97개 (10개 모델, LLM판정으로 확보한 정답 source_chunk_id).

측정 방식: 원본 질문 그대로(단일 주제라 decompose 없이도 큰 차이 없음, #14/#15
참고) + _strip_known_models로 모델 코드 제거 -> _vector_search(top_k=100) +
_faq_search(top_k=100) 병합(청크 id 기준 최소 거리) -> 정렬 -> 정답 청크의
순위를 찾아 MRR/Recall@K 계산. 이 병합 결과가 바로 _llm_rerank가 받는 pool의
출처이므로, Recall@35는 "지금 pool=35가 충분한가"를 직접 검증한다.

정답 판정 2단계(#16의 교훈 반영):
  - exact: chunk_id가 정확히 같음
  - section: chunk_id는 다르지만 같은 product_model+section_title (오버랩된
    조각이라 실질적으로 같은 내용)

사용법: python experiments/retrieval_metrics_offline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp02_retrieval as exp02  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels.json"
OUT_PATH = ROOT / "experiments" / "results" / "retrieval_metrics_offline.json"

POOL_SIZE = 40  # 실제 search_v2()의 candidates_per_subquery=40과 동일하게 맞춤
# (주의: 처음 100으로 돌렸을 때 태깅 전/후 완전히 동일한 결과가 나왔는데,
# 이 10개 모델의 청크 수가 26~128개라 top_k=100이면 사실상 모델 전체를
# 이미 다 가져와서 content_type 추가 검색이 새로 더할 게 없었던 것 - 실제
# 프로덕션 pool 크기(40)로 맞춰야 태깅 효과가 제대로 드러남)
RECALL_KS = (1, 3, 5, 10, 20, 35, 40)


def merged_ranking(query: str, model: str, state: dict, use_content_type: bool = False) -> list[dict]:
    """search_v2()와 동일한 병합 로직 - use_content_type=True면 태깅(#22) 이후
    추가된 content_type 필터 검색 경로까지 재현한다(기존 경로엔 손대지 않고
    추가만 하는 방식이라 그대로 옮겨옴).

    FAQ 컬렉션(#14) 병합은 2026-09-18 제거됨 - 이 스크립트로 실측한 효과가 0이라
    exp02.py에서 _faq_search 자체를 지웠다(NOTES 20.1). 이 파일 히스토리(git log)에
    당시의 use_faq on/off 비교 코드가 남아있음."""
    where = {"product_model": model}
    clean_q = exp02._strip_known_models(query, state)
    found = exp02._vector_search(clean_q, state, top_k=POOL_SIZE, where=where)
    if use_content_type:
        content_type = exp02.detect_content_type(query)
        if content_type:
            typed_where = exp02._merge_where(where, {"content_type": content_type})
            found += exp02._vector_search(clean_q, state, top_k=POOL_SIZE, where=typed_where)
    merged: dict[str, dict] = {}
    for c in found:
        cid = c["id"]
        if cid not in merged or c["distance"] < merged[cid]["distance"]:
            merged[cid] = c
    return sorted(merged.values(), key=lambda c: c["distance"])


def find_rank(ranked: list[dict], gt_chunk_id: str, gt_section: str, model: str) -> tuple[int | None, str]:
    for i, c in enumerate(ranked, start=1):
        if c["id"] == gt_chunk_id:
            return i, "exact"
    for i, c in enumerate(ranked, start=1):
        meta = c.get("metadata", {})
        if meta.get("product_model") == model and meta.get("section_title") == gt_section:
            return i, "section"
    return None, "miss"


def compute_metrics(labeled: list[dict], state: dict, use_content_type: bool) -> dict:
    rows = []
    for item in labeled:
        ranked = merged_ranking(item["question"], item["model"], state, use_content_type)
        rank, match_type = find_rank(
            ranked, item["source_chunk_id"], item["source_section_title"], item["model"]
        )
        rows.append({
            "question": item["question"],
            "model": item["model"],
            "rank": rank,
            "match_type": match_type,
            "pool_size": len(ranked),
        })

    n = len(rows)
    mrr = sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n
    recall_at_k = {
        k: sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n for k in RECALL_KS
    }
    precision_at_1 = sum(1 for r in rows if r["rank"] == 1) / n
    misses = [r for r in rows if r["rank"] is None]
    return {
        "mrr": mrr, "recall_at_k": recall_at_k, "precision_at_1": precision_at_1,
        "n": n, "rows": rows, "misses": misses,
    }


def main() -> None:
    import sys as _sys
    round_filter = _sys.argv[1] if len(_sys.argv) > 1 else None

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labeled = [x for x in labels if x["note"] == "labeled"]
    if round_filter:
        labeled = [x for x in labeled if x["round"] == round_filter]
    suffix = f", round=={round_filter}" if round_filter else ""
    print(f"평가 대상: {len(labeled)}개 (note=='labeled'{suffix})")

    state = exp02._load_state()

    before = compute_metrics(labeled, state, use_content_type=False)
    after = compute_metrics(labeled, state, use_content_type=True)

    def report(name: str, m: dict) -> None:
        print(f"\n=== {name} ===")
        print(f"MRR: {m['mrr']:.4f}")
        for k in RECALL_KS:
            print(f"  Recall@{k:>3}: {m['recall_at_k'][k]*100:.1f}%")
        print(f"Precision@1: {m['precision_at_1']*100:.1f}%")
        print(f"완전 미스: {len(m['misses'])}개")
        for r in m["misses"]:
            print(f"  [{r['model']}] {r['question']}")

    report("태깅 전 (content_type 필터 없음, #22 이전과 동일)", before)
    report("태깅 후 (content_type 필터 추가, #22 반영)", after)

    print("\n=== 변화가 생긴 문항 (태깅) ===")
    changed = 0
    for b, a in zip(before["rows"], after["rows"]):
        if b["rank"] != a["rank"]:
            changed += 1
            print(f"  [{b['model']}] {b['question']} :: {b['rank']} -> {a['rank']}")
    if changed == 0:
        print("  (없음)")

    OUT_PATH.write_text(
        json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
