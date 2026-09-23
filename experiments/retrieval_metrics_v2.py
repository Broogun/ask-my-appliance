"""faq_ground_truth_labels_v2_full.json(순환성·생존편향 제거한 정답 99건)으로
Recall@K/MRR/P@1을 다시 잰다. retrieval_metrics_offline.py와 같은 방식(벡터
검색만, OpenAI 호출 없음)이지만 정답 소스가 다르다 - 예전엔 top-20 안에서
LLM이 고른 것(순환적)이었는데, 이번엔 전체 청크에서 고른 것(독립적)이다.

사용법: python experiments/retrieval_metrics_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import exp02_retrieval as exp02  # noqa: E402

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v2_full.json"
OUT_PATH = ROOT / "experiments" / "results" / "retrieval_metrics_v2.json"
# 2026-09-23: Recall@40이 93.9%로 포화 안 됐다는 지적으로 K를 모델 최대 청크 수(128)까지
# 확장 - "더 늘려도 의미 있는지"를 리랭킹 비용 들이기 전에 무료로 먼저 확인한다.
POOL_SIZE = 128
RECALL_KS = (1, 3, 5, 10, 20, 35, 40, 60, 80, 100, 128)


def rank_of(query: str, model: str, target_id: str, state: dict) -> int | None:
    """target_id가 top_k=POOL_SIZE 안에서 몇 등인지(0-base). 없으면 None."""
    clean_q = exp02._strip_known_models(query, state)
    found = exp02._vector_search(clean_q, state, top_k=POOL_SIZE, where={"product_model": model})
    ids = [c["id"] for c in found]
    return ids.index(target_id) if target_id in ids else None


def main() -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labeled = [d for d in labels if d["note"] == "labeled"]
    print(f"측정 대상: {len(labeled)}건 (v2, 순환성 없는 정답)", flush=True)

    state = exp02._load_state()
    ranks = []
    for i, d in enumerate(labeled, 1):
        r = rank_of(d["question"], d["model"], d["source_chunk_id"], state)
        ranks.append(r)
        if i % 20 == 0:
            print(f"  {i}/{len(labeled)} 완료", flush=True)

    n = len(ranks)
    mrr = sum(1 / (r + 1) for r in ranks if r is not None) / n
    p_at_1 = sum(1 for r in ranks if r == 0) / n
    recall_at_k = {k: sum(1 for r in ranks if r is not None and r < k) / n for k in RECALL_KS}

    print(f"\nMRR: {mrr:.3f}")
    print(f"Precision@1: {p_at_1:.3f}")
    for k in RECALL_KS:
        print(f"Recall@{k}: {recall_at_k[k]:.3f}")

    OUT_PATH.write_text(json.dumps({
        "n": n, "mrr": mrr, "precision_at_1": p_at_1, "recall_at_k": recall_at_k,
        "ranks": ranks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
