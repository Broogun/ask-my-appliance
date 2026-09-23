"""faq_ground_truth_labels_v3_full.json(목차/포괄챕터 오라벨 교정판, 104건)으로
Recall@K/MRR/P@1을 다시 잰다. retrieval_metrics_v2.py와 완전히 동일한 로직,
라벨 소스만 v3로 바뀌었다. 벡터 검색만 쓰므로 OpenAI 호출 없음(무료).

사용법: python experiments/retrieval_metrics_v3.py
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

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v3_full.json"
OUT_PATH = ROOT / "experiments" / "results" / "retrieval_metrics_v3.json"
POOL_SIZE = 128
RECALL_KS = (1, 3, 5, 10, 20, 35, 40, 60, 80, 100, 128)


def rank_of(query: str, model: str, target_id: str, state: dict) -> int | None:
    clean_q = exp02._strip_known_models(query, state)
    found = exp02._vector_search(clean_q, state, top_k=POOL_SIZE, where={"product_model": model})
    ids = [c["id"] for c in found]
    return ids.index(target_id) if target_id in ids else None


def main() -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labeled = [d for d in labels if d["note"] == "labeled"]
    print(f"측정 대상: {len(labeled)}건 (v3, 목차/포괄챕터 오라벨 교정판)", flush=True)

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
