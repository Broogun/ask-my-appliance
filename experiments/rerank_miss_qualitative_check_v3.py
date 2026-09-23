"""v3 라벨(목차/포괄챕터 오라벨 교정판, 104건) 기준 pool=35 잔여 미스 30건을
rerank_miss_qualitative_check.py와 동일한 방식으로 다시 눈으로 대조한다.

v2에서는 34건 미스 중 28건(1차 검색 포함)을 대조해서 21건이 "리랭커가 맞았는데
라벨이 틀렸던 것"이었다. v3로 그 패턴(목차/포괄챕터)을 고쳤으니, 남은 미스는
의도적으로 안 고친 "같은 챕터 내 청크 경계" 패턴이 주가 될 것으로 예상한다 -
그 예상이 맞는지 확인한다.

사용법: python experiments/rerank_miss_qualitative_check_v3.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import exp02_retrieval as exp02  # noqa: E402

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v3_full.json"
SWEEP_PATH = ROOT / "experiments" / "results" / "pool_size_rerank_sweep_v3.json"
OUT_PATH = ROOT / "experiments" / "results" / "rerank_miss_qualitative_check_v3.json"

POOL_SIZE = 35
TOP_K = 5


def rerank_with_pool(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
    if not candidates or len(candidates) <= top_k:
        return candidates
    pool = sorted(candidates, key=lambda c: c["distance"])[:pool_size]
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}"
        for i, c in enumerate(pool)
    )
    try:
        raw = exp02.generate_openai(
            exp02.RERANK_JUDGE_PROMPT.format(top_k=top_k),
            f"[질문]\n{query}\n\n[후보]\n{listing}",
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        return []
    indices, seen, picked = [int(x) for x in re.findall(r"\d+", raw)], set(), []
    for i in indices:
        if i < len(pool) and i not in seen:
            seen.add(i)
            picked.append(pool[i])
        if len(picked) >= top_k:
            break
    return picked if picked else pool[:top_k]


def main() -> None:
    labels = {(d["question"], d["model"]): d for d in json.loads(LABELS_PATH.read_text(encoding="utf-8")) if d["note"] == "labeled"}
    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    misses = [r for r in sweep["rows"] if r["35"] is False]
    print(f"pool=35 미스(v3): {len(misses)}건")

    state = exp02._load_state()
    cases = []
    skipped_recall = 0
    for r in misses:
        key = (r["question"], r["model"])
        label = labels[key]
        target_id = label["source_chunk_id"]
        raw_cands = exp02._vector_search(r["question"], state, top_k=40, where={"product_model": r["model"]})
        in_pool = any(c["id"] == target_id for c in raw_cands)
        if not in_pool:
            skipped_recall += 1
            continue
        picked = rerank_with_pool(r["question"], raw_cands, POOL_SIZE, TOP_K)
        target_fetch = state["collection"].get(ids=[target_id], include=["documents", "metadatas"])
        target_text = target_fetch["documents"][0] if target_fetch["documents"] else ""
        target_title = (target_fetch["metadatas"][0] or {}).get("section_title", "") if target_fetch["metadatas"] else ""
        cases.append({
            "question": r["question"],
            "model": r["model"],
            "target_id": target_id,
            "target_title": target_title,
            "target_text": target_text,
            "picked": [
                {"id": c["id"], "title": c["metadata"].get("section_title", ""), "text": c["text"]}
                for c in picked
            ],
        })
        print(f"  처리: {r['question'][:30]}...")

    same_chapter = sum(1 for c in cases if c["target_title"] in {p["title"] for p in c["picked"]})
    print(f"\n리랭킹 문제로 볼 수 있는 케이스: {len(cases)}건 (1차 검색 자체 누락 {skipped_recall}건 제외)")
    print(f"그중 라벨과 같은 챕터 제목이 top-5에 포함된 케이스(청크 경계 패턴 추정): {same_chapter}건")
    OUT_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
