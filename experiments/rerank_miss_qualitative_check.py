"""리랭킹 "미스" 34건이 진짜 오답인지, 라벨과 다른 청크를 골랐을 뿐 실제로는
쓸만한 답을 준 건지(중복 콘텐츠로 인한 지표 착시) 눈으로 직접 확인한다.

지금까지의 히트율(65.7% 등)은 전부 "정답 청크 ID와 정확히 일치하는가"만 본다.
매뉴얼에 같은 내용이 여러 챕터에 겹쳐 있으면, 리랭커가 의미상 맞는 다른 청크를
골랐는데도 ID가 다르다는 이유만으로 오답 처리될 수 있다 - 이 가능성은 세션 초반부터
"34/99 미스가 실제로 나쁜 답변인지 검증 안 함"으로 계속 미뤄둔 과제였다.

pool_size_rerank_sweep_v2.json(pool=35, 정렬 기준=현재 프로덕션 설정)의 미스 34건 중,
1차 검색(top_k=40)에는 정답이 있었는데 리랭킹에서 떨어진 "진짜 리랭킹 미스"만 골라
(1차 검색에도 없는 6건은 리랭킹 문제가 아니라 이미 알려진 별개 이슈이므로 제외),
각 케이스에서 리랭커가 실제로 상위 5개로 무엇을 골랐는지와 라벨 정답 청크 본문을
나란히 저장한다. 수치 채점은 하지 않는다 - 사람이 읽고 판단하기 위한 자료.

사용법: python experiments/rerank_miss_qualitative_check.py
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

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v2_full.json"
SWEEP_PATH = ROOT / "experiments" / "results" / "pool_size_rerank_sweep_v2.json"
OUT_PATH = ROOT / "experiments" / "results" / "rerank_miss_qualitative_check.json"

POOL_SIZE = 35
TOP_K = 5


def rerank_with_pool(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
    """pool_size_rerank_sweep_v2.rerank_with_pool()과 동일 - 현재 프로덕션 설정 재현."""
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
    print(f"pool=35 미스: {len(misses)}건")

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
            continue  # 1차 검색(top_40) 자체에 없는 건 리랭킹 문제가 아님 - 이미 알려진 별개 이슈
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

    print(f"\n리랭킹 문제로 볼 수 있는 케이스: {len(cases)}건 (1차 검색 자체 누락 {skipped_recall}건 제외)")
    OUT_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
