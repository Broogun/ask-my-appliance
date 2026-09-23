"""faq_ground_truth_labels_v2_full.json의 99건(정확한 chunk_id 매칭, 순환성
없음)으로 pool_size 후보를 리랭킹까지 포함해 실측 비교한다.

15/20/35(첫 스윕)에 40을 추가해 처음부터 다시 깔끔하게 돌린다(2026-09-23) -
벡터검색 단독 Recall@40(93.9%)이 Recall@35(88.9%)보다 뚜렷이 높았고, 프로덕션이
이미 벡터검색에서 40개를 가져오는데 리랭킹 전에 35개로 자르고 있어서 40은
추가 검색 비용 없이 그냥 안 버리는 선택지다.

99건 x pool_size 후보(15/20/35/40) = 최대 396회 gpt-4o-mini 리랭킹 호출(사전
승인, 2026-09-23). worker=1 + 10개마다 중간 저장(--resume 지원).

사용법: python experiments/pool_size_rerank_sweep_v2.py [--resume]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import exp02_retrieval as exp02  # noqa: E402

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v2_full.json"
OUT_PATH = ROOT / "experiments" / "results" / "pool_size_rerank_sweep_v2.json"

POOL_CANDIDATES = (15, 20, 35, 40)
TOP_K = 5


def rerank_with_pool(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
    """exp02._llm_rerank()와 동일한 로직 - pool 절단 크기만 인자로 뺐다."""
    if not candidates:
        return candidates
    if len(candidates) <= top_k:
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
    except Exception:  # noqa: BLE001 - 재라벨링 때 이중 재시도로 데였던 교훈, 여기선 그냥 실패 처리
        return []
    indices, seen, picked = [int(x) for x in re.findall(r"\d+", raw)], set(), []
    for i in indices:
        if i < len(pool) and i not in seen:
            seen.add(i)
            picked.append(pool[i])
        if len(picked) >= top_k:
            break
    return picked if picked else pool[:top_k]


def process_one(question: str, model: str, target_id: str, state: dict) -> dict:
    where = {"product_model": model}
    raw_cands = exp02._vector_search(question, state, top_k=40, where=where)
    row = {}
    for pool_size in POOL_CANDIDATES:
        picked = rerank_with_pool(question, raw_cands, pool_size, TOP_K)
        row[str(pool_size)] = any(c["id"] == target_id for c in picked)
    return row


def main(resume: bool) -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    targets = [d for d in labels if d["note"] == "labeled"]
    print(f"측정 대상: {len(targets)}건 (v2, 순환성 없는 정답)", flush=True)

    done: dict[str, dict] = {}
    if resume and OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        done = {r["question"] + "||" + r["model"]: r for r in existing.get("rows", [])}
        print(f"이어하기: 기존 {len(done)}개 유지", flush=True)

    state = exp02._load_state()
    rows: list[dict | None] = [None] * len(targets)
    todo = []
    for i, d in enumerate(targets):
        key = d["question"] + "||" + d["model"]
        if key in done:
            rows[i] = done[key]
        else:
            todo.append(i)
    print(f"이번에 처리할 문항: {len(todo)}개", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:  # 2에서 낮춤: 반복적으로 정체되는 걸 실측(2026-09-23)
        futures = {
            pool.submit(process_one, targets[i]["question"], targets[i]["model"], targets[i]["source_chunk_id"], state): i
            for i in todo
        }
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            hitmap = fut.result()
            rows[i] = {"question": targets[i]["question"], "model": targets[i]["model"], **hitmap}
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)", flush=True)
                OUT_PATH.write_text(json.dumps({"rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    print(f"\n총 {n}건 (v2_full 정답 99건, 리랭킹까지 포함한 최종 top-{TOP_K})")
    summary = {}
    for pool_size in POOL_CANDIDATES:
        hits = sum(1 for r in rows if r and r.get(str(pool_size)))
        summary[str(pool_size)] = f"{hits}/{n}"
        print(f"pool_size={pool_size:>3}: {hits}/{n} = {hits/n:.1%}")

    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
