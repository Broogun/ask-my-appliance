"""최종 근거 개수(top_k, 현재 프로덕션 값 5 - 원래 exp02 설계는 3이었다가 웹 통합 때
이유 설명 없이 5로 바뀜)를 pool_size와 같은 방식으로 99건(순환성 없음)에 실측한다.

리랭커에게 한 번만 top_k=10(넉넉히)까지 순위를 매기게 하고, 그 순서 그대로에서
top-1/3/5/10을 잘라 각각 히트율을 계산한다 - pool_size처럼 값마다 따로 리랭킹
호출을 반복할 필요가 없어서 99회 호출로 끝난다(비용 최소화).

pool_size는 --pool 인자로 받는다(직전 pool_size 스윕에서 이긴 값을 넣는다).

사용법: python experiments/topk_sweep_v2.py --pool 40 [--resume]
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
OUT_PATH = ROOT / "experiments" / "results" / "topk_sweep_v2.json"

TOPK_CANDIDATES = (1, 3, 5, 10)
RERANK_REQUEST_TOPK = 10  # 리랭커에게 요청하는 순위 개수(위 후보의 최댓값)


def ranked_pick(query: str, candidates: list[dict], pool_size: int) -> list[str]:
    """exp02._llm_rerank()와 동일하되 top_k=10까지 순서 그대로 chunk_id 리스트를 반환."""
    if not candidates:
        return []
    pool = sorted(candidates, key=lambda c: c["distance"])[:pool_size]
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}"
        for i, c in enumerate(pool)
    )
    try:
        raw = exp02.generate_openai(
            exp02.RERANK_JUDGE_PROMPT.format(top_k=RERANK_REQUEST_TOPK),
            f"[질문]\n{query}\n\n[후보]\n{listing}",
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        return []
    indices, seen, order = [int(x) for x in re.findall(r"\d+", raw)], set(), []
    for i in indices:
        if i < len(pool) and i not in seen:
            seen.add(i)
            order.append(pool[i]["id"])
        if len(order) >= RERANK_REQUEST_TOPK:
            break
    return order


def process_one(question: str, model: str, target_id: str, pool_size: int, state: dict) -> dict:
    where = {"product_model": model}
    raw_cands = exp02._vector_search(question, state, top_k=40, where=where)
    order = ranked_pick(question, raw_cands, pool_size)
    rank = order.index(target_id) if target_id in order else None
    return {"rank": rank}


def main(pool_size: int, resume: bool) -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    targets = [d for d in labels if d["note"] == "labeled"]
    print(f"측정 대상: {len(targets)}건, pool_size={pool_size}", flush=True)

    done: dict[str, dict] = {}
    if resume and OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if existing.get("pool_size") == pool_size:
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
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {
            pool.submit(process_one, targets[i]["question"], targets[i]["model"], targets[i]["source_chunk_id"], pool_size, state): i
            for i in todo
        }
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            rows[i] = {"question": targets[i]["question"], "model": targets[i]["model"], **r}
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)", flush=True)
                OUT_PATH.write_text(json.dumps({"pool_size": pool_size, "rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    print(f"\n총 {n}건 (pool_size={pool_size})")
    summary = {}
    for k in TOPK_CANDIDATES:
        hits = sum(1 for r in rows if r and r.get("rank") is not None and r["rank"] < k)
        summary[str(k)] = f"{hits}/{n}"
        print(f"top_k={k:>2}: {hits}/{n} = {hits/n:.1%}")

    OUT_PATH.write_text(json.dumps({"pool_size": pool_size, "summary": summary, "rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.pool, args.resume)
