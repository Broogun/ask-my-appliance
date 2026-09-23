"""topk_sweep_v2.py를 v3 라벨(104건, 목차/포괄챕터 오라벨 교정판) 기준으로 재실행.

pool_size_rerank_sweep_v3.py에서 35/40이 동률로 최선임을 재확인했으니, 그 값으로
top_k=1/3/5/10 히트율도 v3 기준으로 다시 확인한다. 로직은 v2와 완전히 동일.

사용법: python experiments/topk_sweep_v3.py --pool 35 [--resume]
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

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v3_full.json"
OUT_PATH = ROOT / "experiments" / "results" / "topk_sweep_v3.json"

TOPK_CANDIDATES = (1, 3, 5, 10)
RERANK_REQUEST_TOPK = 10


def ranked_pick(query: str, candidates: list[dict], pool_size: int) -> list[str]:
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
