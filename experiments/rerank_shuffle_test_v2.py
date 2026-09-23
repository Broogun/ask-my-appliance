"""리랭커의 "lost in the middle"(목록 중간에 있는 후보를 잘 못 찾음) 가설을 검증한다.

pool_size_rerank_sweep_v2.py 결과를 retrieval_metrics_v2.py의 거리순위와 겹쳐보니,
리랭킹 전 거리순 1~10등은 히트율 80%대인데 11~35등은 17~29%로 급락했다 - 리랭커가
"진짜 관련성"이 아니라 "프롬프트에서 몇 번째로 나왔는지"에 끌려간다는 뜻일 수 있다.

이 스크립트는 거리순 정렬을 없애고 후보 순서를 무작위로 섞어서 보여준다 - 가설이
맞다면, 원래 11~35등이던 정답도 우연히 앞쪽에 걸리는 경우가 생겨 히트율이 올라가야
한다. 같은 99건, pool=35 고정(사전 승인, 2026-09-23).

사용법: python experiments/rerank_shuffle_test_v2.py [--resume]
"""
from __future__ import annotations

import argparse
import json
import random
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
METRICS_PATH = ROOT / "experiments" / "results" / "retrieval_metrics_v2.json"
OUT_PATH = ROOT / "experiments" / "results" / "rerank_shuffle_test_v2.json"

POOL_SIZE = 35
TOP_K = 5
RNG_SEED = 42


def rerank_shuffled(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
    """pool_size_rerank_sweep_v2.rerank_with_pool()과 동일하되, 거리순 정렬 후
    프롬프트에 넣기 직전에 순서를 섞는다(정답 포함 여부 자체는 거리순 pool 컷으로
    그대로 정해짐 - 순서 편향만 따로 떼어 보려는 것)."""
    if not candidates:
        return candidates
    if len(candidates) <= top_k:
        return candidates
    pool = sorted(candidates, key=lambda c: c["distance"])[:pool_size]
    rng = random.Random(RNG_SEED)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}"
        for i, c in enumerate(shuffled)
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
        if i < len(shuffled) and i not in seen:
            seen.add(i)
            picked.append(shuffled[i])
        if len(picked) >= top_k:
            break
    return picked if picked else shuffled[:top_k]


def process_one(question: str, model: str, target_id: str, state: dict) -> dict:
    where = {"product_model": model}
    raw_cands = exp02._vector_search(question, state, top_k=40, where=where)
    picked = rerank_shuffled(question, raw_cands, POOL_SIZE, TOP_K)
    hit = any(c["id"] == target_id for c in picked)
    return {"hit": hit}


def main(resume: bool) -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    targets = [d for d in labels if d["note"] == "labeled"]
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    ranks = metrics["ranks"]
    assert len(ranks) == len(targets), "라벨-랭크 순서 불일치"
    print(f"측정 대상: {len(targets)}건, pool_size={POOL_SIZE}(순서 셔플)", flush=True)

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
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {
            pool.submit(process_one, targets[i]["question"], targets[i]["model"], targets[i]["source_chunk_id"], state): i
            for i in todo
        }
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            rows[i] = {"question": targets[i]["question"], "model": targets[i]["model"], "rank_before": ranks[i], **r}
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)", flush=True)
                OUT_PATH.write_text(json.dumps({"rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    hits = sum(1 for r in rows if r and r["hit"])
    print(f"\n전체 히트율(셔플): {hits}/{n} = {hits/n:.1%}")

    buckets = {"1-5": [0, 0], "6-10": [0, 0], "11-20": [0, 0], "21-35": [0, 0], "36+/없음": [0, 0]}
    for r in rows:
        if r is None:
            continue
        rank = r["rank_before"]
        b = "36+/없음" if (rank is None or rank >= 35) else "1-5" if rank < 5 else "6-10" if rank < 10 else "11-20" if rank < 20 else "21-35"
        buckets[b][1] += 1
        if r["hit"]:
            buckets[b][0] += 1
    summary = {}
    for b, (h, t) in buckets.items():
        summary[b] = f"{h}/{t}"
        print(f"  {b:>10}: {h}/{t} = {h/t:.1%}" if t else f"  {b:>10}: 0/0")

    OUT_PATH.write_text(json.dumps({
        "overall": f"{hits}/{n}", "by_rank_bucket": summary,
        "rows": [r for r in rows if r is not None],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
