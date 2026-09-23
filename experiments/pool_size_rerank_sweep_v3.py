"""faq_ground_truth_labels_v3_full.json(104건, 목차/포괄챕터 오라벨 교정판)으로
pool_size_rerank_sweep_v2.py를 그대로 재실행 - 라벨 소스만 v3로 바꿨다.

v2(99건, 순환성은 없지만 목차/포괄챕터가 정답으로 잘못 뽑히는 경우가 있었음) 기준
pool=35 히트율 65.7%(65/99) 중 34건 미스를 정성 확인해보니(rerank_miss_qualitative_check.py)
28/34건이 실제로는 리랭커가 맞았는데 라벨 오류로 미스 처리된 것이었다. 그중 목차 제외 +
포괄vs전용 챕터 판단 가드레일을 프롬프트에 추가해 재라벨링한 게 v3(104건)다.
같은 pool_size 후보로 다시 스윕해서 실제로 히트율이 올라가는지, 35/40이 여전히
동률로 최적인지 확인한다.

104건 x pool_size 후보(15/20/35/40) = 최대 416회 gpt-4o-mini 리랭킹 호출(사전
승인, 2026-09-23). worker=1 + 10개마다 중간 저장(--resume 지원).

사용법: python experiments/pool_size_rerank_sweep_v3.py [--resume]
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
OUT_PATH = ROOT / "experiments" / "results" / "pool_size_rerank_sweep_v3.json"

POOL_CANDIDATES = (15, 20, 35, 40)
TOP_K = 5


def rerank_with_pool(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
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
    print(f"측정 대상: {len(targets)}건 (v3, 목차/포괄챕터 오라벨 교정판)", flush=True)

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
            hitmap = fut.result()
            rows[i] = {"question": targets[i]["question"], "model": targets[i]["model"], **hitmap}
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)", flush=True)
                OUT_PATH.write_text(json.dumps({"rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    print(f"\n총 {n}건 (v3_full 정답 104건, 리랭킹까지 포함한 최종 top-{TOP_K})")
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
