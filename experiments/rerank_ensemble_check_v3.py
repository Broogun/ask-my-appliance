"""앙상블(정렬+셔플 합집합) 효과가 v2 라벨 노이즈의 착시였는지 v3로 재검증한다.

v2 기준 앙상블(정렬 65.7% + 셔플 57.6% -> 합집합 71.7%)은 라벨을 고치기 전
숫자였다. v3로 라벨만 고쳤더니(리랭커는 그대로) 정렬 단독으로 71.2%가 나와서
옛 앙상블 수치와 거의 같다 - 앙상블이 진짜 위치 편향을 상쇄한 건지, 셔플 런이
우연히 목차/포괄챕터 오라벨 케이스에서 "다른 유효한 청크"를 집어서 착시로
합쳐진 건지 구분이 안 된다.

v2->v3에서 정답 청크가 안 바뀐 72건은 rerank_shuffle_test_v2.json의 셔플 히트
여부를 그대로 재사용한다(같은 target_id면 셔플 결과도 그대로 유효). 바뀌거나
새로 라벨링된 32건만 셔플 리랭킹을 새로 호출한다(거의 공짜 재검증).

사용법: python experiments/rerank_ensemble_check_v3.py [--resume]
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

V2_LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v2_full.json"
V3_LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v3_full.json"
OLD_SHUFFLE_PATH = ROOT / "experiments" / "results" / "rerank_shuffle_test_v2.json"
SORTED_V3_PATH = ROOT / "experiments" / "results" / "pool_size_rerank_sweep_v3.json"
OUT_PATH = ROOT / "experiments" / "results" / "rerank_ensemble_check_v3.json"

POOL_SIZE = 35
TOP_K = 5
RNG_SEED = 42


def rerank_shuffled(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
    if not candidates or len(candidates) <= top_k:
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
    v2_map = {(d["question"], d["model"]): d for d in json.loads(V2_LABELS_PATH.read_text(encoding="utf-8"))}
    v3_labeled = [d for d in json.loads(V3_LABELS_PATH.read_text(encoding="utf-8")) if d["note"] == "labeled"]
    old_shuffle = {(r["question"], r["model"]): r["hit"] for r in json.loads(OLD_SHUFFLE_PATH.read_text(encoding="utf-8"))["rows"]}
    sorted_v3 = {(r["question"], r["model"]): r["35"] for r in json.loads(SORTED_V3_PATH.read_text(encoding="utf-8"))["rows"]}

    reuse, todo_items = [], []
    for d in v3_labeled:
        key = (d["question"], d["model"])
        o = v2_map.get(key)
        unchanged = o is not None and o.get("note") == "labeled" and o["source_chunk_id"] == d["source_chunk_id"]
        if unchanged and key in old_shuffle:
            reuse.append({"question": d["question"], "model": d["model"], "shuffle_hit": old_shuffle[key], "source": "reuse"})
        else:
            todo_items.append(d)
    print(f"재사용: {len(reuse)}건, 신규 호출 필요: {len(todo_items)}건", flush=True)

    done: dict[str, dict] = {}
    if resume and OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        done = {r["question"] + "||" + r["model"]: r for r in existing.get("fresh_rows", [])}
        print(f"이어하기: 기존 {len(done)}개 유지", flush=True)

    state = exp02._load_state()
    fresh_rows: list[dict | None] = [None] * len(todo_items)
    todo = []
    for i, d in enumerate(todo_items):
        key = d["question"] + "||" + d["model"]
        if key in done:
            fresh_rows[i] = done[key]
        else:
            todo.append(i)
    print(f"이번에 처리할 문항: {len(todo)}개", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {
            pool.submit(process_one, todo_items[i]["question"], todo_items[i]["model"], todo_items[i]["source_chunk_id"], state): i
            for i in todo
        }
        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()
            fresh_rows[i] = {"question": todo_items[i]["question"], "model": todo_items[i]["model"], "shuffle_hit": r["hit"], "source": "fresh"}
            OUT_PATH.write_text(json.dumps({"fresh_rows": [r for r in fresh_rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"신규 호출 완료 ({time.time()-t0:.0f}초)", flush=True)

    all_rows = reuse + [r for r in fresh_rows if r is not None]
    n_sorted_hit = n_shuffle_hit = n_union_hit = 0
    detail = []
    for r in all_rows:
        key = (r["question"], r["model"])
        sorted_hit = sorted_v3.get(key, False)
        shuffle_hit = r["shuffle_hit"]
        union_hit = sorted_hit or shuffle_hit
        n_sorted_hit += sorted_hit
        n_shuffle_hit += shuffle_hit
        n_union_hit += union_hit
        detail.append({**r, "sorted_hit": sorted_hit, "union_hit": union_hit})

    n = len(all_rows)
    print(f"\n=== v3 라벨 기준 앙상블 재검증 (n={n}) ===")
    print(f"정렬 단독: {n_sorted_hit}/{n} = {n_sorted_hit/n:.1%}")
    print(f"셔플 단독: {n_shuffle_hit}/{n} = {n_shuffle_hit/n:.1%}")
    print(f"합집합(앙상블): {n_union_hit}/{n} = {n_union_hit/n:.1%}")
    print(f"앙상블이 정렬 단독보다 추가로 회수한 건수: {n_union_hit - n_sorted_hit}")

    OUT_PATH.write_text(json.dumps({
        "n": n, "sorted_only": f"{n_sorted_hit}/{n}", "shuffle_only": f"{n_shuffle_hit}/{n}",
        "union": f"{n_union_hit}/{n}", "fresh_rows": [r for r in fresh_rows if r is not None],
        "detail": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
