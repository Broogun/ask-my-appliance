"""리랭킹 위치 편향("lost in the middle") 완화 시도 2: 순서를 안 건드리고, 대신
후보 하나하나를 개별적으로 짧게 판단하게 강제한 뒤 최종 순위를 내게 한다(CoT).

순서 셔플(rerank_shuffle_test_v2.py)은 실패했다(65.7%->57.6%) - 정렬 자체가
잡음이 아니라 실제 신호와 겹쳐 있어서였다. 이번엔 정렬은 그대로 두고, "후보를
훑어보고 바로 번호만 내라"는 기존 프롬프트 대신 "후보마다 한 줄씩 판단부터 적고
마지막에 순위를 내라"고 시켜서, 위치보다 내용을 더 보게 강제하는 방향을 시도한다.

같은 99건, pool=35 고정(사전 승인, 2026-09-23).

사용법: python experiments/rerank_cot_test_v2.py [--resume]
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
METRICS_PATH = ROOT / "experiments" / "results" / "retrieval_metrics_v2.json"
OUT_PATH = ROOT / "experiments" / "results" / "rerank_cot_test_v2.json"

POOL_SIZE = 35
TOP_K = 5

COT_JUDGE_PROMPT = (
    "너는 검색 결과를 심사하는 심사위원이야. 아래 [후보] 목록을 번호 순서대로 하나씩 보면서, "
    "[질문]에 실제로 답이 되는지 아주 짧게(5단어 이내) 판단을 한 줄씩 적어 "
    "(형식: '번호: 판단', 예: '0: 무관, 필터 청소 얘기임' / '3: 관련, 정확한 절차 설명'). "
    "후보 전부에 대해 이렇게 한 줄씩 적은 뒤, 맨 마지막 줄에 'RANK:' 다음에 관련도 높은 순으로 "
    "번호만 콤마로 구분해서 최대 {top_k}개까지 적어(예: RANK: 2,0,5). 관련 없는 후보는 RANK에서 "
    "빼도 되고, 관련 있는 게 하나도 없으면 마지막 줄에 'RANK: NONE'이라고만 적어."
)


def rerank_cot(query: str, candidates: list[dict], pool_size: int, top_k: int) -> list[dict]:
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
            COT_JUDGE_PROMPT.format(top_k=top_k),
            f"[질문]\n{query}\n\n[후보]\n{listing}",
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        return []
    # "RANK:" 이후 줄만 파싱 - 그 앞의 후보별 판단 텍스트에 섞인 숫자(청소 5단계 등)를
    # 순위로 오인하지 않기 위해 반드시 RANK 라인 이후만 본다.
    m = re.search(r"RANK\s*:\s*(.*)", raw, re.IGNORECASE)
    if not m:
        return pool[:top_k]  # 형식 안 지켜지면 거리순 폴백
    tail = m.group(1)
    if "NONE" in tail.upper():
        return []
    indices, seen, picked = [int(x) for x in re.findall(r"\d+", tail)], set(), []
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
    picked = rerank_cot(question, raw_cands, POOL_SIZE, TOP_K)
    hit = any(c["id"] == target_id for c in picked)
    return {"hit": hit}


def main(resume: bool) -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    targets = [d for d in labels if d["note"] == "labeled"]
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    ranks = metrics["ranks"]
    assert len(ranks) == len(targets), "라벨-랭크 순서 불일치"
    print(f"측정 대상: {len(targets)}건, pool_size={POOL_SIZE}(CoT 프롬프트)", flush=True)

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
    print(f"\n전체 히트율(CoT): {hits}/{n} = {hits/n:.1%}")

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
