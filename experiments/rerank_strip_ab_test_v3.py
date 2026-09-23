"""_strip_known_models가 현재 프로덕션 리랭커(listwise LLM)에도 도움이 되는지
v3 라벨(104건)로 직접 A/B 비교한다.

pool_size_rerank_sweep_v3.py는 raw question(모델 코드 포함)을 그대로 벡터검색/
리랭킹에 넣었다(71.2%) - _strip_known_models가 원래 검증된 건 예전 pointwise
CrossEncoder 기준이라, 지금 listwise LLM 리랭커에도 같은 효과가 있는지 확인된 적이
없었다. web_myretriever.py에 연결한 뒤 round1 실서비스 SUCCESS가 92.3%->86.0%로
떨어져서(2026-09-23), 생성·판정 단계 노이즈 없이 리랭킹 자체만 떼어 비교한다.

pool=35(채택값) 고정, 스트립 적용 버전만 새로 측정해서 기존 71.2%(미적용)와 비교.

사용법: python experiments/rerank_strip_ab_test_v3.py [--resume]
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
OUT_PATH = ROOT / "experiments" / "results" / "rerank_strip_ab_test_v3.json"

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


def process_one(question: str, model: str, target_id: str, state: dict) -> dict:
    where = {"product_model": model}
    # 스트립 적용: 벡터검색/리랭킹 둘 다 clean_query로 - 실제 rag_myretriever.retrieve()와 동일 조건
    clean_q = exp02._strip_known_models(question, state)
    raw_cands = exp02._vector_search(clean_q, state, top_k=40, where=where)
    picked = rerank_with_pool(clean_q, raw_cands, POOL_SIZE, TOP_K)
    hit = any(c["id"] == target_id for c in picked)
    return {"hit": hit}


def main(resume: bool) -> None:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    targets = [d for d in labels if d["note"] == "labeled"]
    print(f"측정 대상: {len(targets)}건, pool_size={POOL_SIZE}(모델코드 스트립 적용)", flush=True)

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
            rows[i] = {"question": targets[i]["question"], "model": targets[i]["model"], **r}
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)", flush=True)
                OUT_PATH.write_text(json.dumps({"rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    hits = sum(1 for r in rows if r and r["hit"])
    print(f"\n스트립 적용 히트율: {hits}/{n} = {hits/n:.1%} (스트립 미적용 기존값: 74/104 = 71.2%)")

    OUT_PATH.write_text(json.dumps({"overall": f"{hits}/{n}", "rows": [r for r in rows if r is not None]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
