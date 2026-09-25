"""리랭킹 모델 비교 — 같은 후보 풀(pool=35)에 gpt-4o-mini(대조군)와 gpt-4o로 리랭킹해 최종 top-5 적중을 비교한다.

pool_size_rerank_sweep_v3.py 와 같은 독립 라벨 104건(faq_ground_truth_labels_v3_full.json)·같은 프롬프트
(exp02.RERANK_JUDGE_PROMPT)·같은 후보 목록 형식을 쓰고 리랭킹 모델만 바꾼다. 문항마다 벡터 검색은 한 번만 하고
두 모델이 같은 후보를 보게 한다. gpt-4o-mini 를 다시 도는 이유는 실행 간 변동(기존 스윕 74/104)과 모델 효과를
구분하기 위해서다. 팀 통일 사항은 gpt-4o-mini 이므로 이 스크립트는 프로덕션 코드에 영향을 주지 않는 비교 실험이다.

사용법: python experiments/rerank_model_compare_v3.py [--n 3] [--resume]     (OpenAI 비용 발생: 문항당 2회, 후보 35개 전문)
결과:   experiments/results/rerank_model_compare_v3_20260926.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import exp02_retrieval as exp02  # noqa: E402

LABELS_PATH = ROOT / "experiments" / "results" / "faq_ground_truth_labels_v3_full.json"
OUT_PATH = ROOT / "experiments" / "results" / "rerank_model_compare_v3_20260926.json"

MODELS = ("gpt-4o-mini", "gpt-4o")
POOL_SIZE = 35
TOP_K = 5


def rerank(query: str, pool: list[dict], model: str) -> list[dict]:
    listing = "\n\n".join(f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}" for i, c in enumerate(pool))
    raw = exp02.generate_openai(
        exp02.RERANK_JUDGE_PROMPT.format(top_k=TOP_K),
        f"[질문]\n{query}\n\n[후보]\n{listing}",
        model=model, temperature=0.0,
    )
    seen, picked = set(), []
    for i in (int(x) for x in re.findall(r"\d+", raw)):
        if i < len(pool) and i not in seen:
            seen.add(i)
            picked.append(pool[i])
        if len(picked) >= TOP_K:
            break
    return picked if picked else pool[:TOP_K]


def vector_search_with_retry(question: str, model: str, state: dict) -> list[dict]:
    """공유 DB 연결이 일시적으로 끊기는 경우(psycopg.OperationalError)를 대비한 재시도."""
    for attempt in range(5):
        try:
            return exp02._vector_search(question, state, top_k=40, where={"product_model": model})
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"  벡터 검색 재시도 {attempt + 1}/5: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def process_one(target: dict, state: dict) -> dict:
    raw_cands = vector_search_with_retry(target["question"], target["model"], state)
    pool = sorted(raw_cands, key=lambda c: c["distance"])[:POOL_SIZE]
    ids = [c["id"] for c in pool]
    row = {
        "question": target["question"], "model": target["model"],
        "target_vector_rank": ids.index(target["source_chunk_id"]) + 1 if target["source_chunk_id"] in ids else None,
        "pool_chars": sum(len(c["text"]) for c in pool),
    }
    for model in MODELS:
        try:
            picked = rerank(target["question"], pool, model)
            row[model] = any(c["id"] == target["source_chunk_id"] for c in picked)
        except Exception as e:  # noqa: BLE001 - 오류를 미적중으로 세지 않도록 따로 기록
            row[model] = None
            row[f"{model}_error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def main(n: int | None, resume: bool) -> None:
    targets = [d for d in json.loads(LABELS_PATH.read_text(encoding="utf-8")) if d["note"] == "labeled"]
    if n:
        targets = targets[:n]
    done: dict[str, dict] = {}
    if resume and OUT_PATH.exists():
        done = {r["question"] + "||" + r["model"]: r for r in json.loads(OUT_PATH.read_text(encoding="utf-8")).get("rows", [])}
        print(f"이어하기: 기존 {len(done)}개 유지", flush=True)

    state = exp02._load_state()
    rows: list[dict] = []
    t0 = time.time()
    for i, target in enumerate(targets, 1):
        key = target["question"] + "||" + target["model"]
        rows.append(done[key] if key in done else process_one(target, state))
        if i % 10 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)} 완료 ({time.time()-t0:.0f}초)", flush=True)
            OUT_PATH.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(rows)
    summary: dict[str, object] = {"n": total, "pool_size": POOL_SIZE, "top_k": TOP_K,
                                  "pool_chars_total": sum(r["pool_chars"] for r in rows)}
    for model in MODELS:
        ok = [r for r in rows if r[model] is not None]
        hits = sum(1 for r in ok if r[model])
        summary[model] = f"{hits}/{len(ok)}" + (f" (오류 {total - len(ok)}건 제외)" if len(ok) != total else "")
        print(f"{model:12s}: {hits}/{len(ok)} = {hits / len(ok):.1%}" if ok else f"{model}: 전부 오류", flush=True)
    both = [r for r in rows if r[MODELS[0]] is not None and r[MODELS[1]] is not None]
    summary["paired"] = {
        "둘 다 적중": sum(1 for r in both if r[MODELS[0]] and r[MODELS[1]]),
        "mini만 적중": sum(1 for r in both if r[MODELS[0]] and not r[MODELS[1]]),
        "4o만 적중": sum(1 for r in both if not r[MODELS[0]] and r[MODELS[1]]),
        "둘 다 실패": sum(1 for r in both if not r[MODELS[0]] and not r[MODELS[1]]),
    }
    print("짝지은 비교:", summary["paired"], flush=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="파일럿용 일부만")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.n, args.resume)
