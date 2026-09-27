"""재라벨링 v4 기준으로 pool_size·top_k 스윕을 다시 재기 위한 후보 수집 (리랭킹은 전부 gpt-4o-mini).

기존 스윕(pool_size_rerank_sweep_v3.py, topk_sweep_v3.py)과 같은 프롬프트·같은 질문(독립 라벨 104건)·같은 벡터 검색(원문
질문, top-40)을 쓰되, 정답 라벨과 비교하는 대신 리랭커가 고른 청크 id를 그대로 저장한다. 저장된 청크를 판정(Y/P/N)한 뒤
relabel_v4_sweep_analyze.py 로 pool·top_k별 적중을 센다.

- pool 15/20/40: top_k=5 요청 (pool 35 는 이미 판정한 재라벨링 v4 의 mini5 를 그대로 쓴다)
- top_k 스윕: pool 35 에서 top_k=10 요청을 한 번 하고 앞 1/3/5/10개를 자른다(기존 topk_sweep_v3.py 와 같은 방식)

비용: 문항당 gpt-4o-mini 리랭킹 4회 × 104문항 = 416회(사전 승인). 10개마다 중간 저장, --resume 지원.
사용법: python experiments/relabel_v4_sweep_collect.py [--n 3] [--resume] [--text-out PATH]
결과:   experiments/results/relabel_v4_sweep_candidates_20260926.json   (chunk_id 와 순서만, 청크 본문 없음)
        --text-out 을 주면 판정용 청크 본문을 그 경로(저장소 밖 권장)에 따로 저장한다.
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
OUT_PATH = ROOT / "experiments" / "results" / "relabel_v4_sweep_candidates_20260926.json"

MODEL = "gpt-4o-mini"
POOL_SIZES = (15, 20, 40)
TOP_K = 5
TOPK_REQUEST = 10
TOPK_POOL = 35


def ranked_ids(query: str, cands: list[dict], pool_size: int, top_k: int) -> list[str]:
    pool = sorted(cands, key=lambda c: c["distance"])[:pool_size]
    listing = "\n\n".join(f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}" for i, c in enumerate(pool))
    raw = exp02.generate_openai(
        exp02.RERANK_JUDGE_PROMPT.format(top_k=top_k),
        f"[질문]\n{query}\n\n[후보]\n{listing}",
        model=MODEL, temperature=0.0,
    )
    seen, order = set(), []
    for i in (int(x) for x in re.findall(r"\d+", raw)):
        if i < len(pool) and i not in seen:
            seen.add(i)
            order.append(pool[i]["id"])
        if len(order) >= top_k:
            break
    return order if order else [c["id"] for c in pool[:top_k]]  # 파싱 실패/NONE 이면 거리순 폴백(기존 스윕과 동일)


def vector_search_with_retry(question: str, model: str, state: dict) -> list[dict]:
    for attempt in range(5):
        try:
            return exp02._vector_search(question, state, top_k=40, where={"product_model": model})
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"  벡터 검색 재시도 {attempt + 1}/5: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def process_one(target: dict, state: dict, texts: dict) -> dict:
    cands = vector_search_with_retry(target["question"], target["model"], state)
    for c in cands:
        texts[c["id"]] = c["text"]
    row = {"question": target["question"], "model": target["model"], "v3_label": target["source_chunk_id"],
           "vec_order": [c["id"] for c in sorted(cands, key=lambda c: c["distance"])]}
    for p in POOL_SIZES:
        row[f"pool{p}_top5"] = ranked_ids(target["question"], cands, p, TOP_K)
    row["pool35_top10_order"] = ranked_ids(target["question"], cands, TOPK_POOL, TOPK_REQUEST)
    return row


def main(n: int | None, resume: bool, text_out: Path | None) -> None:
    targets = [d for d in json.loads(LABELS_PATH.read_text(encoding="utf-8")) if d["note"] == "labeled"]
    if n:
        targets = targets[:n]
    done: dict[str, dict] = {}
    if resume and OUT_PATH.exists():
        done = {r["question"] + "||" + r["model"]: r for r in json.loads(OUT_PATH.read_text(encoding="utf-8"))["rows"]}
        print(f"이어하기: 기존 {len(done)}개 유지", flush=True)
    texts: dict[str, str] = {}
    if resume and text_out and text_out.exists():
        texts = json.loads(text_out.read_text(encoding="utf-8"))

    state = exp02._load_state()
    rows: list[dict] = []
    t0 = time.time()
    for i, target in enumerate(targets, 1):
        key = target["question"] + "||" + target["model"]
        rows.append(done[key] if key in done else process_one(target, state, texts))
        if i % 10 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)} 완료 ({time.time() - t0:.0f}초)", flush=True)
            OUT_PATH.write_text(json.dumps({"model": MODEL, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
            if text_out:
                text_out.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="파일럿용 일부만")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--text-out", type=Path, default=None)
    args = parser.parse_args()
    main(args.n, args.resume, args.text_out)
