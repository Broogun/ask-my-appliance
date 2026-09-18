"""v1 판정(_llm_judged_20260916.json)에서 FALSE_DECLINE/FALSE_CONFIDENCE였던
케이스만 골라 새 판정 로직(round1_llm_judge.py의 grounding 강화판)으로
다시 채점한다. 전체 300문항 재실행은 오래 걸려서(동시성/네트워크 이슈로
반복 지연) 실제로 검증이 필요한 문제 케이스만 빠르게 재확인하는 용도.

사용법: python experiments/rejudge_v1_problems.py round2
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from round1_llm_judge import judge_one  # noqa: E402


def main() -> None:
    round_name = sys.argv[1] if len(sys.argv) > 1 else "round2"
    v1_path = Path(f"experiments/results/박형건_{round_name}_llm_judged_20260916.json")
    v1 = json.load(open(v1_path, encoding="utf-8"))

    targets = [
        r for r in v1["results"]
        if r["llm_judge_label"] in ("FALSE_DECLINE", "FALSE_CONFIDENCE")
    ]
    print(f"재판정 대상(v1 FALSE_DECLINE+FALSE_CONFIDENCE): {len(targets)}개")

    t0 = time.time()
    judged = [None] * len(targets)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(judge_one, r): i for i, r in enumerate(targets)}
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            judged[i] = fut.result()
            n_done += 1
            print(f"  {n_done}/{len(targets)} 완료 ({time.time()-t0:.0f}초)", flush=True)

    from collections import Counter
    new_labels = Counter(j["label"] for j in judged)
    print(f"\n완료: {time.time()-t0:.0f}초")
    print("새 판정 분포:", dict(new_labels))

    print("\n=== 상세 변화 ===")
    for r, j in zip(targets, judged):
        old = r["llm_judge_label"]
        new = j["label"]
        marker = "  (변화없음)" if old == new else " ***변화***"
        print(f"[{r['model']}] {r['question']}")
        print(f"  {old} -> {new}{marker}")
        print(f"  이유: {j['reason'][:150]}")
        print()

    out = []
    for r, j in zip(targets, judged):
        out.append({
            **r,
            "v1_label": r["llm_judge_label"],
            "v2_label": j["label"],
            "v2_reason": j["reason"],
            "v2_quote": j.get("quote", ""),
        })
    out_path = Path(f"experiments/results/박형건_{round_name}_rejudge_problems_20260917.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"new_label_counts": dict(new_labels), "results": out}, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
