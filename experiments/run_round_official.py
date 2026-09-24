"""round1/2/3 공식 재검증 공용 스크립트 (run_round1_official.py 일반화판).
동시성을 5->3으로 낮추고(리랭킹 단계 추가로 API 호출이 늘어서 RateLimitError가
났었음, round1에서 4건 발생 후 개별 재시도로 복구한 경험 반영), 에러 발생 시
자동으로 순차 재시도까지 포함한다."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util

spec = importlib.util.spec_from_file_location("exp02", Path(__file__).resolve().parent / "exp02_retrieval.py")
exp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp02)

from experiments.harness.evaluate import MODEL_QUESTIONS  # noqa: E402

MAX_WORKERS = 3


def run_one(item):
    question, keyword, model, description, round_name = item
    t0 = time.time()
    try:
        out = exp02.my_answer(question)
        error = None
    except Exception as e:
        out = {"answer": "", "candidates": []}
        error = str(e)
    elapsed = time.time() - t0
    answer = out.get("answer", "")
    candidates = out.get("candidates", [])
    all_text = answer + " ".join(c.get("text", "") for c in candidates)
    keyword_hit = keyword in all_text
    distances = [c.get("distance", 0.0) for c in candidates]
    avg_dist = sum(distances) / len(distances) if distances else 1.0
    return {
        "question": question, "keyword": keyword, "model": model,
        "description": description, "round": round_name,
        "keyword_hit": keyword_hit, "elapsed_sec": round(elapsed, 2),
        "avg_distance": round(avg_dist, 4), "error": error,
        "answer": answer, "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round_name", choices=["round1", "round2", "round3"])
    args = parser.parse_args()

    questions = [q for q in MODEL_QUESTIONS if q[4] == args.round_name]
    print(f"{args.round_name} 질문 수: {len(questions)}")
    exp02._load_state()

    t0 = time.time()
    results = [None] * len(questions)
    n_done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, item): i for i, item in enumerate(questions)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done}/{len(questions)} 완료 ({time.time()-t0:.0f}초)", flush=True)

    # 에러난 것만 순차 재시도 (병렬 rate limit 회피)
    n_retry = sum(1 for r in results if r["error"])
    if n_retry:
        print(f"\n에러 {n_retry}건 순차 재시도 중...")
        for i, r in enumerate(results):
            if r["error"]:
                time.sleep(3)
                item = (r["question"], r["keyword"], r["model"], r["description"], r["round"])
                results[i] = run_one(item)
                print(f"  재시도: {r['question']} -> {'성공' if not results[i]['error'] else '실패: ' + results[i]['error']}")

    elapsed_total = time.time() - t0
    n_errors = sum(1 for r in results if r["error"])
    hit_rate = sum(r["keyword_hit"] for r in results) / len(results)
    zero_hit_models = sorted({
        r["model"] for r in results
        if not any(rr["keyword_hit"] for rr in results if rr["model"] == r["model"])
    })

    print(f"\n완료: {elapsed_total:.0f}초, 히트율 {hit_rate:.1%}, 에러 {n_errors}건")
    print(f"0히트 모델: {len(zero_hit_models)}개 {zero_hit_models}")

    # 2026-09-18 재생성판(FAQ 제거 반영, #21/리랭커 전문화는 9/17에 이미 반영) -
    # 이전 날짜 파일들은 각각 그 시점 코드 상태의 기록으로 보존
    # (덮어쓰면 코드 변경 전/후 비교가 불가능해짐).
    out_path = Path(f"experiments/results/exp02_{args.round_name}_official_20260924.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "hit_rate": hit_rate, "total_elapsed": elapsed_total,
                "n_errors": n_errors, "zero_hit_models": zero_hit_models,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
