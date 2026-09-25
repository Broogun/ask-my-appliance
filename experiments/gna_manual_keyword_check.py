"""서비스 경로 baseline 판정을 LLM과 독립적으로 교차 확인한다 (OpenAI 비용 0, 공유 DB 읽기 전용).

질문의 키워드(MODEL_QUESTIONS의 keyword)가 그 모델 설명서 전체 청크(rag_chunks)에 나오는지 센다.
라벨마다 세 그룹으로 나눈다.
  A. 설명서 전체에 키워드가 없음          -> 설명서에 정보가 없을 가능성이 높다(동의어 표현이면 예외)
  B. 설명서에는 있으나 이 질문의 답변·근거 청크에는 없음 -> 우연한 언급이거나 검색 누락
  C. 답변 또는 근거 청크에 키워드가 있음   -> 이 방법으로는 판정을 검증할 수 없음(사람 검수 대상)
키워드가 있다고 답이 있다는 뜻은 아니고, 없다고 정보가 없다는 뜻도 아니라서 참고용 교차 검증이다.

사용법: python experiments/gna_manual_keyword_check.py      (DATABASE_URL 이 공유 DB(Postgres)를 가리켜야 함)
결과: experiments/results/gna_keyword_check_20260926.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from web import rdb  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
OUT_PATH = RESULTS / "gna_keyword_check_20260926.json"
ROUNDS = ("round1", "round2", "round3")


def classify(item: dict, model_chunks: list[str]) -> str:
    keyword = item["keyword"]
    if keyword in item["answer"] + " ".join(c["text"] for c in item["candidates"]):
        return "C"
    return "B" if any(keyword in text for text in model_chunks) else "A"


def main() -> None:
    if not rdb.use_postgres():
        raise SystemExit("공유 DB(Postgres) 모드에서만 동작한다 - DATABASE_URL 을 확인하세요")
    chunks_by_model: dict[str, list[str]] = defaultdict(list)
    for row in rdb.rows("SELECT product_model, text FROM rag_chunks"):
        chunks_by_model[row["product_model"]].append(row["text"])

    summary: dict[str, dict[str, dict[str, int]]] = {}
    review_b: list[dict] = []
    for round_name in ROUNDS:
        results = json.loads((RESULTS / f"web_baseline_{round_name}.json").read_text(encoding="utf-8"))["results"]
        counts: dict[str, Counter] = defaultdict(Counter)
        for item in results:
            group = classify(item, chunks_by_model.get(item["model"], []))
            counts[item["llm_judge_label"]][group] += 1
            if group == "B" and item["llm_judge_label"] == "GENUINE_NO_ANSWER":
                review_b.append({"round": round_name, "question": item["question"], "keyword": item["keyword"]})
        summary[round_name] = {label: dict(c) for label, c in counts.items()}
        print(f"[{round_name}]")
        for label, c in sorted(counts.items()):
            print(f"  {label:18s} 합계 {sum(c.values()):3d} | A(설명서에 없음) {c['A']:3d} | B(있으나 답·근거엔 없음) {c['B']:3d} | C(답·근거에 있음) {c['C']:3d}")

    OUT_PATH.write_text(json.dumps({"summary": summary, "gna_group_b": review_b}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
