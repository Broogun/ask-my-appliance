"""실험 결과 비교 보고서 생성기.

experiments/results/*.json 을 모두 읽어서
전략 비교 + 성능 비교 마크다운 보고서를 출력한다.

사용법:
  python experiments/compare_results.py
  python experiments/compare_results.py --out reports/latest.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_results() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("results/ 폴더에 JSON 파일이 없습니다.", file=sys.stderr)
        return []
    results = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            data["_file"] = f.name
            results.append(data)
    return results


def build_report(results: list[dict]) -> str:
    lines = ["# 실험 결과 비교 보고서", ""]
    lines.append(f"총 실험 수: **{len(results)}**")
    lines.append("")

    # ── 1. 전략 비교 ──────────────────────────────────────────────────────
    lines.append("## 1. 전략 비교")
    lines.append("")
    lines.append("| 실험명 | 청킹 | 이유 | 임베딩 | 이유 | DB/검색 | 이유 |")
    lines.append("|---|---|---|---|---|---|---|")

    for r in results:
        s = r.get("strategy", {})
        name = r.get("experiment_name", r["_file"])
        chunking       = s.get("chunking", "-")
        chunking_why   = s.get("chunking_reason", "-")
        embedding      = s.get("embedding", "-")
        embedding_why  = s.get("embedding_reason", "-")
        db             = s.get("db", "-") + " / " + s.get("retrieval", "-")
        db_why         = s.get("db_reason", "-") + " / " + s.get("retrieval_reason", "-")
        lines.append(f"| {name} | {chunking} | {chunking_why} | {embedding} | {embedding_why} | {db} | {db_why} |")

    lines.append("")

    # ── 2. 성능 요약 ──────────────────────────────────────────────────────
    lines.append("## 2. 성능 요약")
    lines.append("")
    lines.append("| 실험명 | 키워드 히트율 | 평균 거리 ↓ | 평균 응답시간 |")
    lines.append("|---|---|---|---|")

    sorted_results = sorted(results, key=lambda r: r.get("summary", {}).get("keyword_hit_rate", 0), reverse=True)
    for r in sorted_results:
        s = r.get("summary", {})
        hit    = s.get("keyword_hit_rate", 0)
        dist   = s.get("avg_distance", 1)
        elapsed = s.get("avg_elapsed_sec", 0)
        name   = r.get("experiment_name", r["_file"])
        lines.append(f"| {name} | {hit:.0%} | {dist:.4f} | {elapsed:.1f}s |")

    lines.append("")

    # ── 3. 질문별 상세 비교 ───────────────────────────────────────────────
    lines.append("## 3. 질문별 상세 비교")
    lines.append("")

    if results:
        questions = [item["question"] for item in results[0].get("results", [])]
        for q_idx, question in enumerate(questions):
            desc = results[0]["results"][q_idx].get("description", "")
            lines.append(f"### Q{q_idx+1}. [{desc}] {question}")
            lines.append("")
            lines.append("| 실험명 | 키워드 히트 | 거리 | 응답시간 | 답변 (앞 120자) |")
            lines.append("|---|---|---|---|---|")
            for r in results:
                q_results = r.get("results", [])
                if q_idx >= len(q_results):
                    continue
                qr = q_results[q_idx]
                hit_mark = "O" if qr.get("keyword_hit") else "X"
                dist     = qr.get("avg_distance", 1)
                elapsed  = qr.get("elapsed_sec", 0)
                answer   = qr.get("answer", "")[:120].replace("\n", " ")
                name     = r.get("experiment_name", r["_file"])
                lines.append(f"| {name} | {hit_mark} | {dist:.4f} | {elapsed:.1f}s | {answer}... |")
            lines.append("")

    # ── 4. 결론 ───────────────────────────────────────────────────────────
    if sorted_results:
        best = sorted_results[0]
        best_name = best.get("experiment_name", best["_file"])
        best_hit  = best.get("summary", {}).get("keyword_hit_rate", 0)
        lines.append("## 4. 결론")
        lines.append("")
        lines.append(f"키워드 히트율 기준 최우수 실험: **{best_name}** ({best_hit:.0%})")
        best_strategy = best.get("strategy", {})
        if best_strategy:
            lines.append("")
            lines.append(f"- 청킹: {best_strategy.get('chunking', '-')}")
            lines.append(f"- 임베딩: {best_strategy.get('embedding', '-')}")
            lines.append(f"- DB/검색: {best_strategy.get('db', '-')} / {best_strategy.get('retrieval', '-')}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = load_results()
    if not results:
        sys.exit(1)

    report = build_report(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"보고서 저장: {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
