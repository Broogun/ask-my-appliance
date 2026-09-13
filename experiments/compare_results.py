"""실험 결과 JSON 비교 스크립트.

experiments/results/*.json 을 모두 읽어서
마크다운 비교 보고서를 출력한다.
GitHub Actions에서 자동 실행되거나 로컬에서 직접 실행할 수 있다.

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

    # ── 요약 테이블 ──
    lines.append("## 요약")
    lines.append("")
    lines.append("| 실험명 | 키워드 히트율 | 평균 거리 | 평균 응답시간 |")
    lines.append("|---|---|---|---|")

    for r in results:
        s = r.get("summary", {})
        hit = s.get("keyword_hit_rate", 0)
        dist = s.get("avg_distance", 1)
        elapsed = s.get("avg_elapsed_sec", 0)
        name = r.get("experiment_name", r["_file"])
        lines.append(f"| {name} | {hit:.0%} | {dist:.4f} | {elapsed:.1f}s |")

    lines.append("")

    # ── 질문별 상세 비교 ──
    lines.append("## 질문별 상세 비교")
    lines.append("")

    # 첫 번째 결과에서 질문 목록 추출
    if results:
        questions = [r_item["question"] for r_item in results[0].get("results", [])]
        for q_idx, question in enumerate(questions):
            lines.append(f"### Q{q_idx + 1}. {question}")
            lines.append("")
            lines.append("| 실험명 | 히트 | 거리 | 답변 (앞 100자) |")
            lines.append("|---|---|---|---|")
            for r in results:
                q_results = r.get("results", [])
                if q_idx >= len(q_results):
                    continue
                qr = q_results[q_idx]
                hit_mark = "O" if qr.get("keyword_hit") else "X"
                dist = qr.get("avg_distance", 1)
                answer = qr.get("answer", "")[:100].replace("\n", " ")
                name = r.get("experiment_name", r["_file"])
                lines.append(f"| {name} | {hit_mark} | {dist:.4f} | {answer}... |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="출력 파일 경로 (없으면 stdout)")
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
