"""RAG 실험 공통 평가 유틸리티.

팀원 실험 파일(experiments/)에서 import해서 사용한다.
모든 팀원이 동일한 질문 셋과 동일한 지표로 결과를 비교할 수 있게 한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

# ── 공통 테스트 질문 셋 ──────────────────────────────────────────────────────
# 팀원 전원이 이 질문으로 실험한다. 추가·수정은 팀장 승인 후 PR.
COMMON_QUESTIONS = [
    # (질문, 기대 키워드, 설명)
    ("에어컨 UE 오류가 뭐야?",              "UE",      "에러코드 단순 조회"),
    ("에어컨 필터 청소 방법 알려줘",         "필터",    "일반 사용법"),
    ("세탁기 탈수가 너무 시끄러워",          "탈수",    "증상 기반"),
    ("냉장고 온도를 어떻게 설정해?",         "온도",    "설정/조작"),
    ("UE 오류랑 필터 청소 방법 같이 알려줘", "필터",    "복합 질문"),
]


# ── 결과 데이터 구조 ─────────────────────────────────────────────────────────
@dataclass
class QuestionResult:
    question: str
    description: str
    answer: str
    candidates: list[dict]
    source: str          # "rdb" | "vector" | "combined"
    elapsed_sec: float
    keyword_hit: bool
    distances: list[float] = field(default_factory=list)

    @property
    def avg_distance(self) -> float:
        return sum(self.distances) / len(self.distances) if self.distances else 1.0


@dataclass
class EvalReport:
    experiment_name: str
    results: list[QuestionResult]
    notes: str = ""

    @property
    def keyword_hit_rate(self) -> float:
        return sum(r.keyword_hit for r in self.results) / len(self.results)

    @property
    def avg_distance(self) -> float:
        vals = [r.avg_distance for r in self.results if r.distances]
        return sum(vals) / len(vals) if vals else 1.0

    @property
    def avg_elapsed(self) -> float:
        return sum(r.elapsed_sec for r in self.results) / len(self.results)


# ── 실험 실행 ────────────────────────────────────────────────────────────────
def run_eval(
    answer_fn,
    experiment_name: str,
    questions: list[tuple] | None = None,
    top_k: int = 3,
    notes: str = "",
) -> EvalReport:
    """answer_fn(query, top_k) → dict 형태의 함수를 받아 공통 질문 셋을 돌린다.

    answer_fn은 pipeline.query_engine.answer 또는 팀원이 직접 구현한 함수여도 된다.
    반환값은 최소한 {"answer": str, "candidates": list} 구조여야 한다.
    """
    qs = questions or COMMON_QUESTIONS
    results = []

    for question, expected_keyword, description in qs:
        start = time.time()
        try:
            result = answer_fn(question, top_k=top_k)
        except Exception as exc:
            result = {"answer": f"[ERROR] {exc}", "candidates": [], "source": "error"}
        elapsed = time.time() - start

        candidates = result.get("candidates", [])
        answer_text = result.get("answer", "")
        source = result.get("source", "unknown")

        distances = [c.get("distance", 1.0) for c in candidates if "distance" in c]
        all_text = answer_text + " ".join(c.get("text", "") for c in candidates)
        keyword_hit = expected_keyword in all_text if expected_keyword else True

        results.append(
            QuestionResult(
                question=question,
                description=description,
                answer=answer_text,
                candidates=candidates,
                source=source,
                elapsed_sec=elapsed,
                keyword_hit=keyword_hit,
                distances=distances,
            )
        )

    return EvalReport(experiment_name=experiment_name, results=results, notes=notes)


# ── 리포트 출력 ───────────────────────────────────────────────────────────────
def print_report(report: EvalReport) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  실험명  : {report.experiment_name}")
    if report.notes:
        print(f"  메모    : {report.notes}")
    print(f"  키워드 히트율  : {report.keyword_hit_rate:.0%}")
    print(f"  평균 거리(낮을수록 좋음): {report.avg_distance:.4f}")
    print(f"  평균 응답 시간 : {report.avg_elapsed:.1f}s")
    print(sep)

    for r in report.results:
        hit_mark = "O" if r.keyword_hit else "X"
        print(f"\n[{hit_mark}] {r.description} — {r.question!r}")
        print(f"    source={r.source} | elapsed={r.elapsed_sec:.1f}s | "
              f"candidates={len(r.candidates)} | avg_dist={r.avg_distance:.4f}")
        print(f"    답변: {r.answer[:120].strip()}{'...' if len(r.answer) > 120 else ''}")

    print(f"\n{sep}\n")


def save_report(report: EvalReport, path: str) -> None:
    """결과를 JSON으로 저장한다 (experiments/results/ 권장)."""
    data = {
        "experiment_name": report.experiment_name,
        "notes": report.notes,
        "summary": {
            "keyword_hit_rate": report.keyword_hit_rate,
            "avg_distance": report.avg_distance,
            "avg_elapsed_sec": report.avg_elapsed,
        },
        "results": [
            {
                "question": r.question,
                "description": r.description,
                "source": r.source,
                "keyword_hit": r.keyword_hit,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "avg_distance": round(r.avg_distance, 4),
                "answer": r.answer,
                "candidates": [
                    {"id": c.get("id"), "distance": c.get("distance"),
                     "text": c.get("text", "")[:200]}
                    for c in r.candidates
                ],
            }
            for r in report.results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[저장] {path}")
