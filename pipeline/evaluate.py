"""RAG 실험 공통 평가 하니스.

팀원 실험 파일(experiments/{이름}_expNN.py)에서 이 모듈을 import해서 쓴다.
COMMON_QUESTIONS/SYSTEM_PROMPT(통일 기준)와 채점/보고서 저장 로직(RULES.md 기준
"수정 불필요" 부분)을 여기 한 곳에 모아뒀다 - 예전에는 실험 파일마다 이 코드를
그대로 복붙해서 4명이 똑같은 코드를 반복해서 들고 있었는데, 정작 이 부분은
"비교 대상"이 아니라 다 같아야 하는 채점 기준이라 복붙할 이유가 없었다. 이렇게
공용 모듈로 빼면 실험 파일에는 본인이 실제로 다르게 구현하는 my_answer()만
남는다.

사용법 (experiments/{이름}_expNN.py 안에서):
    from pipeline.evaluate import COMMON_QUESTIONS, SYSTEM_PROMPT, run_and_save

    def my_answer(query: str) -> dict:
        ...

    if __name__ == "__main__":
        run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

# ── 공통 테스트 질문 (수정 금지 - RULES.md 참고) ─────────────────────────────
COMMON_QUESTIONS = [
    ("에어컨 UE 오류가 뭐야?", "UE", "에러코드 단순 조회 (LG)"),
    ("에어컨 필터 청소 방법 알려줘", "필터", "일반 사용법"),
    ("세탁기 탈수가 너무 시끄러워", "탈수", "증상 기반"),
    ("냉장고 온도를 어떻게 설정해?", "온도", "설정/조작"),
    ("UE 오류랑 필터 청소 방법 같이 알려줘", "필터", "복합 질문"),
    ("삼성 에어컨 스스로 청소 기능은 어떻게 써?", "청소", "삼성 특화 기능"),
    ("삼성 냉장고에서 소음이 나는데 왜 그래?", "소음", "삼성 증상 기반"),
]

# ── 공통 시스템 프롬프트 (수정 금지) ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 검색된 문서 내용만 근거로 자연스럽고 친절하게 답변해. "
    "문서에 없는 내용을 지어내면 안 돼. "
    "안전에 관련될 수 있는 내용(감전, 화재, 가스 누출 등)이면 "
    "전원을 끄고 서비스센터에 문의하라고 안내해."
)

MyAnswerFn = Callable[[str], dict]


def run_eval(my_answer: MyAnswerFn) -> list[dict]:
    """COMMON_QUESTIONS 전체에 대해 my_answer(query)를 호출하고 채점한다.

    my_answer는 {"answer": str, "candidates": [{"id","text","distance"}, ...]}
    형식을 반환해야 한다 (RULES.md 반환 형식 참고).
    """
    results = []
    for question, keyword, description in COMMON_QUESTIONS:
        start = time.time()
        try:
            out = my_answer(question)
        except Exception as e:
            out = {"answer": f"[ERROR] {e}", "candidates": []}
        elapsed = time.time() - start

        answer = out.get("answer", "")
        candidates = out.get("candidates", [])
        all_text = answer + " ".join(c.get("text", "") for c in candidates)
        keyword_hit = keyword in all_text
        distances = [c.get("distance", 0.0) for c in candidates]
        avg_dist = sum(distances) / len(distances) if distances else 1.0

        mark = "O" if keyword_hit else "X"
        print(f"[{mark}] {description}: {question}")
        print(f"    elapsed={elapsed:.1f}s | avg_dist={avg_dist:.4f}")

        results.append({
            "question": question,
            "description": description,
            "keyword_hit": keyword_hit,
            "elapsed_sec": round(elapsed, 2),
            "avg_distance": round(avg_dist, 4),
            "answer": answer,
            "candidates": [
                {"id": c.get("id", ""), "distance": c.get("distance", 0.0), "text": c.get("text", "")[:200]}
                for c in candidates
            ],
        })
    return results


def build_report(experiment_name: str, notes: str, strategy: dict, results: list[dict]) -> dict:
    hit_rate = sum(r["keyword_hit"] for r in results) / len(results)
    avg_dist = sum(r["avg_distance"] for r in results) / len(results)
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)

    print(f"\n{'=' * 55}")
    print(f"  키워드 히트율  : {hit_rate:.0%}")
    print(f"  평균 거리      : {avg_dist:.4f}")
    print(f"  평균 응답 시간 : {avg_time:.1f}s")
    print(f"{'=' * 55}")

    return {
        "experiment_name": experiment_name,
        "notes": notes,
        "strategy": strategy,
        "summary": {
            "keyword_hit_rate": hit_rate,
            "avg_distance": avg_dist,
            "avg_elapsed_sec": avg_time,
        },
        "results": results,
    }


def save_report(report: dict, out_dir: str | Path = "experiments/results") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report['experiment_name']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")
    return out_path


def run_and_save(experiment_name: str, notes: str, strategy: dict, my_answer: MyAnswerFn) -> dict:
    """실험 파일의 `if __name__ == "__main__":` 블록에서 이 함수 하나만 호출하면
    질문 실행 -> 채점 -> 보고서 생성 -> experiments/results/{experiment_name}.json
    저장까지 전부 처리된다."""
    print(f"\n{'=' * 55}")
    print(f"  실험명  : {experiment_name}")
    print(f"  메모    : {notes}")
    print(f"{'=' * 55}\n")
    results = run_eval(my_answer)
    report = build_report(experiment_name, notes, strategy, results)
    save_report(report)
    return report
