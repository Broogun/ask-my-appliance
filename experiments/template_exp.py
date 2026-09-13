"""RAG 실험 템플릿.

사용법:
  1. 이 파일을 복사한다.
  2. 파일명을 {본인이름}_exp{번호}.py 로 바꾼다.  예: minsoo_exp01.py
  3. EXPERIMENT_NAME과 NOTES를 채운다.
  4. ── 실험 구간 ── 안에 본인 아이디어를 구현한다.
  5. python experiments/{본인이름}_exp01.py 로 실행한다.
  6. results/ 폴더에 JSON 결과가 저장된다.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.query_engine import answer
from pipeline.evaluate import COMMON_QUESTIONS, run_eval, print_report, save_report

# ── 실험 메타 정보 (반드시 채울 것) ──────────────────────────────────────────
EXPERIMENT_NAME = "template_baseline"   # 예: "minsoo_reranking_v1"
NOTES = "베이스라인 그대로 실행"         # 무엇을 바꿨는지 간단히 기록


# ── 실험 구간: 여기를 자유롭게 수정한다 ──────────────────────────────────────
# 기본값: 기존 answer() 함수를 그대로 사용한다.
# 리랭킹, 청킹 전략, 프롬프트 등 원하는 부분을 교체/확장해서 실험한다.

def my_answer(query: str, top_k: int = 3) -> dict:
    """팀원이 구현하는 RAG 함수. 반환값은 반드시 아래 형식을 지킨다.

    {
        "answer": str,
        "candidates": [{"id": str, "text": str, "distance": float, "metadata": dict}, ...],
        "source": str,   # "vector" | "rdb" | "combined" | 본인이 정한 이름
    }
    """
    # ↓ 이 줄을 지우고 본인 구현으로 교체한다
    return answer(query, top_k=top_k)


# ── 실험 실행 (수정 불필요) ───────────────────────────────────────────────────
if __name__ == "__main__":
    report = run_eval(
        answer_fn=my_answer,
        experiment_name=EXPERIMENT_NAME,
        questions=COMMON_QUESTIONS,
        top_k=3,
        notes=NOTES,
    )
    print_report(report)

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_path = os.path.join(
        os.path.dirname(__file__), "results", f"{EXPERIMENT_NAME}.json"
    )
    save_report(report, out_path)
