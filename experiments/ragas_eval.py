"""RAGAS Faithfulness / AnswerRelevancy를 우리 round1 결과에 계산하는 스크립트.

⚠ 준비만 해둔 상태 - 아직 실행 안 함(2026-09-18). 실행하면 OpenAI 비용이 든다
(질문당 Faithfulness 1회+AnswerRelevancy 1회 LLM 호출, AnswerRelevancy는
임베딩도 추가로 씀). 실행 전 반드시 비용/시간 확인받을 것(팀 규칙).

배경(NOTES_ARCHITECTURE.md 14.5/20.2): 우리 자체 LLM판정(SUCCESS/FALSE_DECLINE/
FALSE_CONFIDENCE/GENUINE_NO_ANSWER)은 이진분류라 "얼마나" 근거에 충실한지는
못 본다. RAGAS의 두 지표는 0~1 연속값이라 교차검증 용도로 쓸 수 있다:
  - Faithfulness: 답변의 주장이 검색된 근거로 실제로 뒷받침되는가
  - AnswerRelevancy: 답변이 질문의 핵심에 부합하는가
둘 다 "기준답변(reference)"이 필요 없어서(질문+답변+근거만 있으면 됨) 새 라벨링
없이 지금 있는 round1 데이터로 바로 돌릴 수 있다. ContextPrecision/ContextRecall은
reference가 필요해서 여기 포함 안 함(별도 라벨링 작업 필요, 백로그).

설치(아직 안 함): pip install ragas==0.4.3
참고: C:\\Users\\Admin\\Desktop\\lG CNS MCP\\my_llm_service\\my_llm_service\\RAG\\
      7. [LGCNS]MCP_RAG_신뢰성_평가.ipynb (RAGAS 부분)

사용법(준비되면):
    python experiments/ragas_eval.py --date 20260917 --round round1 --n 30
    (--n 생략하면 전체, 먼저 --n 30~50으로 파일럿 추천)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import exp02_retrieval as exp02  # noqa: E402


def load_round(round_name: str, date: str) -> list[dict]:
    path = ROOT / "experiments" / "results" / f"exp02_{round_name}_official_{date}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["results"]


async def score_one(faithfulness, answer_relevancy, row: dict) -> dict:
    """한 행(question/answer/candidates)에 두 지표를 계산. candidates[].text를
    contexts로 그대로 쓴다 - my_answer()가 실제로 사용한 근거와 동일."""
    question = row["question"]
    response = row["answer"]
    contexts = [c["text"] for c in row.get("candidates", [])]

    if not contexts:
        # 근거가 없으면(NO_ANSWER_MESSAGE 케이스) Faithfulness/AnswerRelevancy 계산이
        # 의미가 없다(근거 없이 "모르겠다"고 답한 게 오히려 정상 동작 - NOTES 14.3).
        return {"question": question, "faithfulness": None, "answer_relevancy": None, "skipped": "no_context"}

    # 근거(candidates)는 있지만 관련성 검증(_is_relevant_enough)에서 걸려 거절한
    # 케이스 - response는 우리가 설계한 고정 거절 문구(NO_ANSWER_MESSAGE) 그대로라
    # 질문마다 다른 정보가 전혀 없다. RAGAS AnswerRelevancy는 이런 무성의
    # (noncommittal) 응답을 역질문 생성이 무의미하다고 보고 항상 0.0으로 채점함
    # (2026-09-18 실측: round2/3 거절 답변 전수 AnswerRelevancy==0.0000) - 이건
    # 답변 품질 저하가 아니라 "정답 없음"을 올바르게 판단한 결과인데 정상 답변과
    # 섞어서 평균 내면 거절 비율이 늘 때마다 평균이 왜곡된다. 별도 집계 위해 태깅.
    is_decline = response.strip() == exp02.NO_ANSWER_MESSAGE

    try:
        f_result = await faithfulness.ascore(user_input=question, response=response, retrieved_contexts=contexts)
        ar_result = await answer_relevancy.ascore(user_input=question, response=response)
    except Exception as e:
        # 한 행이 실패해도(예: IncompleteOutputException) 전체 배치를 죽이지 않는다 -
        # round1_llm_judge.py에서 쓴 것과 같은 격리 패턴.
        return {"question": question, "faithfulness": None, "answer_relevancy": None,
                "skipped": f"{type(e).__name__}: {e}"}
    return {
        "question": question, "model": row.get("model"), "is_decline": is_decline,
        "faithfulness": f_result.value, "answer_relevancy": ar_result.value,
    }


def _checkpoint_path(round_name: str, date: str) -> Path:
    return ROOT / "experiments" / "results" / f"ragas_checkpoint_{round_name}_{date}.json"


async def main_async(round_name: str, date: str, n: int | None, concurrency: int = 5) -> None:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI()
    # max_tokens 기본값이 낮아서 답변이 길 때(우리 답변은 평균 5~10줄) NLI
    # verdict 생성이 중간에 끊겨 IncompleteOutputException이 나는 걸 확인함
    # (2026-09-18) - 넉넉하게 올려둠.
    evaluator_llm = llm_factory("gpt-4o-mini", client=client, temperature=0, max_tokens=2000)
    evaluator_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1)

    rows = load_round(round_name, date)
    if n:
        rows = rows[:n]
    print(f"평가 대상: {len(rows)}개 ({round_name}, official_{date})", flush=True)

    # 300문항을 순차로 돌리면 너무 오래 걸려서(30문항 파일럿도 5분+) 동시성을
    # 추가함 - round1_llm_judge.py와 같은 체크포인트 패턴(중단돼도 이어서 재개).
    ckpt_path = _checkpoint_path(round_name, date)
    checkpoint: dict[str, dict] = {}
    if ckpt_path.exists():
        checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
        print(f"체크포인트에서 {len(checkpoint)}개 재개", flush=True)

    def _key(i: int) -> str:
        return str(i)

    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(rows)
    todo = [i for i in range(len(rows)) if _key(i) not in checkpoint]
    for i in range(len(rows)):
        if _key(i) in checkpoint:
            results[i] = checkpoint[_key(i)]

    done_count = len(rows) - len(todo)

    async def _worker(i: int) -> None:
        nonlocal done_count
        async with sem:
            r = await score_one(faithfulness, answer_relevancy, rows[i])
        results[i] = r
        checkpoint[_key(i)] = r
        done_count += 1
        if done_count % 10 == 0:
            ckpt_path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")
            print(f"  {done_count}/{len(rows)} 완료", flush=True)

    await asyncio.gather(*(_worker(i) for i in todo))
    ckpt_path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")

    scored = [r for r in results if r and r.get("faithfulness") is not None]
    normal = [r for r in scored if not r.get("is_decline")]
    decline = [r for r in scored if r.get("is_decline")]

    def _avg(rs: list[dict], key: str) -> float:
        return sum(r[key] for r in rs) / len(rs) if rs else float("nan")

    if normal:
        print(f"\n[정상 답변 {len(normal)}개] 평균 Faithfulness: {_avg(normal, 'faithfulness'):.4f}"
              f" / 평균 AnswerRelevancy: {_avg(normal, 'answer_relevancy'):.4f}", flush=True)
    if decline:
        # 거절 답변의 AnswerRelevancy는 RAGAS 설계상 거의 항상 0.0 - 정상군과
        # 분리해서만 의미가 있어 여기 따로 표시(위 주석 참고).
        print(f"[거절 답변 {len(decline)}개] 평균 Faithfulness: {_avg(decline, 'faithfulness'):.4f}"
              f" / 평균 AnswerRelevancy: {_avg(decline, 'answer_relevancy'):.4f} (설계상 0에 수렴, 참고용)", flush=True)
    if scored:
        print(f"[전체(혼합) {len(scored)}개] 평균 Faithfulness: {_avg(scored, 'faithfulness'):.4f}"
              f" / 평균 AnswerRelevancy: {_avg(scored, 'answer_relevancy'):.4f}"
              f" — 거절 비율에 따라 왜곡되므로 라운드 간 비교엔 위 '정상 답변' 값을 쓸 것", flush=True)

    out_path = ROOT / "experiments" / "results" / f"exp02_{round_name}_ragas_{date}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out_path}", flush=True)
    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="round1", choices=["round1", "round2", "round3"])
    parser.add_argument("--date", default="20260917")
    parser.add_argument("--n", type=int, default=None, help="샘플 개수(생략하면 전체) - 처음엔 30~50 추천")
    args = parser.parse_args()

    print("=" * 60)
    print("이 스크립트는 실행 시 OpenAI 비용이 발생합니다.")
    print("팀 규칙: gpt-4o-mini 평가 테스트는 실행 전 확인받을 것.")
    print("=" * 60)
    asyncio.run(main_async(args.round, args.date, args.n))
