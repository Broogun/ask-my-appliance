"""LG 에어컨 에러코드 RAG 품질 테스트: data/test_questions.json을 순회하며 답변을 받아
data/eval_results.json 에 저장한다. (모델을 한 번만 로드해서 재사용하기 위해 단일 프로세스로 실행)
"""
import json
import time

import config  # noqa: F401
from pipeline.query_engine import answer

questions = json.loads(open("data/test_questions.json", encoding="utf-8").read())

results = []
for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] {q}")
    t0 = time.time()
    r = answer(q, top_k=2)
    elapsed = time.time() - t0
    results.append(
        {
            "question": q,
            "answer": r["answer"],
            "top_distance": r["candidates"][0]["distance"] if r["candidates"] else None,
            "matched_titles": [c["metadata"].get("section_title") for c in r["candidates"]],
            "elapsed_sec": round(elapsed, 1),
        }
    )
    open("data/eval_results.json", "w", encoding="utf-8").write(
        json.dumps(results, ensure_ascii=False, indent=2)
    )

print(f"\n완료: {len(results)}개 결과 -> data/eval_results.json")
