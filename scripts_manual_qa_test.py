import config

config.CHROMA_DIR = "chroma_db_manual_test"
config.COLLECTION_NAME = "manual_test"

import json

from pipeline.query_engine import answer

questions = [
    "에어컨 필터 청소는 어떻게 하나요?",
    "리모컨 건전지는 어떻게 끼우나요?",
    "에어컨 소음이 심한데 어떻게 해야 하나요?",
    "와이파이 연결이 안 되는데 어떻게 해요?",
    "AI 수면 모드가 뭔가요?",
    "냉장고 온도 설정은 어떻게 하나요?",
]

results = []
for q in questions:
    r = answer(q, top_k=2)
    results.append(
        {
            "question": q,
            "answer": r["answer"],
            "top_distance": r["candidates"][0]["distance"],
            "matched": [(c["metadata"]["section_title"], c["metadata"]["page"]) for c in r["candidates"]],
        }
    )
    print(f"Q: {q}")

open("data/manual_qa_results.json", "w", encoding="utf-8").write(json.dumps(results, ensure_ascii=False, indent=2))
print("done")
