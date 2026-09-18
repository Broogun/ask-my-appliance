"""MODEL_QUESTIONS(round1~3, 자연어 질문)에 대해 gpt-4o-mini를 판정자로 써서
정답 chunk_id를 라벨링한다. 단순 키워드 substring보다 정확함 - 실제 청크
내용을 LLM이 읽고 "이게 진짜 답이 되는지" 판단하기 때문."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util

spec = importlib.util.spec_from_file_location("exp02", Path(__file__).resolve().parent / "박형건_exp02.py")
exp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp02)

from pipeline.evaluate import MODEL_QUESTIONS  # noqa: E402

JUDGE_PROMPT = (
    "너는 가전제품 사용설명서 검색 결과를 채점하는 심사위원이야. [질문]에 실제로 "
    "답할 수 있는 내용이 [후보] 중에 있는지 판단해. 번호가 붙은 후보들을 읽고, "
    "질문에 명확히 답이 되는 후보가 있으면 그 번호만, 없으면(전부 무관하거나 "
    "일부만 겹치고 핵심은 안 담고 있으면) 'NONE'만 출력해. 다른 설명 없이 숫자 "
    "하나 또는 NONE만 출력해."
)


def label_question(question: str, model: str, category: str, state: dict) -> dict:
    """모델 스코프 top-20 원본 청크 검색 -> gpt-4o-mini가 정답 후보 번호를 고른다."""
    where = {"product_model": model}
    cands = exp02._vector_search(question, state, top_k=20, where=where)
    if not cands:
        return {"question": question, "model": model, "source_chunk_id": None, "note": "no_candidates"}

    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('section_title')}) {c['text'][:150]}"
        for i, c in enumerate(cands)
    )
    verdict = exp02.generate_openai(
        JUDGE_PROMPT, f"[질문]\n{question}\n\n[후보]\n{listing}", temperature=0.0,
    ).strip()

    match = re.search(r"\d+", verdict)
    if "NONE" in verdict.upper() or not match:
        return {"question": question, "model": model, "source_chunk_id": None, "note": "no_answer_in_doc"}

    idx = int(match.group(0))
    if idx >= len(cands):
        return {"question": question, "model": model, "source_chunk_id": None, "note": "bad_index"}
    return {
        "question": question, "model": model,
        "source_chunk_id": cands[idx]["id"],
        "source_section_title": cands[idx]["metadata"].get("section_title"),
        "note": "labeled",
    }


if __name__ == "__main__":
    MODELS = ["FQ18GU1BHN", "AR-EH05", "M402ND", "RS63R557EB4", "RP20C3111S9",
              "WF17M9100KG", "DV90TA040TE", "FC2521KX6C", "SC3GBE50", "DJ30T9500FE"]

    state = exp02._load_state()
    targets = [q for q in MODEL_QUESTIONS if q[2] in MODELS]
    print(f"라벨링 대상 질문: {len(targets)}개")

    labels = []
    for i, (question, keyword, model, description, round_name) in enumerate(targets):
        category = description.split()[1] if len(description.split()) > 1 else ""
        r = label_question(question, model, category, state)
        r["round"] = round_name
        r["keyword"] = keyword
        labels.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(targets)} 완료")

    n_labeled = sum(1 for r in labels if r["note"] == "labeled")
    n_no_answer = sum(1 for r in labels if r["note"] == "no_answer_in_doc")
    print(f"\n라벨링 완료: {n_labeled}개 정답 확보, {n_no_answer}개는 문서에 답 없음으로 판정")

    out_path = Path(__file__).resolve().parent / "results" / "faq_ground_truth_labels.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
