"""#18/#19/#21(타이틀 하드코딩 구조적 매칭)이 #22(content_type 태깅)만으로도
대체 가능한지 API 비용 없이(로컬 임베딩만) 검증한다.

배경: 사용자가 "#18/#19/#21이 과적합 로직 아니냐"고 지적함 - 정확히는
"일반화 안 되는 하드코딩"(새 모델/새 타이틀 표현엔 안 통함)에 가깝다는 게
결론이었음(NOTES_ARCHITECTURE.md 15절 참고). 그럼 실제로 #22(content_type
태깅) 만으로 #18/#19/#21이 하드코딩 타이틀 리스트로 찾아주는 챕터를 다시
찾아낼 수 있는지 직접 측정해본다.

방법: MODEL_QUESTIONS(900개) 중 트러블슈팅/패널/스펙 트리거가 걸리는 질문만
골라서, state["troubleshooting_chunks"]/["panel_chunks"]/["spec_chunks"]
(하드코딩 타이틀 리스트가 찾아준 chunk_id들 = "정답"으로 취급)와
content_type 필터를 건 순수 벡터검색 결과를 비교한다.

세 가지 조건을 비교:
  A. 일반 벡터검색만 (product_model where만, content_type 필터 없음)
  B. content_type 필터만 (product_model + content_type where)
  C. A+B 병합 (지금 search_v2()가 실제로 하는 것과 동일)

각 조건에서 "정답"(하드코딩 리스트의 chunk_id) 중 하나라도 top-K 안에
들어오면 히트로 카운트. 이 스크립트는 로컬 임베딩(SentenceTransformer)+
Chroma만 사용 - OpenAI 호출 없음.

사용법: python experiments/tagging_vs_structural_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import 박형건_exp02 as exp02  # noqa: E402
from experiments.harness.evaluate import MODEL_QUESTIONS  # noqa: E402

TOP_K = 35  # 실제 _llm_rerank pool 크기와 동일 - 이게 진짜 의미있는 기준

CONTENT_TYPE_FOR = {
    # troubleshooting 트리거의 실제 정답 챕터("LG ThinQ로 고장 진단하기" 등)는
    # 증상-원인-해결이 아니라 "진단 절차"라 태거가 TROUBLESHOOTING/PROCEDURE
    # 경계에서 혼동함(실측: 정답 청크 253개 중 TROUBLESHOOTING 202/GENERAL
    # 28/PROCEDURE 19) - 둘 다 허용하도록 확장
    "troubleshooting": ["TROUBLESHOOTING", "PROCEDURE"],
    "panel": "PROCEDURE",
    "spec": "SPEC",
}


def classify_trigger(query: str) -> str | None:
    if exp02.is_generic_troubleshooting_question(query):
        return "troubleshooting"
    if exp02.is_panel_setting_question(query):
        return "panel"
    if exp02.is_spec_question(query):
        return "spec"
    return None


def gt_chunk_ids(kind: str, model: str, state: dict) -> list[str]:
    key = {"troubleshooting": "troubleshooting_chunks", "panel": "panel_chunks", "spec": "spec_chunks"}[kind]
    return state.get(key, {}).get(model, [])


def search_ids(query: str, model: str, state: dict, content_type: str | list[str] | None, top_k: int) -> set[str]:
    where = {"product_model": model}
    if content_type:
        ct_filter = {"$in": content_type} if isinstance(content_type, list) else content_type
        where = exp02._merge_where(where, {"content_type": ct_filter})
    clean_q = exp02._strip_known_models(query, state)
    found = exp02._vector_search(clean_q, state, top_k=top_k, where=where)
    found += exp02._faq_search(clean_q, state, top_k=top_k, where=where)
    return {c["id"] for c in found}


def main() -> None:
    state = exp02._load_state()

    counts = {"troubleshooting": 0, "panel": 0, "spec": 0}
    hits = {
        kind: {"A_generic": 0, "B_content_type": 0, "C_combined": 0}
        for kind in ("troubleshooting", "panel", "spec")
    }
    skipped_no_gt = {"troubleshooting": 0, "panel": 0, "spec": 0}

    for question, keyword, model, description, round_name in MODEL_QUESTIONS:
        kind = classify_trigger(question)
        if kind is None:
            continue
        gt = set(gt_chunk_ids(kind, model, state))
        if not gt:
            skipped_no_gt[kind] += 1
            continue
        counts[kind] += 1

        content_type = CONTENT_TYPE_FOR[kind]
        a = search_ids(question, model, state, content_type=None, top_k=TOP_K)
        b = search_ids(question, model, state, content_type=content_type, top_k=TOP_K)
        c = a | b

        if gt & a:
            hits[kind]["A_generic"] += 1
        if gt & b:
            hits[kind]["B_content_type"] += 1
        if gt & c:
            hits[kind]["C_combined"] += 1

    print(f"TOP_K = {TOP_K}\n")
    for kind in ("troubleshooting", "panel", "spec"):
        n = counts[kind]
        print(f"=== {kind} (트리거 {n}개, GT 없어서 제외 {skipped_no_gt[kind]}개) ===")
        if n == 0:
            print("  (해당 없음)\n")
            continue
        for label in ("A_generic", "B_content_type", "C_combined"):
            h = hits[kind][label]
            print(f"  {label:>16}: {h}/{n} = {h/n*100:.1f}%")
        print()

    total_n = sum(counts.values())
    total = {label: sum(hits[k][label] for k in counts) for label in ("A_generic", "B_content_type", "C_combined")}
    print(f"=== 합계 (트리거 {total_n}개) ===")
    for label in ("A_generic", "B_content_type", "C_combined"):
        print(f"  {label:>16}: {total[label]}/{total_n} = {total[label]/total_n*100:.1f}%")


if __name__ == "__main__":
    main()
