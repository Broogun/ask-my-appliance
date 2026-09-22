"""모델별 검증(회귀 테스트) 스크립트 — `experiments/harness/evaluate.py`의 `COMMON_QUESTIONS`(22개,
팀 공용 비교 기준, 수정 금지)와는 완전히 별개의 개인용 QA 도구다.

**왜 필요한가**: `COMMON_QUESTIONS`만으로는 60개 제품 모델 중 39개만 우연히
검색됐고 21개는 이번 실험에서 단 한 번도 검증되지 않았다 (실제로 확인함). 삼성
문서 TOC 손상 버그(AC_AF19TX978MZ3N, RF_RH82M9152SL)도 "이 파일 하나만 자세히
들여다보다가" 발견한 거라, 60개 전체를 최소 한 번씩은 훑어보는 절차가 없으면
비슷한 버그를 놓칠 수 있다.

**설계 원칙**:
- 모델당 최소 몇 개의 질문으로 "이 모델 자체의 청킹/임베딩이 잘 검색되는지"를
  본다. 1라운드는 모델당 최대 5문항(총 최대 300개), 다음 라운드에서 다른 유형의
  질문을 더 추가하는 식으로 점진적으로 쌓아간다 (사용자 아이디어).
- **LLM 최종 답변 생성은 포함하지 않는다** - 순수 벡터 검색만 채점한다. 이유:
  (1) 지금 확인하려는 건 "청킹/임베딩이 맞는 콘텐츠를 찾아오는지"이지 "LLM이
  답을 잘 쓰는지"가 아니다 - 실제로 이번 세션에서 발견한 버그(TOC 손상, 카테고리
  오염)는 전부 검색 단계 문제였다. (2) 로컬 LLM(Ollama)으로 300개를 전부 돌리면
  3~4시간에 타임아웃까지 나서 실용적이지 않다 (실제로 여러 번 겪음). 순수 벡터
  검색은 GPU에서 300개도 1분 안팎이라 지금 바로 전체 실행 가능하다.
  최종 답변 품질까지 포함한 전수 테스트는 gpt-4o-mini로 전환한 뒤(API 비용
  계산 결과 300문항에 약 $0.3로 무시할 수준 - 대신 속도/안정성도 훨씬 나음)
  별도로 진행한다.

사용법:
  python experiments/model_verification.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

exp = importlib.import_module("exp02_retrieval")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"

QUESTIONS_PER_MODEL_ROUND1 = 5

# 페이지 폴백 처리된 문서는 section_title이 "N페이지"처럼 의미 없는 라벨이라
# 그대로 질문을 만들 수 없다 - 대신 청크 본문에서 직접 키워드를 뽑는다.
PAGE_FALLBACK_TITLE = re.compile(r"^\d+페이지$")


def _clean_title(title: str) -> str:
    """'[리모컨으로 사용하기] 추가 기능 및 설정하기' -> '추가 기능 및 설정하기'"""
    return re.sub(r"^\[[^\]]*\]\s*", "", title).strip()


def _extract_keyword_from_text(text: str) -> str | None:
    """페이지 폴백 청크(제목이 'N페이지')용: 본문에서 2자 이상 한글 단어 중
    가장 먼저 나오는 의미 있어 보이는 걸 골라 키워드로 쓴다."""
    body = re.sub(r"^\[[^\]]*\]\s*", "", text)
    words = re.findall(r"[가-힣]{2,}", body)
    stop = {"제품", "사용", "이때", "경우", "때문", "그림", "참고", "확인"}
    for w in words:
        if w not in stop:
            return w
    return words[0] if words else None


def build_round1_questions(state, questions_per_model: int = QUESTIONS_PER_MODEL_ROUND1) -> list[dict]:
    """모델별로 실제 section_title(또는 페이지 폴백이면 본문 키워드) 기반 질문을 만든다.

    각 질문은 어떤 청크에서 만들어졌는지(source_chunk_id)를 정확히 기록한다 -
    keyword substring 매칭은 "문이 안 열려요"류 질문에서 실제론 완전히 엉뚱한
    내용인데도 키워드가 우연히 들어가서 hit으로 잡히는 걸 실제로 확인했다
    (거짓 양성). chunk_id 정확 일치로 채점하면 이런 애매함이 없다."""
    all_docs = state["collection"].get(limit=6000)
    model_chunks: dict[str, list[dict]] = defaultdict(list)
    for cid, doc, meta in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
        model_chunks[meta["product_model"]].append(
            {"id": cid, "text": doc, "section_title": meta["section_title"]}
        )

    questions = []
    for model, chunks in sorted(model_chunks.items()):
        # 섹션 제목 기준으로 고르게 퍼진 대표 청크를 questions_per_model개 뽑는다.
        n = len(chunks)
        step = max(1, n // questions_per_model)
        picked_idxs = list(range(0, n, step))[:questions_per_model]

        for idx in picked_idxs:
            chunk = chunks[idx]
            title = chunk["section_title"]
            if PAGE_FALLBACK_TITLE.match(title):
                keyword = _extract_keyword_from_text(chunk["text"])
                if not keyword:
                    continue
                question = f"{model} {keyword} 관련 내용 알려줘"
            else:
                clean = _clean_title(title)
                keyword = clean
                question = f"{clean} 어떻게 하나요?"

            questions.append({
                "product_model": model,
                "question": question,
                "keyword": keyword,
                "source_section_title": title,
                "source_chunk_id": chunk["id"],
            })
    return questions


def run_round(questions: list[dict], state, top_k: int = 3) -> list[dict]:
    """모델 스코프로 좁힌 순수 벡터 검색만 채점한다 (LLM 답변 생성 없음 - 위 설명 참고).

    채점 기준은 keyword substring이 아니라 **정확한 source_chunk_id 일치**다 -
    "DF90H24R5C 스타일러 문이 안 열려요" 질문에서 완전히 엉뚱한 트러블슈팅
    청크가 반환됐는데도 "문"이라는 글자가 우연히 들어가 있어서 keyword_hit=True로
    잡히는 거짓 양성을 실제로 확인했다. chunk_id로 채점하면 이런 애매함이 없다."""
    results = []
    for q in questions:
        candidates = exp._vector_search(
            q["question"], state, top_k=top_k, where={"product_model": q["product_model"]}
        )
        returned_ids = [c["id"] for c in candidates]
        chunk_hit = q["source_chunk_id"] in returned_ids
        min_distance = min((c["distance"] for c in candidates), default=None)
        results.append({
            **q,
            "chunk_hit": chunk_hit,
            "hit_rank": returned_ids.index(q["source_chunk_id"]) + 1 if chunk_hit else None,
            "min_distance": round(min_distance, 4) if min_distance is not None else None,
            "top_candidate_text": candidates[0]["text"][:150] if candidates else "",
        })
    return results


def save_round(round_name: str, results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"model_verification_{round_name}.json"
    hit_rate = sum(r["chunk_hit"] for r in results) / len(results) if results else 0.0
    models_tested = sorted(set(r["product_model"] for r in results))
    models_failed = sorted(set(r["product_model"] for r in results if not any(
        rr["chunk_hit"] for rr in results if rr["product_model"] == r["product_model"]
    )))
    report = {
        "round": round_name,
        "total_questions": len(results),
        "models_tested": len(models_tested),
        "chunk_hit_rate": round(hit_rate, 4),
        "models_with_zero_hits": models_failed,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")
    return out_path


if __name__ == "__main__":
    state = exp._load_state()

    print("=== 1라운드 질문 생성 (모델당 최대 5개, section_title 기반) ===")
    questions = build_round1_questions(state)
    print(f"총 질문 수: {len(questions)}개 (모델 {len(set(q['product_model'] for q in questions))}개)")

    print("\n=== 순수 벡터 검색 채점 (LLM 답변 생성 없음) ===")
    results = run_round(questions, state)

    hit_rate = sum(r["chunk_hit"] for r in results) / len(results)
    print(f"\n정답 청크 정확 일치율: {hit_rate:.1%}")

    misses = [r for r in results if not r["chunk_hit"]]
    print(f"놓친 질문: {len(misses)}개")
    for r in misses[:20]:
        print(f"  [{r['product_model']}] {r['question']}  (정답청크: {r['source_chunk_id']}, min_dist={r['min_distance']})")
    if len(misses) > 20:
        print(f"  ... 외 {len(misses) - 20}개")

    save_round("round1", results)
