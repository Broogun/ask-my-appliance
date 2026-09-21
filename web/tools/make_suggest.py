"""첫 화면 기본 질문 고르기 — 제품군별 후보 질문을 그 제품군의 모든 모델에 실제로 검색해 보고, 전부에서 근거가 나오는 질문 4개를 고른다.

    python web/tools/make_suggest.py          # GPU ~10분 (질문 이해 결과는 understand_cache.json에 캐시)

출력: 제품군별 커버율 표 + app.js에 붙일 `const SUGGEST = {...}` 블록. 검색기나 후보 목록을 바꿨으면 다시 실행한다.
(모델별로 질문을 빼는 방식은 안 쓴다 — 어떤 모델을 골라도 같은 4개가 뜨고, 4개 전부 답이 나온다)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from siyeon.rag.chunking_lg import build_all_chunks  # noqa: E402
from siyeon.rag.chunking_samsung import build_samsung_chunks  # noqa: E402
from siyeon.rag.retrieval import LGAirconRetriever  # noqa: E402
from siyeon.rag.understand import understand  # noqa: E402

N_PICK = 4
# 제품군별 후보 — 어느 모델이든 설명서에 있을 법한 것들. 앞쪽이 우선(커버율이 같으면 앞의 것을 고른다)
CANDIDATES = {
    "에어컨":   ["필터 청소는 어떻게 해요?", "찬바람이 안 나와요", "예약 기능 설정하는 법", "이상한 냄새가 나요",
                "리모컨이 안 돼요", "에어컨에서 물이 새요", "소음이 심해요", "전원이 안 켜져요"],
    "냉장고":   ["냉장실 음식이 얼어요", "냉장고에서 소리가 나요", "냉장고가 시원하지 않아요", "냉장고 청소는 어떻게 해요?",
                "냉장실 온도 조절하는 법", "문이 잘 안 닫혀요", "냉장고 옆면이 뜨거워요", "성에가 생겨요", "문 열림 알림 기능 있어요?", "냉장고에서 냄새가 나요"],
    "김치냉장고": ["김치 숙성은 어떻게 해요?", "김치가 빨리 시어요", "온도 설정 방법", "김치냉장고에서 소리가 나요",
                "김치냉장고가 시원하지 않아요", "문이 잘 안 닫혀요", "청소는 어떻게 해요?", "경보음이 울려요", "전원이 안 켜져요"],
    "세탁기":   ["탈수가 안 돼요", "UE 떴어요", "세제는 어디에 넣어요?", "세탁통 청소 방법",
                "물이 안 빠져요", "세탁기에서 소음이 나요", "문이 안 열려요", "전원이 안 켜져요"],
    "세탁건조기": ["건조가 잘 안 돼요", "문이 안 열려요", "탈수가 안 돼요", "세제는 어디에 넣어요?",
                "세탁통 청소 방법", "물이 안 빠져요", "세탁기에서 소음이 나요", "탈수만 하려면?", "세제 자동투입 설정", "UE 떴어요"],
    "건조기":   ["건조가 너무 오래 걸려요", "필터 청소는 어떻게 해요?", "옷에 보풀이 묻어나요", "이불 건조 코스 있어요?",
                "건조기에서 소음이 나요", "옷이 덜 말랐어요", "문이 안 열려요", "전원이 안 켜져요"],
    "스타일러":  ["물통에 물 채우는 법", "옷에서 냄새가 나요", "연기가 나와요", "예약 설정하는 법",
                "옷이 덜 말랐어요", "소음이 심해요", "물통 비우는 법", "문이 안 열려요"],
    "슈드레서":  ["신발 관리 코스 뭐 있어요?", "물통 비우는 법", "냄새가 안 빠져요", "소음이 심해요",
                "물통에 물 채우는 법", "문이 안 열려요", "전원이 안 켜져요", "예약 설정하는 법"],
}


def main() -> None:
    r = LGAirconRetriever(build_all_chunks() + build_samsung_chunks(), verbose=False)
    r.reranker.predict([("워밍업", "리랭커")])
    by_type: dict[str, list[str]] = {}
    for doc in r.known_doc_ids:
        by_type.setdefault(r.appliance_for(doc)["product_type"], []).append(doc)
    picked: dict[str, list[str]] = {}
    for pt, cands in CANDIDATES.items():
        docs = by_type.get(pt, [])
        cover: dict[str, list[str]] = {q: [] for q in cands}      # 질문 → 근거 안 나온 모델들
        for doc in docs:
            appl = r.appliance_for(doc)
            for q in cands:
                und = understand(q, appl)
                res = r.retrieve(q, doc_id=doc, feature=und.feature)
                if not res.used:
                    cover[q].append(doc)
        ranked = sorted(cands, key=lambda q: (len(cover[q]), cands.index(q)))
        picked[pt] = ranked[:N_PICK]
        print(f"\n[{pt}] 모델 {len(docs)}개")
        for q in ranked:
            miss = cover[q]
            mark = "★" if q in picked[pt] else " "
            print(f"  {mark} {len(docs)-len(miss):2d}/{len(docs)}  {q}" + (f"   (안 나옴: {', '.join(miss)})" if miss else ""))
    print("\n// app.js 에 붙일 것 — 모든 모델에서 근거가 나오는 질문만 (make_suggest.py)")
    print("const SUGGEST = {")
    for pt, qs in picked.items():
        print(f"  '{pt}': [{', '.join(repr(q) for q in qs)}],".replace('"', "'"))
    print("};")


if __name__ == "__main__":
    main()
