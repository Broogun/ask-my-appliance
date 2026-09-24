# -*- coding: utf-8 -*-
"""검색(벡터) 품질 회귀 테스트.

거창한 nDCG/Recall 프레임워크 대신, 이번 세션에서 실제로 테스트했던 질문들을
그대로 재사용한다. LLM 답변 생성(수십 초)은 건너뛰고 search()만 호출해서 빠르게
반복 확인할 수 있게 한다. 청킹/그림-매칭 로직을 건드릴 때마다 이 스크립트를 다시
돌려서 회귀가 없는지 확인한다.

체크 항목:
  - 최대 청크 길이(그림-페이지 매칭 버그로 4,221자까지 커졌던 사고 재발 방지용)
  - 첨부된 그림 개수(과다 연결 감지)
  - 상위 후보 안에 기대 키워드가 실제로 들어있는지
"""
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.query_engine import search

# (질문, 모델 필터 또는 None, 상위 후보 중 최소 1개엔 있어야 하는 키워드)
CASES = [
    ("세탁기 탈수가 안 돼요", None, "탈수"),
    ("냉장고에서 소음이 심하게 나요", "B242S32", "소음"),
    ("냉장고에서 냄새가 나고 있음", "GC-B40BSCQ", "냄새"),
    ("냉장고가 터지기 직전이야", "S825AW35", None),  # 실제 대응 내용 없음 - 키워드 체크 생략, 청크 크기만 확인
]

MAX_CHUNK_LEN_WARN = 1500  # 이보다 크면 그림-매칭 사고 재발 의심


def run():
    total_fail = 0
    for query, model_filter, expect_keyword in CASES:
        print(f"\n=== 질문: {query!r} (model_filter={model_filter}) ===")
        candidates = search(query, top_k=3, model_filter=model_filter)
        if not candidates:
            print("  [FAIL] 검색 결과 없음")
            total_fail += 1
            continue

        keyword_hit = False
        for c in candidates:
            text = c["text"]
            meta = c["metadata"]
            n_figs = len(meta.get("figures", []) or [])
            flag = "  [!] 청크 길이 경고" if len(text) > MAX_CHUNK_LEN_WARN else ""
            print(
                f"  - {meta.get('product_model')} | {meta.get('section_title','')[:30]!r} "
                f"| dist={c['distance']:.3f} | len={len(text)} | figures={n_figs}{flag}"
            )
            if expect_keyword and expect_keyword in text:
                keyword_hit = True
            if len(text) > MAX_CHUNK_LEN_WARN:
                total_fail += 1

        if expect_keyword:
            if keyword_hit:
                print(f"  [OK] 키워드 {expect_keyword!r} 상위 후보에서 발견")
            else:
                print(f"  [FAIL] 키워드 {expect_keyword!r} 상위 후보 어디에도 없음")
                total_fail += 1

    print(f"\n총 실패 항목: {total_fail}")
    return total_fail


if __name__ == "__main__":
    n_fail = run()
    sys.exit(1 if n_fail else 0)
