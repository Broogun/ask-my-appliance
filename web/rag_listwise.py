"""CrossEncoder 리랭킹을 listwise LLM 리랭킹으로 교체한 검색기.

팀 4명 비교(RAG 4종 구성 비교 문서, 2026-09-18)에서 유일하게 cross-validated된
결론만 수술적으로 반영한다: 박형건/엄지민이 독립적으로 CrossEncoder(pointwise)
를 버리고 LLM listwise로 옮겼고, 그 이유(어휘만 겹치는 오답에 정답보다 높은
점수를 주는 함정 - 박형건_exp02.py `_llm_rerank` 문서 참고)가 시연님의
CrossEncoder 구현에도 동일하게 적용될 가능성이 높다.

나머지(RDB 에러코드/모델↔매뉴얼 매핑/모델별 기능유무, 본문+예상질문 RRF,
관련도 컷)는 시연님의 검증된 구현을 그대로 쓴다 - 이 부분들은 팀 비교에서
우리 것보다 더 풍부하다고 확인됐다(우리는 에러코드 RDB만 있고 모델매핑/
기능유무 테이블이 없었음). 검색 엔진 전체를 갈아끼우지 않고 리랭킹 단계
하나만 바꾸는 이유가 이것 - 검증 안 된 부분까지 손대면 그만큼 회귀 위험만
커진다.

`LGAirconRetriever.search_dense()`만 오버라이드한다 - `_route()`의 벡터 검색
경로와 에러코드 보강 경로가 전부 이 메서드를 통해서만 리랭킹을 거치므로,
한 곳만 바꾸면 전체 검색 흐름에 자동 반영된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from siyeon.rag.retrieval import LGAirconRetriever, N_CANDIDATES  # noqa: E402
from 박형건_exp02 import _llm_rerank  # noqa: E402


class ListwiseRetriever(LGAirconRetriever):
    """search_dense()의 CrossEncoder 리랭킹 단계만 listwise LLM(_llm_rerank)으로 교체."""

    def search_dense(
        self, query: str, doc_id=None, top_k: int = 3, n_candidates: int = N_CANDIDATES,
    ) -> list[tuple[dict, float]]:
        fused = self._candidates(query, doc_id, n_candidates)
        if not fused:
            return []
        # _llm_rerank는 거리로 1차 압축(top 35) 후 LLM에 보여주는데, 여기 후보는
        # 이미 RRF로 n_candidates(기본 8)개까지 좁혀진 상태라 전부 동일 거리를
        # 줘도 순서에 영향 없음(35 밑이라 전부 통과) - 실제 순위는 RRF가 아니라
        # LLM 판정이 최종 결정한다.
        candidates = [
            {"id": cid, "text": self.chunk_by_id[cid]["text"], "distance": 0.5, "metadata": {}}
            for cid in fused
        ]
        picked = _llm_rerank([query], candidates, {}, top_k=top_k)
        # picked는 순위 정보만 있고 점수(확률)가 없다 - CONTEXT_CUTOFF(0.9)보다
        # 확실히 낮은 값을 순서대로 매겨서 "픽 됐으면 관련 있다"는 의미를 그대로
        # 살린다. _llm_rerank는 파싱 실패/전부 무관 판정("NONE")이어도 빈 리스트
        # 대신 거리 순 상위 top_k로 안전하게 폴백한다(exp02의 기존 설계) - 여기서도
        # 그대로 상속된다. "진짜 근거 없음"은 apply_cutoff()의 관련도 컷이 걸러낸다.
        return [(self.chunk_by_id[p["id"]], 0.1 + i * 0.05) for i, p in enumerate(picked)]
