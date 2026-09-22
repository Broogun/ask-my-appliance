"""[siyeon] exp06 — 최종 구성. LG 30 + 삼성 30 전체 색인 (exp05와 같은 데이터), 검색기는 슬림 정리 후 버전.

실행 (프로젝트 루트에서):
    python experiments/siyeon/exp/exp06_all.py
    → experiments/results/siyeon_exp06.json      (900문항 × ~4초 ≈ 60분. 답변 캐시 data/answer_cache/ 로 중단 재실행 가능)

exp05 → exp06 변경점:
  1. 청킹 수정 — 삼성 냉장고 첫 페이지 증상 누락 복구, RH82M 깨진 목차 수정, LG 드럼세탁기 규격 표를 '제품 규격' 조각으로 분리
  2. 검색 앞에 LLM 1회(understand.py): 기능명을 기능 표의 정식 이름으로("자동청소" → 클린봇) → 표에서 '없음'이면 검색 없이 "이 모델엔 없는 기능"
  3. 뺀 것 — 코드 질문 한정 BM25, 기능명 별칭 사전·'X 있어요?' 정규식, 브랜드·제품군 단어 추정, 태그 가산, 형제 모델 경로.
     시험 후 기각 — LLM 후속 질문 재작성(새 증상을 후속으로 오판), listwise 재정렬(채점 +1.6pt지만 웹은 top-3를 다 보므로 무의미), RRF 동점 처리(악화)
  4. 프롬프트: 등록 가전을 첫 줄에 명시, 수치는 문서 값만(추정 금지), 공용 설명서 값이 여러 개면 모델별 나열

자체 테스트셋 256문항: top-1 92% / hit@3 98% / MRR 0.95 (규칙 기반 기준선 92.5 / 98.8 / 0.95 와 동급, 자연어 규칙 0개).
구현: experiments/siyeon/rag/ (chunking_lg.py, chunking_samsung.py, retrieval.py, understand.py). 설계: docs/overview_lg.md, 실험 기록: docs/experiment_notes.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
load_dotenv(ROOT / ".env")

from experiments.harness.evaluate import run_and_save  # noqa: E402
from siyeon.rag.chunking_lg import build_all_chunks  # noqa: E402
from siyeon.rag.chunking_samsung import build_samsung_chunks  # noqa: E402
from siyeon.rag.retrieval import LGAirconRetriever  # noqa: E402

EXPERIMENT_NAME = "siyeon_exp06"

NOTES = (
    "최종 구성(슬림). exp05와 같은 데이터(LG 30 + 삼성 30)에 청킹 수정 3건(삼성 냉장고 첫 페이지 증상·RH82M 깨진 목차·LG 세탁기 규격 표 분리) + "
    "검색 앞 LLM 1회로 기능명 정규화(기능 표 enum) → 표에서 '없음'이면 검색 없이 없는 기능으로. "
    "exp05에 있던 코드 질문 한정 BM25·기능명 별칭 사전·브랜드 단어 추정은 제거(실측 기여 없음). LLM 후속 질문 재작성·listwise·RRF 동점 처리는 시험 후 기각. "
    "청킹: PDF 목차(TOC) 2·3단계 소제목을 청크 경계로, 고장신고 표는 '증상 1개 = 청크 1개', 에러코드 JSON 6개(LG/삼성 × AC/RF/WM) 1항목=1청크, "
    "목차 없는 구형 PDF는 폰트·번호 기반 폴백 파서(+kiwi 띄어쓰기 복원). "
    "검색: 질문 속 모델명 → 매뉴얼 해석(doc_id 필터) → 에러코드 SQLite 조회 / 기능 표 조회 → 벡터(본문+예상질문) → RRF 8개 → bge-reranker-v2-m3 → top3, "
    "관련확률 10% 미만 후보는 LLM 컨텍스트 제외. 자체 테스트셋 256문항 top-1 92% / hit@3 98%."
)

STRATEGY = {
    "chunking":         "PDF 목차(TOC) 소제목 경계 청킹 + 고장신고는 증상 단위 — LG(프레임메이커)·삼성(InDesign) 브랜드 프로필 2개, 목차 없는 6개는 폰트 기반 폴백",
    "chunking_reason":  (
        "삼성 30개도 27개에 목차 북마크가 있고 소제목이 본문 줄과 거의 100% 일치해 같은 전략이 통함 (챕터는 목차 1단계 페이지, 삼성 세탁기류는 증상|확인/조치 표, "
        "에어컨·냉장고는 표 없는 텍스트라 글자 크기·'~나요!' 패턴으로 증상 감지. 코드|진단|해결방법 표의 코드 셀은 이미지라 진단 문장으로 대체). "
        "LG 최신 매뉴얼 20개 중 17개가 같은 프레임메이커 템플릿(목차 북마크 내장, 챕터명 두 줄 반복, [증상|원인 및 해결책] 표). "
        "목차 2·3단계 제목이 본문 줄과 98~100% 일치해 이 경계를 쓰면 '필터 청소하기', '냉장 & 냉동 > 증상' 같은 질문 단위와 청크 단위가 맞음. "
        "에어컨 실측: 1200자 고정 분할 대비 top-1 83→90%(자체 63케이스). 고장신고는 원인 1행=청크 1개면 같은 증상 행이 top_k를 독식해 증상 단위로 합침. "
        "세탁기 매뉴얼엔 에러 메시지(IE/UE/OE …) 표가 직접 있어 증상 청크로 들어감"
    ),
    "embedding":        "BAAI/bge-m3 (로컬, sentence-transformers) — 본문 청크 5,052개 + 청크별 gpt-4o-mini 생성 예상 질문 4~8개(37,651개) 별도 컬렉션",
    "embedding_reason": (
        "한국어 구어체 질문('말로 조작하려면?')과 격식체 매뉴얼('음성인식 기능') 사이 매칭이 약해 청크마다 '물어볼 법한 질문'을 미리 만들어 임베딩 "
        "(오프라인 1회, 응답 지연 없음, 캐시 chunk_questions.json 커밋). 에어컨 실측 hit@3 92→100%"
    ),
    "db":               "ChromaDB(디스크 영속, brand/category/doc_id 메타데이터) + SQLite(에러코드 6제품군 51항목·별칭 143, 모델↔매뉴얼, 모델별 기능 표)",
    "db_reason":        (
        "등록 가전(doc_id) 필터가 핵심이라 메타데이터 필터가 되는 벡터 DB 필요. 에러코드는 키 조회 문제라 RDB(0ms, 정확도 100%), "
        "'이 모델엔 없는 기능' 판정은 점수로 안 갈려서 부록 기능 표를 룰로. 파일명 모델≠매뉴얼 모델(AC_FQ18GC1EHN.pdf 표엔 GC2/GG1)이라 매핑 표 필요"
    ),
    "retrieval":        (
        "질문 속 모델명 → manual_models로 매뉴얼 해석(= 등록 가전) → ⓪ gpt-4o-mini 1회로 기능명을 기능 표 enum으로 정규화 "
        "→ ① 코드 있으면 SQLite error_codes 조회 1등 고정 ② 기능 표(model_features)가 '없음'이면 검색 없이 없는 기능 "
        "③ doc_id+브랜드·카테고리 공통 에러코드 필터로 벡터(본문)+벡터(예상질문) → RRF 후보 8 → bge-reranker-v2-m3(384토큰) → top3 "
        "(리랭커 확률 50% 미만 에러코드 조각 제거) → ④ 관련확률<10% 컷, 근거 0개면 route=none"
    ),
    "retrieval_reason": (
        "자체 테스트셋 256케이스(LG·삼성 17모델, hard_pair·vocab_gap·없는 기능 포함): top-1 92% / hit@3 98% / MRR 0.95. "
        "에어컨 초기 68케이스: 벡터만 83% → +리랭커+예상질문 88% → +RDB 라우터 90% (hit@3 100%). "
        "BM25 항상 섞으면 66%로 악화(구어체 흔한 단어), 코드 질문 한정도 RDB가 먹어 기여 0 → 제거. 리랭커 후보 12→8, 512→384토큰이 더 빠르고 정확. "
        "후속 질문 재작성(규칙·LLM 둘 다)은 실사용에서 새 증상을 후속으로 오판해 제거 — 멀티턴은 답변 LLM에 대화를 그대로(웹). "
        "listwise(+1.6pt)·RRF 동점 처리(−0.8pt)·제품군 접두어(−7pt)는 채점 후 기각. 남은 부품은 전부 실측 근거가 있는 것만"
    ),
}


def main() -> None:
    chunks = build_all_chunks(include_legacy=True) + build_samsung_chunks()   # LG 30 + 삼성 30
    retriever = LGAirconRetriever(chunks)
    # 질문별 검색 결과·답변 캐시: API 한도(429)로 중단돼도 재실행 때 성공분은 LLM을 다시 부르지 않음 (data/는 git 제외)
    my_answer = retriever.make_answer_fn(None, use_understanding=True,   # LLM 질문 이해(기능명 정규화)
                                         cache_path=ROOT / "data" / "answer_cache" / f"{EXPERIMENT_NAME}.json")

    os.chdir(ROOT)   # run_and_save()가 상대경로 experiments/results/ 에 저장
    run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)


if __name__ == "__main__":
    main()
