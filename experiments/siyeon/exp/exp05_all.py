"""[기록용 — 2026-09-15 실행분. experiments/results/siyeon_exp05.json 을 만든 코드. 그 뒤 retrieval.py API가 바뀌어(search_v3·infer_scope 제거) 지금은 실행되지 않는다. 현재 구성은 exp06_all.py]

[siyeon] exp05 — LG 30개 + 삼성 30개 = 평가 대상 60개 매뉴얼 전부 색인. 삼성은 chunking_samsung.py (목차 1단계 페이지로 챕터, 표 3종 + 텍스트형 증상 감지): TOC 청킹 + 벡터(본문·예상질문)/조건부 BM25/리랭커
+ SQLite 라우터(에러코드·기능 유무) + 질문 속 모델명으로 등록 가전 해석.

실행 (프로젝트 루트에서):
    python experiments/siyeon/exp/exp05_all.py
    → experiments/results/siyeon_exp05.json

평가 체계는 experiments/harness/evaluate.py의 MODEL_QUESTIONS (60모델 × 3라운드 × 5문항 = 900개, 질문에 모델명 포함).
  - LG 30모델 450문항: 질문의 모델명 → manual_models → 해당 매뉴얼로 필터해 검색 (등록 가전이 있는 웹 시나리오와 동일)
  - 삼성 30모델 450문항: exp05부터 인덱스에 포함 (InDesign 템플릿 — 목차 1단계 페이지로 챕터 범위, `증상|확인/조치`·`코드|진단|해결방법` 표,
    에어컨·냉장고의 표 없는 문제해결은 글자 크기·'~나요!' 패턴으로 증상 감지. 목차 없는 3개는 폴백 파서)
  - LG 구형 냉장고 3개(CA-H17DC, S825AW35, S835S31): 목차가 없어 폰트(16pt 굵게)·번호('7. 냉장실홈바') 기반 폴백 파서로 청킹,
    띄어쓰기 없는 PDF(S825/S835)는 kiwi로 띄어쓰기 복원. exp03과의 유일한 차이 (이 3모델 45문항 비교용)
소요: 900문항 × ~5초(리랭커 CPU + gpt-4o-mini) ≈ 75~90분. 백그라운드 실행 권장.
처음 실행이면 chroma_lg/ 인덱스 생성(임베딩 2,729청크 + 예상 질문 ≈ 20분)이 먼저 돌고, 이후엔 재사용.

exp01(에어컨 10개, 옛 22문항 체계) → exp03 변경점: 인덱스 범위 LG 전체, 평가 체계 900문항, 모델명 기반 가전 해석.
구현: experiments/siyeon/rag/ (chunking.py, retrieval.py). 설계 설명: experiments/siyeon/docs/overview_lg.md
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

EXPERIMENT_NAME = "siyeon_exp05"

NOTES = (
    "LG 17개 매뉴얼(에어컨·냉장고·김치냉장고·세탁기·건조기·세탁건조기·스타일러) 공용 파이프라인. "
    "PDF 목차(TOC) 2·3단계 소제목을 청크 경계로, 고장신고 표는 '증상 1개 = 청크 1개'(카테고리는 목차에서: 운전/냉장 & 냉동/에러 메시지 …), "
    "세탁건조기의 '사용하기 - 세탁기/건조기' 챕터 접두 매칭, 에러코드 JSON 3개(LG-AC/RF/WM-COMMON) 1항목=1청크. "
    "구형 템플릿 냉장고 3개(CA-H17DC, S825AW35, S835S31)는 폰트·번호 기반 폴백 파서(+kiwi 띄어쓰기 복원)로 청킹 — exp03 대비 유일한 변경. "
    "검색: 질문 속 모델명 → 매뉴얼 해석(doc_id 필터) → 에러코드 SQLite 조회 / 기능 유무 룰 → 벡터(본문+예상질문) + 코드 질문 한정 BM25 → RRF 8개 → bge-reranker-v2-m3 → top3, "
    "관련확률 10% 미만 후보는 LLM 컨텍스트 제외."
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
    "embedding":        "BAAI/bge-m3 (로컬, sentence-transformers) — 본문 청크 2,729개 + 청크별 gpt-4o-mini 생성 예상 질문 4개씩 별도 컬렉션",
    "embedding_reason": (
        "한국어 구어체 질문('말로 조작하려면?')과 격식체 매뉴얼('음성인식 기능') 사이 매칭이 약해 청크마다 '물어볼 법한 질문'을 미리 만들어 임베딩 "
        "(오프라인 1회, 응답 지연 없음). 에어컨 실측 hit@3 92→100%"
    ),
    "db":               "ChromaDB(디스크 영속, brand/category/doc_id 메타데이터) + SQLite(에러코드 6제품군 51항목·별칭 143, 모델↔매뉴얼, 모델별 기능 표)",
    "db_reason":        (
        "등록 가전(doc_id) 필터가 핵심이라 메타데이터 필터가 되는 벡터 DB 필요. 에러코드는 키 조회 문제라 RDB(0ms, 정확도 100%), "
        "'이 모델엔 없는 기능' 판정은 점수로 안 갈려서 부록 기능 표를 룰로. 파일명 모델≠매뉴얼 모델(AC_FQ18GC1EHN.pdf 표엔 GC2/GG1)이라 매핑 표 필요"
    ),
    "retrieval":        (
        "질문 속 모델명 → manual_models로 매뉴얼 해석(= 등록 가전) → ① 코드 있으면 SQLite error_codes 조회 1등 고정 "
        "② 기능명 있으면 model_features/매뉴얼 언급 0회로 '없음' 판정 ③ doc_id+카테고리 공통 에러코드 필터로 벡터(본문)+벡터(예상질문)[+코드 질문만 BM25(kiwi)] "
        "→ RRF 후보 8 → bge-reranker-v2-m3(384토큰) → top3 → 관련확률<10% 컨텍스트 제외. 모델명 없으면 브랜드·제품군 단어로 범위 추정"
    ),
    "retrieval_reason": (
        "에어컨 자체 테스트셋 68케이스 실측: 벡터만 83% → +리랭커+예상질문 88% → +RDB 라우터 90% (hit@3 100%). "
        "BM25 항상 섞으면 66%로 악화(구어체 흔한 단어)라 코드 질문 한정. 리랭커 후보 12→8, 512→384토큰이 더 빠르고 정확. "
        "새 평가 체계는 질문에 모델명이 있어 웹의 '등록 가전' 시나리오를 그대로 재현 가능"
    ),
}


def main() -> None:
    chunks = build_all_chunks(include_legacy=True) + build_samsung_chunks()   # LG 30 + 삼성 30
    retriever = LGAirconRetriever(chunks, infer_scope_when_no_appliance=True)
    # 질문별 검색 결과·답변 캐시: API 한도(429)로 중단돼도 재실행 때 성공분은 LLM을 다시 부르지 않음 (data/는 git 제외)
    my_answer = retriever.make_answer_fn(None, search_fn=retriever.search_v3,
                                         cache_path=ROOT / "data" / "answer_cache" / f"{EXPERIMENT_NAME}.json")

    os.chdir(ROOT)   # run_and_save()가 상대경로 experiments/results/ 에 저장
    run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)


if __name__ == "__main__":
    main()
