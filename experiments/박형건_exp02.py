"""박형건 exp02 — exp01 + 삼성 문서 청킹 버그 수정.

2026-09-18 기준 팀 통합 RAG 파이프라인의 베이스라인으로 채택됨. 청킹(700자/15%
오버랩, TOC 기반)·리랭킹(listwise LLM)·확신 게이트(3단계)는 다른 3인의 구현과
비교 + 자체 청킹 변형 실험(오버랩 0%/25%, 문맥반복 추가) 전부와 교차검증해서
이 파일의 현재 설계가 그대로 유지됐다 - 자세한 비교/실험 과정은
`NOTES_ARCHITECTURE.md`(로컬 전용) 참고. RDB는 에러코드 정확매칭만 있고
모델-매뉴얼 매핑/기능유무 테이블은 아직 없음(백로그, 시연 스키마 참고 예정).
BM25는 실측 근거 부족(팀 내 유일한 A/B 테스트 결과가 "효과 없어서 제거")으로
보류 상태.

exp01과 검색/생성 코드는 동일하다. 차이는 전부 `experiments/rag_tutorial.ipynb`에서
재구축한 벡터 DB(`chroma_db/`) 쪽에 있다 — 삼성 PDF를 자세히 검토하다가 발견한
2가지를 고쳤다:

1. `AC_AF19TX978MZ3N.pdf`/`RF_RH82M9152SL.pdf`는 TOC 북마크 자체가 손상돼 있어서
   (페이지 번호가 0이거나 중복된 목차에 엉뚱한 페이지가 찍혀 있음) 섹션 하나가
   문서 거의 전체(30~40페이지, 3~4만자)를 삼켜버리는 사고가 있었다. 섹션 span이
   20페이지를 넘으면 TOC 손상으로 보고 안전하게 1페이지로 축소하도록 방어 로직 추가.
2. 삼성 에어컨은 LG와 트러블슈팅 챕터 이름이 달라서("서비스를 요청하기 전에") 기존
   키워드 목록에 안 걸려 냉방안됨/소리/냄새 등 증상별 항목이 세분화되지 않고
   있었다 - 키워드 추가 + 자기 자신뿐 아니라 하위 트리 전체에서 키워드를 찾도록
   개선(중첩된 TOC에서도 놓치지 않게).

수정 후 전체 컬렉션 5,713개 -> 5,847개 청크로 재구축됨 (`chroma_db/`, 노트북에서
재실행 완료). 이 파일은 그 결과를 그대로 읽기만 한다.

사용법:
  python experiments/박형건_exp02.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import chromadb
import torch
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.evaluate import SYSTEM_PROMPT, run_and_save  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")
# timeout 명시 - 기본 SDK 타임아웃(10분)이라 네트워크가 걸리면 재시도 로직
# (generate_openai의 RateLimitError 백오프)이 발동하기도 전에 스레드가
# 무한정 붙잡혀 있을 수 있음. tag_content_type.py에서 같은 문제를 이미 겪고
# 고쳤는데(2026-09-16), 정작 이 공용 클라이언트(my_answer/search_v2/
# round1_llm_judge.py가 전부 공유)엔 반영을 안 해서 round2 재판정 작업이
# 30분 넘게 멈춰있었던 걸로 확인됨(2026-09-17) - 여기도 동일하게 반영.
OAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=30.0)

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"
ERROR_DB_PATH = ROOT / "data" / "error_codes.db"

# 모든 LLM 호출은 generate_openai(gpt-4o-mini)로 통일(CLAUDE.md 통일 사항 준수).
# SYSTEM_PROMPT는 pipeline.evaluate에서 그대로 가져다 쓰고 있어 수정하지 않았다.

# ── 실험 메타 정보 ────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "박형건_exp02"

NOTES = (
    "exp01 대비 3가지 수정. (1) 삼성 문서 청킹 버그: TOC 북마크가 손상된 2개 PDF"
    "(AC_AF19TX978MZ3N, RF_RH82M9152SL)가 섹션 하나로 문서 전체를 삼키던 문제를 "
    "span 캡(20페이지 초과 시 1페이지로 축소)으로 방어, 삼성 에어컨 트러블슈팅 "
    "챕터명('서비스를 요청하기 전에')이 기존 키워드 목록에 없어 증상별 세분화가 "
    "안 되던 걸 키워드 추가 + 하위 트리 전체 탐색으로 해결(컬렉션 5,713 -> 5,847개 "
    "청크로 재구축). (2) candidates의 id를 product_model+page 조합(수제작, 충돌 "
    "가능 - 실제로 5,847개 중 1,296개가 겹쳤음)에서 Chroma의 실제 고유 청크 id로 "
    "교체, search_v2의 중복 제거도 같은 id 기준으로 정확해짐. (3) 질문에 카테고리/"
    "브랜드가 명시돼 있으면(예: '에어컨') 벡터 검색에 where 필터로 반영 - 실제로 "
    "'에어컨에서 CH04 에러' 질문에 세탁기/건조기 청크가 (distance 0.98~1.02로 "
    "나쁜데도) top_k를 채우려고 섞여 들어온 걸 확인해서 수정함. 이하 청킹/임베딩/RDB/"
    "검색 전략은 exp01과 동일: PDF 내장 TOC로 재귀 청킹(중복 헤딩은 조상 경로 "
    "접두어로 구분, 문제해결/고장진단 챕터는 키워드 매칭으로 max_depth 무시하고 "
    "강제 세분화) + TOC 자체가 없는 구형 PDF는 다단 레이아웃이라 안전하게 페이지 "
    "단위 폴백. 임베딩은 dragonkue/BGE-m3-ko"
    "(한국어 검색 벤치마크로 파인튜닝된 BGE-M3, GPU) 하나로 단일 Chroma 컬렉션에 "
    "brand/category/product_model 메타데이터와 함께 적재. 에러코드는 벡터 유사도보다"
    " 정확 매칭이 신뢰도가 높아 별도 SQLite RDB로 분리. 검색 시 질문에서 에러코드 "
    "토큰을 뽑아 RDB부터 확인하고, '에러 확인 방법'/'이상한 소리가 나요'류 일반적 "
    "질문은 모델별 대표 챕터를 구조적으로 직접 매칭하며, gpt-4o-mini로 질문을 "
    "재구성(decomposition)한 뒤 서브 질문별 벡터+합성질문(FAQ) 검색 결과를 합쳐서 "
    "listwise LLM 판정(gpt-4o-mini)으로 최종 top_k를 선정한다."
)

STRATEGY = {
    "chunking": "TOC 재귀 분할(max_depth=2) + 중복 헤딩 조상경로 접두어 + 문제해결 키워드 강제 세분화(하위 트리 전체 탐색) + span 캡(20페이지 초과 시 TOC 손상으로 간주해 1페이지로 축소) + 700자/15% 오버랩 슬라이싱, TOC 없는 문서는 인쇄된 목차(찾아보기) 파싱 우선 시도 후 실패 시 폰트 크기 기반 페이지 내부 재분할",
    "chunking_reason": "매뉴얼이 이미 TOC로 의미 단위 구조화돼 있어 고정길이 분할보다 유리. 문제해결 챕터는 하위 항목이 뭉쳐 나오는 걸 실제 확인해서 강제 세분화 추가 - 삼성은 챕터명이 달라서('서비스를 요청하기 전에') 키워드 확장 필요. 삼성 PDF 2개는 TOC 북마크 자체가 손상돼(page=0/중복) 섹션이 문서 전체를 삼키는 걸 실제로 확인해서 span 캡 방어 로직 추가. TOC 없는 구형 PDF 5개 중 2개는 인쇄된 목차의 물리:인쇄 페이지가 1:1로 정확히 맞는 걸 확인해 그대로 파싱, 나머지 3개는 1:1이 아니라 오귀속 위험이 있어(실측 확인) 폰트 크기 기반 폴백 유지.",
    "embedding": "dragonkue/BGE-m3-ko (GPU)",
    "embedding_reason": "BGE-M3(다국어)를 한국어 검색 벤치마크(Ko-StrategyQA, AutoRAGRetrieval, MIRACLRetrieval)로 파인튜닝한 로컬/무료 모델. GPU로 60개 PDF 6,100여 청크를 1분 내 임베딩 가능해서 반복 실험에 부담이 없음.",
    "db": "ChromaDB 컬렉션 1개(appliance_manuals 원본청크) + SQLite 에러코드 RDB(error_solutions/error_codes). 합성질문(FAQ) 컬렉션(appliance_faq)은 2026-09-18 제거 - 97문항 재검증에서 실측 효과 0(NOTES 20.1)",
    "db_reason": "브랜드/카테고리를 특정하지 않는 질문도 많아서(COMMON_QUESTIONS 참고) 컬렉션을 6개로 쪼개는 대신 단일 컬렉션 + 메타데이터 필터를 선택. 에러코드는 근사 매칭(임베딩)보다 정확 매칭이 훨씬 신뢰도가 높아 벡터 DB와 별도로 관계형 스키마로 분리.",
    "retrieval": "질문에서 카테고리/브랜드/모델명 감지해 where 필터로 반영 -> 에러코드처럼 보이는 토큰을 뽑아 RDB 정확 매칭 -> '에러 확인 방법'/증상 보고형 질문이면 모델별 대표 트러블슈팅 챕터를 구조적으로 직접 매칭 -> gpt-4o-mini로 질문 재구성(decomposition) -> 서브 질문마다 모델 코드 텍스트 제거 후 원본청크 검색(top_k=40, content_type 태깅 매칭 시 추가 경로 병행) -> bi-encoder 거리로 상위 20개 압축 후 listwise LLM 판정(gpt-4o-mini)으로 최종 top_k=3 선정 -> 거리 임계값+애매구간 LLM 관련성 검증으로 응답 보류 여부 결정 -> gpt-4o-mini로 최종 답변 생성 후 페이지 출처 자동 부착",
    "retrieval_reason": "에러코드 조회는 정확도가 생명이라 임베딩 유사도만으로는 부족해 RDB 정확 매칭을 우선함. 질문에 모델 코드가 섞여 있으면 리랭킹 점수가 노이즈로 뭉개지는 걸 실측 확인해서 제거. cross-encoder 리랭커(pointwise)가 어휘만 겹치는 오답에 정답보다 훨씬 높은 점수를 주는 걸 반복 확인(후보 수를 줄여도 동일)해서 listwise LLM 판정으로 교체 - 후보끼리 직접 비교시키고 내용을 읽고 판단하게 해서 어휘 유사도 함정을 피함. '에러' 같은 단어가 문서에 아예 없어 생성 단계가 과도하게 보수적으로 답변을 거절하는 사례가 있어, 이런 질문 유형은 구조적 챕터 매칭으로 우회. 응답 보류는 거리만으로 판단하면 관련 없어도 top_k를 억지로 채우는 문제를 막기 위함.",
}


# ════════════════════════════════════════════════════════════════════════════
# 모델/DB 로딩 (my_answer가 22번 호출되는 동안 한 번만 로드하도록 지연 초기화)
# ════════════════════════════════════════════════════════════════════════════

_state: dict = {}


def _load_state() -> dict:
    if _state:
        return _state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[박형건_exp02] 임베딩 device: {device}")

    _state["embed_model"] = SentenceTransformer("dragonkue/BGE-m3-ko", device=device)
    # bge-reranker-v2-m3-ko(cross-encoder) 로딩 제거 - listwise LLM 리랭킹으로
    # 교체됨(#20, _llm_rerank 참고). pointwise 방식이 어휘만 겹치는 오답에
    # 정답보다 높은 점수를 주는 걸 반복 확인해서 신뢰할 수 없다고 결론 내림.

    if os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql")):
        # 공유 DB(Supabase, pgvector) 모드 - Chroma 컬렉션과 같은 모양(get/query/count)으로 감싼 어댑터
        from web.vecstore import PgCollection
        _state["collection"] = PgCollection()
        print("[박형건_exp02] 벡터 저장소: Postgres(pgvector)")
    else:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _state["collection"] = client.get_collection("appliance_manuals")

    # 합성 질문(FAQ) 컬렉션(#14)은 2026-09-18 제거 - retrieval_metrics_offline.py의
    # use_faq 토글로 97문항(FAQ가 실제 생성된 10모델 전체) 재검증한 결과 MRR·
    # Recall@K가 소수점까지 동일, 순위가 바뀐 문항 0개로 실측 효과가 0이었음
    # (NOTES_ARCHITECTURE.md 20.1). #21과 같은 "증명 안 되면 뗀다" 원칙 적용.

    # 질문에 정확한 모델명이 포함된 경우(예: MODEL_QUESTIONS는 항상 그럼) where
    # 필터에 반영하기 위해 알고 있는 모델명 목록을 미리 뽑아둔다 - 실제로 확인:
    # 카테고리 필터만으로는 같은 계열(예: "세탁기"=드럼/건조기/스타일러 등)
    # 안에서 다른 모델의 콘텐츠가 더 가깝게 나오는 경우가 있었음(스타일러 질문에
    # 엉뚱한 모델의 표지 페이지가 근거로 잡힌 사례).
    all_docs = _state["collection"].get(include=["metadatas"])
    all_meta = all_docs["metadatas"]
    _state["known_models"] = sorted({m["product_model"] for m in all_meta}, key=len, reverse=True)

    # 모델별 "트러블슈팅 대표 챕터" 위치 미리 인덱싱 (버그 #18 대책).
    # "에어컨 에러 메시지 확인하는 방법" 같은 질문은 "에러"라는 단어가 문서에
    # 없어서(실측: LG 에어컨 20개 중 6개만 존재) 벡터검색이 어휘 갭으로 헤매거나
    # 생성 단계에서 "문서에 없다"고 과도하게 보수적으로 거절하는 걸 실제로
    # 확인함(round1 LLM판정 FALSE_DECLINE 23개 중 8개가 이 패턴). 근데 실제로는
    # "고장신고 전 확인하기"(LG)/"서비스를 요청하기 전에"(삼성) 같은 대표
    # 트러블슈팅 챕터가 이미 존재하고 검색도 그걸 찾아오고 있었음 - 문제는
    # "에러"↔"고장" 어휘 연결을 LLM이 안 해준 것. 그래서 이 챕터를 구조적으로
    # (임베딩 유사도에 기대지 않고) 바로 찾아서 최우선 후보로 꽂아준다.
    # 실제로 확인: 타이틀 변형이 예상보다 훨씬 다양함(모델/카테고리마다 다름).
    # FALSE_DECLINE이었던 8개 에어컨 모델(FQ16FV6EDN 등)을 직접 확인해서
    # "LG ThinQ로 고장 진단하기"/"신호음으로 고장 진단하기"가 이 모델들의
    # 사실상 유일한 진단 경로임을 확인하고 추가함 - 화면에 에러코드가 안 뜨는
    # 모델은 오히려 이런 방식(앱 진단, 신호음)이 "에러 확인 방법"의 실제 정답임.
    _TS_TITLES = (
        "고장신고 전 확인하기", "고장 신고 전 확인하기", "고장신고 전 확인사항",
        "서비스를 요청하기 전에", "제품 이상 및 고장이 발생했을 때",
        "안심하세요! 고장이 아닙니다", "점검 문자 발생 시 대처하기",
        "LG ThinQ로 고장 진단하기", "LG ThinQ로 고장의 원인 진단하기",
        "신호음으로 고장 진단하기", "신호음으로 고장의 원인 진단하기",
        "제품에서 고장 진단하기", "고장 감지 기능",
    )
    ts_map: dict[str, list[str]] = {}
    for chunk_id, meta in zip(all_docs["ids"], all_meta):
        if meta.get("section_title") in _TS_TITLES:
            ts_map.setdefault(meta["product_model"], []).append(chunk_id)
    _state["troubleshooting_chunks"] = ts_map

    # 냉장고/김치냉장고 "제어판" 대표 챕터 위치 미리 인덱싱 (#19 잔여 대책).
    # "온도 설정 방법 알려줘" 질문의 FALSE_CONFIDENCE 5건(B242S32/M402ND/
    # RS84B5041WW/RT53DG7A1CWW/RP20C3111S9)을 직접 확인 - LG는 "제어창",
    # 삼성은 "버튼과 표시부"/"기능 선택 방법"이 실제 온도조절 절차가 있는 곳.
    # RT53DG7A1CWW는 이런 챕터 자체가 없고 매뉴얼 전체가 증상별 FAQ 형태라
    # 이 방식으로 못 잡음(별도 문제로 남겨둠 - NOTES 참고).
    _PANEL_TITLES = ("제어창", "버튼과 표시부", "기능 선택 방법")
    panel_map: dict[str, list[str]] = {}
    for chunk_id, meta in zip(all_docs["ids"], all_meta):
        if meta.get("section_title") in _PANEL_TITLES:
            panel_map.setdefault(meta["product_model"], []).append(chunk_id)
    _state["panel_chunks"] = panel_map

    # "제품규격"(스펙표) 하드코딩 타이틀 매칭(#21)은 2026-09-17 제거됨 - 태깅
    # (#22, content_type=SPEC)만으로 동일 트리거 63문항 커버리지를 97~100%
    # 재현하는 걸 실측 확인(tagging_vs_structural_coverage.py, pool=35 기준
    # content_type 필터 단독 100.0%) - 하드코딩 타이틀 리스트 없이도 완전히
    # 대체 가능해서 정리함. search_v2()의 content_type 검색 경로가 이 역할을
    # 대신함(detect_content_type -> SPEC).

    conn = sqlite3.connect(str(ERROR_DB_PATH))
    _state["conn"] = conn
    _state["all_codes"] = {
        row[0].upper(): row[0]
        for row in conn.execute("SELECT DISTINCT code FROM error_codes").fetchall()
    }
    return _state


def generate_openai(
    system_prompt: str, user_message: str, model: str = "gpt-4o-mini",
    temperature: float | None = None,
) -> str:
    """로컬 Ollama가 GPU 경합 때문에 너무 느려서(측정 작업과 동시에 돌리면 수십 분대)
    decompose_query 같은 짧고 구조적인 호출은 gpt-4o-mini로 돌린다 - 값싸고
    (수백~수천 호출해도 $1 미만) GPU를 안 써서 로컬 임베딩/리랭커 작업과도 안 겹친다."""
    kwargs = {"temperature": temperature} if temperature is not None else {}
    for attempt in range(5):
        try:
            resp = OAI.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAI RateLimitError 재시도 5회 초과")


# ════════════════════════════════════════════════════════════════════════════
# RDB(에러코드) 조회
# ════════════════════════════════════════════════════════════════════════════

def find_error_codes_in_query(query: str, state: dict) -> list[str]:
    """질문에서 에러코드로 보이는 토큰(영문+숫자, 2자 이상)을 뽑아 RDB에 존재하는 것만 반환."""
    tokens = re.findall(r"[A-Za-z0-9()]{2,}", query)
    found = []
    for t in tokens:
        code = state["all_codes"].get(t.upper())
        if code and code not in found:
            found.append(code)
    return found


def lookup_code(conn: sqlite3.Connection, code: str) -> list[dict]:
    """코드 하나로 관련 조치법을 전부 조회한다 (브랜드가 달라도 같은 코드면 다 나옴)."""
    rows = conn.execute(
        """
        SELECT s.solution_id, s.brand, s.category, s.title, s.content
        FROM error_codes c JOIN error_solutions s ON c.solution_id = s.solution_id
        WHERE c.code = ?
        """,
        (code,),
    ).fetchall()
    return [
        {"solution_id": r[0], "brand": r[1], "category": r[2], "title": r[3], "content": r[4]}
        for r in rows
    ]


# ════════════════════════════════════════════════════════════════════════════
# 벡터 검색 + 질문 재구성(decomposition) + 리랭커
# ════════════════════════════════════════════════════════════════════════════

DECOMPOSE_SYSTEM_PROMPT = (
    "너는 사용자 질문을 검색하기 좋은 형태로 정리하는 어시스턴트야. "
    "질문에 서로 다른 주제(예: 오류코드 문의 + 사용법 문의)가 섞여 있으면 "
    "각 주제를 독립된 질문으로 분리해. 하나의 주제면 원래 질문 그대로 하나만 담아. "
    "반드시 JSON 배열만 출력해. 다른 설명이나 마크다운 없이 [\"질문1\", \"질문2\"] 형식만."
)


def decompose_query(query: str) -> list[str]:
    """복합 질문이면 주제별로 쪼개서 리스트로 반환, 아니면 [query] 그대로.

    temperature=0으로 고정 - 분해는 창의성이 필요 없는 구조적 판단이라, 매번
    다른 표현으로 재구성되면 이후 검색 결과 전체가 흔들린다(실제로 확인함).

    gpt-4o-mini로 호출 - 로컬 Ollama는 임베딩/리랭커와 같은 GPU를 나눠 쓰느라
    측정 작업 중엔 극도로 느려짐(실제로 300문항 테스트가 GPU 경합으로 정체됨).
    OpenAI는 GPU를 안 쓰고 짧은 구조적 호출이라 비용도 미미함(사용자 승인,
    2026-09-16)."""
    raw = generate_openai(DECOMPOSE_SYSTEM_PROMPT, query, temperature=0.0)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [query]
    try:
        parts = json.loads(match.group(0))
        parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        return parts if parts else [query]
    except json.JSONDecodeError:
        return [query]


def _strip_known_models(text: str, state: dict) -> str:
    """질문 텍스트에서 알고 있는 제품 모델명(영숫자 코드)을 제거한다.

    실제로 확인: 크로스인코더 리랭커에 "M402ND 냉장고 온도 설정 방법 알려줘"처럼
    모델 코드가 섞인 질문을 그대로 넣으면 정답 청크 점수가 0.000014까지 뭉개짐
    (raw logit -13대) - 같은 청크, 모델명만 뺀 질문으로는 0.9964. bi-encoder
    벡터검색 순위도 소폭 개선됨(예: GC-B40BSCQ '제어창' 청크 22위->12위). 모델
    스코프는 이미 `where={"product_model": ...}` 필터로 좁혀놨으니 텍스트에
    모델 코드가 또 들어갈 필요가 없고, 오히려 영숫자 토큰이 노이즈로 작용해
    bi-encoder/cross-encoder 둘 다의 판단을 해친다."""
    stripped = text
    for model in state.get("known_models", []):
        if model in stripped:
            stripped = stripped.replace(model, "").strip()
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or text  # 모델명 제거 후 텍스트가 통째로 사라지면(드묾) 원본 유지


def _vector_search(query: str, state: dict, top_k: int, where: dict | None) -> list[dict]:
    query_emb = state["embed_model"].encode([query], normalize_embeddings=True).tolist()
    result = state["collection"].query(query_embeddings=query_emb, n_results=top_k, where=where)
    candidates = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        candidates.append({"id": chunk_id, "text": doc, "metadata": meta, "distance": dist})
    return candidates


RERANK_JUDGE_PROMPT = (
    "너는 검색 결과를 심사하는 심사위원이야. [질문]에 가장 정확하게 답할 수 있는 "
    "후보를 [후보] 목록에서 골라서, 관련도가 높은 순서로 번호만 나열해. 최대 "
    "{top_k}개까지만. 관련 없는 후보는 아예 빼도 돼. 다른 설명 없이 번호만 콤마로 "
    "구분해서 출력해(예: 2,0,5). 관련 있는 게 하나도 없으면 'NONE'만 출력해."
)


def _llm_rerank(sub_queries: list[str], candidates: list[dict], state: dict, top_k: int) -> list[dict]:
    """listwise LLM 리랭킹 - 후보 전체를 한 번에 보여주고 그 안에서 비교시켜 선택.

    원래는 cross-encoder(pointwise - 후보를 하나씩 독립적으로 채점)를 썼는데,
    실제로 확인해보니 어휘는 겹치지만 의미상 틀린 오답에 정답보다 훨씬 높은
    점수를 주는 사례가 반복됐다(#20) - 예: "M402ND 온도 설정 방법" 질문에서
    트러블슈팅 문단("설정 온도를 확인했나요? 낮게 설정하세요" - "설정"/"온도"
    단어만 겹침)에 0.976점을 주고, 진짜 정답("상칸 버튼을 눌러 온도를 조절"
    - 방법은 정확히 설명하지만 "설정"이란 단어를 안 씀)에는 0.098점을 줬다.
    후보 수를 줄여도(#18/#19 구조적 챕터 매칭처럼 3~6개로 좁혀도) 똑같이
    발생해서 후보 수의 문제가 아니라 판단 방식(pointwise, 어휘 유사도 의존)
    자체의 한계로 결론 내림. listwise(후보끼리 직접 비교)+LLM(어휘가 아니라
    내용을 읽고 판단)으로 교체.

    1차 압축 폭(bi-encoder distance 기준)은 20이었다가 35로 올림 - 97문항
    검증에서 남은 미스 7개 중 5개가 정답이 21~32등이라 top20 컷에서 LLM이
    보기도 전에 잘려나간 걸 확인함(예: DJ30T9500FE "문이 안 열려요" 32등,
    FC2521KX6C "응축수 비우는 방법" 22등). LLM 리랭킹 자체는 잘 판단하는데
    그 앞단 압축이 너무 타이트했던 것.
    """
    if not candidates:
        return candidates
    if len(candidates) <= top_k:
        return candidates
    # bi-encoder 거리로 1차 압축(비용 억제) - cross-encoder는 신뢰 못 해서 필터로도 안 씀.
    pool = sorted(candidates, key=lambda c: c["distance"])[:35]
    # 청크 전문을 그대로 보여준다 - 200자로 자르면(700자 청크의 앞부분만) 진짜
    # 답이 되는 내용이 뒷부분에 있을 때 리랭커가 못 보고 다른 청크를 잘못
    # 고를 위험이 있음(round1_llm_judge.py에서 같은 종류의 절단 버그를 먼저
    # 발견해서 고쳤고, 2026-09-17 여기(실제 프로덕션 리랭킹 경로)에도 동일하게
    # 있는 걸 확인해서 반영). pool=35 x 최대 700자라 gpt-4o-mini 비용 영향은 미미함.
    listing = "\n\n".join(
        f"[{i}] ({c['metadata'].get('section_title', '')}) {c['text']}"
        for i, c in enumerate(pool)
    )
    query_block = "\n".join(sub_queries)
    raw = generate_openai(
        RERANK_JUDGE_PROMPT.format(top_k=top_k),
        f"[질문]\n{query_block}\n\n[후보]\n{listing}",
        temperature=0.0,
    )
    indices, seen, picked = [int(x) for x in re.findall(r"\d+", raw)], set(), []
    for i in indices:
        if i < len(pool) and i not in seen:
            seen.add(i)
            picked.append(pool[i])
        if len(picked) >= top_k:
            break
    return picked if picked else pool[:top_k]  # 파싱 실패/NONE이면 거리 순 폴백(안전장치)


def search_v2(
    query: str,
    state: dict,
    top_k: int = 3,
    where: dict | None = None,
    candidates_per_subquery: int = 40,
) -> list[dict]:
    """decomposition -> 서브 질문별 벡터 검색 -> 중복 제거(실제 청크 id 기준) -> 서브 질문 기준 리랭커로 최종 top_k.

    중복 제거 키로 예전엔 product_model+section_title+page를 조합해서 썼는데,
    긴 섹션이 여러 조각(piece)으로 잘린 경우 조각들이 전부 같은 키로 뭉개져서
    서로 다른 내용인데 하나만 살아남는 문제가 있었다 (Chroma의 실제 청크 id는
    product_model_섹션번호_조각번호라 조각 단위까지 구분됨 - 실제로 확인해보니
    5,847개 청크 중 1,296개가 예전 방식으로는 id가 겹쳤음). 실제 청크 id를 그대로
    쓰면 "같은 조각이 여러 서브 질문에서 다시 걸린 경우"만 정확히 병합되고, 서로
    다른 조각은 안 뭉개진다.

    candidates_per_subquery(리랭크 전 1차 벡터검색 top_k, 최종 답변용 top_k=3과는
    다름)는 5였다가 40으로 올렸다 - openai_round1_v3_pagefallback.json 실제 답변
    감사 중 "냉장고 온도 설정 방법" 계열 질문에서 정답 청크(제어창 섹션)가 코사인
    유사도 기준 모델 내에서 12~35등까지 나오는 걸 확인함(모델마다 편차 큼).
    model_verification.py 300문항(모델당 5개, 실제 section 기반)으로 5->20 측정
    시 히트율 95.7%->99.0%였지만, 실제 MODEL_QUESTIONS 라운드1 템플릿 질문으로는
    20으로도 못 잡는 사례가 있어 40으로 추가 상향.

    아울러 서브 질문은 _strip_known_models()로 모델 코드를 뺀 텍스트를 벡터검색/
    리랭크 둘 다에 쓴다 - 모델 스코프는 이미 where 필터로 처리되는데 텍스트에
    모델 코드가 또 있으면 노이즈로 작용해 리랭커 점수가 거의 0으로 뭉개지는 걸
    확인함(아래 _strip_known_models 문서 참고).
    """
    sub_queries = decompose_query(query)
    clean_sub_queries = [_strip_known_models(sq, state) for sq in sub_queries]

    # 질문 의도(content_type)가 감지되면 그 유형으로 좁힌 추가 검색 경로도 병행
    # (#22) - 원본 질문 기준 1회만 감지(서브 질문 루프 밖).
    content_type = detect_content_type(query)
    typed_where = _merge_where(where, {"content_type": content_type}) if content_type else None

    merged: dict[str, dict] = {}
    for sub_q in clean_sub_queries:
        found = _vector_search(sub_q, state, top_k=candidates_per_subquery, where=where)
        if typed_where:
            found += _vector_search(sub_q, state, top_k=candidates_per_subquery, where=typed_where)
        for c in found:
            chunk_id = c["id"]
            if chunk_id not in merged or c["distance"] < merged[chunk_id]["distance"]:
                merged[chunk_id] = c

    return _llm_rerank(clean_sub_queries, list(merged.values()), state, top_k=top_k)


# ════════════════════════════════════════════════════════════════════════════
# 카테고리/브랜드 필터 감지 — 질문에 이미 있는 정보를 활용해 엉뚱한 기기 문서가
# 섞여 들어오는 걸 막는다 (예: "에어컨에서 CH04 에러" 질문에 세탁기/건조기 청크가
# 딸려온 걸 실제로 확인했음 - distance가 나쁜데도 top_k를 채우려고 억지로 포함됨).
# ════════════════════════════════════════════════════════════════════════════

# 실제 Chroma 메타데이터의 category는 폴더 기준 3종(에어컨/냉장고/세탁기)뿐이다.
# MODEL_QUESTIONS의 세분화된 카테고리(김치냉장고/건조기/세탁건조기/스타일러/
# 슈드레서)는 이 3종 중 하나로 매핑한다 - 이걸 빠뜨리면 해당 카테고리 질문이
# where 필터 없이 전체 코퍼스를 검색해서 다른 브랜드/기기 내용이 섞여 들어온다
# (실제로 확인: "스타일러 문이 안 열려요" 질문에 완전히 다른 모델의 표지
# 페이지가 근거로 잡힌 사례 발견).
CATEGORY_KEYWORDS = {
    "에어컨": "에어컨",
    "냉장고": "냉장고",
    "김치냉장고": "냉장고",
    "세탁기": "세탁기",
    "건조기": "세탁기",
    "세탁건조기": "세탁기",
    "스타일러": "세탁기",
    "슈드레서": "세탁기",
}
BRAND_KEYWORDS = {"LG": "LG", "엘지": "LG", "삼성": "삼성"}


def detect_where_filter(query: str, state: dict | None = None) -> dict | None:
    """질문에 카테고리/브랜드/모델명이 명시돼 있으면 그걸로 Chroma where 필터를 만든다.
    아무것도 없으면 None(필터 없이 전체 검색) - 브랜드/카테고리 미언급 질문도
    COMMON_QUESTIONS에 실제로 있어서, 그런 경우까지 강제로 좁히면 안 된다.

    모델명까지 감지하는 이유: 카테고리만으로는 같은 계열(예: "세탁기" 카테고리엔
    드럼세탁기/건조기/스타일러/슈드레서가 다 섞여 있음) 안에서 다른 모델의
    콘텐츠가 더 가깝게 나오는 경우가 실제로 있었다(스타일러 질문에 엉뚱한
    모델 표지 페이지가 근거로 잡힌 사례) - 질문에 정확한 모델명이 있으면
    (MODEL_QUESTIONS는 항상 그럼) 그걸로 확실하게 좁힌다."""
    clauses = []
    for kw, cat in CATEGORY_KEYWORDS.items():
        if kw in query:
            clauses.append({"category": cat})
            break
    for kw, brand in BRAND_KEYWORDS.items():
        if kw in query:
            clauses.append({"brand": brand})
            break
    if state is not None:
        for model in state.get("known_models", []):
            if model in query:
                clauses.append({"product_model": model})
                break
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ════════════════════════════════════════════════════════════════════════════
# 일반적 "에러/고장 확인 방법" 질문 → 구조적 트러블슈팅 챕터 직접 매칭 (버그 #18)
#
# "에어컨 에러 메시지 확인하는 방법 알려줘"류 질문은 특정 코드(UE, CH04 등)를
# 묻는 게 아니라 "어디서/어떻게 확인하는지" 위치를 묻는다. RDB(find_error_codes_
# in_query)는 구체적 코드만 정확매칭하지 이런 일반 질문은 못 잡는다. 실제로
# round1 LLM판정에서 확인: 벡터검색은 "고장신고 전 확인하기" 관련 내용을
# 제대로 찾아오는데도, "에러"라는 단어가 문서에 문자 그대로 없어서(참고3)
# 생성 단계가 "문서에 없다"고 과도하게 보수적으로 거절함(FALSE_DECLINE).
#
# 그래서 이런 질문은 임베딩 유사도에 기대지 않고, 청킹 때 이미 식별해둔
# "고장신고 전 확인하기"/"서비스를 요청하기 전에" 같은 모델별 대표 트러블슈팅
# 챕터 청크를 구조적으로 바로 찾아서 최우선 후보로 넣는다. 그 챕터 안에서
# 여러 조각이 있으면 리랭커로 그 중 가장 관련 높은 것만 추린다(챕터 확정은
# 구조적으로, 챕터 내 세부 선택은 의미 기반으로 - 두 단계 혼합).
# ════════════════════════════════════════════════════════════════════════════

_GENERIC_TROUBLESHOOTING_PATTERN = re.compile(
    r"(에러|오류|고장|이상)\s*(메시지|코드|증상)?\s*.{0,6}(확인|어디)"
)

# round1 LLM판정 FALSE_CONFIDENCE 13건 재분석(2026-09-16) - 절반 이상(7/13)이
# "이상한 소리가 나요"/"성에가 계속 생겨요"/"냄새가 나요"류 **증상 보고형** 질문
# 이었다. 이런 질문도 결국 트러블슈팅 챕터가 정답인데, 위 패턴("확인"/"어디"가
# 붙어있는 형태)엔 안 걸려서 놓치고 있었다. 증상을 보고하는 종결어미 패턴을
# 추가해서 같은 트러블슈팅 챕터 매칭을 태운다.
_SYMPTOM_REPORT_PATTERN = re.compile(
    r"(소리|소음|냄새|성에|진동|누수|물)\s*.{0,10}(나요|나는데|그래요|생겨요)|"
    r"(안\s*(돼요|되요|열려요|닫혀요|켜져요|나와요))"
)


def is_generic_troubleshooting_question(query: str) -> bool:
    """구체적 에러 코드(UE, CH04 등)가 아니라 (1) 일반적인 "확인 방법/위치"를
    묻거나 (2) 특정 증상을 보고하는 질문인지 판단한다 - 둘 다 결국 모델별
    트러블슈팅 챕터가 정답이다. 구체적 코드는 find_error_codes_in_query가 이미
    RDB로 정확매칭하니 여기서 다시 다뤄도 무해하다(중복 매칭 OK)."""
    return bool(
        _GENERIC_TROUBLESHOOTING_PATTERN.search(query)
        or _SYMPTOM_REPORT_PATTERN.search(query)
    )


def _structural_chapter_candidates(chunk_ids: list[str], query: str, state: dict, top_k: int) -> list[dict]:
    """구조적으로 확정된 챕터 청크 중, 질문과 가장 관련 높은 것만 LLM으로 추린다.
    (챕터 확정은 구조적으로, 챕터 내 세부는 LLM 판단으로 - 두 단계 혼합)

    원래 cross-encoder를 썼는데, 후보가 단 3개뿐인 SQ07GA3WBN 사례에서도 진짜
    정답("LG ThinQ로 고장 진단하기")보다 무관한 청소 안전수칙 문단에 더 높은
    점수를 준 걸 확인해서(#20) - 후보 수를 줄여도 pointwise 방식 자체의 한계는
    그대로라 _llm_rerank로 통일함."""
    if not chunk_ids:
        return []
    res = state["collection"].get(ids=chunk_ids, include=["documents", "metadatas"])
    candidates = [
        {"id": cid, "text": doc, "metadata": meta, "distance": 0.3}  # 구조적으로 확정된 위치라 낮은 거리 부여
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    return _llm_rerank([query], candidates, state, top_k)


def find_troubleshooting_chapter(query: str, model: str, state: dict, top_k: int = 3) -> list[dict]:
    """모델의 대표 트러블슈팅 챕터 청크 중, 질문과 가장 관련 높은 것만 리랭커로 추린다."""
    chunk_ids = state.get("troubleshooting_chunks", {}).get(model, [])
    return _structural_chapter_candidates(chunk_ids, query, state, top_k)


_PANEL_QUESTION_PATTERN = re.compile(r"(온도|기능|모드)\s*.{0,6}(설정|조절|선택)\s*.{0,6}(방법|어떻게)")


def is_panel_setting_question(query: str) -> bool:
    """냉장고/김치냉장고류 "온도 설정 방법" 같은 조작판 사용법 질문인지 판단.
    round1 LLM판정 FALSE_CONFIDENCE 중 절차 안내(증상 보고 아님) 계열이 이 패턴."""
    return bool(_PANEL_QUESTION_PATTERN.search(query))


def find_panel_chapter(query: str, model: str, state: dict, top_k: int = 3) -> list[dict]:
    """모델의 대표 "제어판/버튼과 표시부" 챕터 청크 중 관련 높은 것만 추린다."""
    chunk_ids = state.get("panel_chunks", {}).get(model, [])
    return _structural_chapter_candidates(chunk_ids, query, state, top_k)


_SPEC_QUESTION_PATTERN = re.compile(
    r"(전력|소비|효율|등급|사용량|무게|용량|크기|사이즈)\s*.{0,10}(궁금|어떻게|얼마)|"
    r"얼마나\s*넣"
)


def is_spec_question(query: str) -> bool:
    """전력 소비량/용량/에너지 등급 등 스펙 조회형 질문인지 판단.
    이런 질문의 정답은 "제품규격" 표에 있는데, 표 형태(숫자·기호 밀집)라
    bi-encoder 임베딩이 약하게 걸리는 걸 실측 확인함(#21).

    `find_spec_chapter`(하드코딩 타이틀 매칭)는 2026-09-17 제거됨 - 태깅
    (detect_content_type -> SPEC)만으로 동일 커버리지가 실측 확인돼서
    이 감지 함수는 이제 detect_content_type()에서만 쓰인다."""
    return bool(_SPEC_QUESTION_PATTERN.search(query))


# ════════════════════════════════════════════════════════════════════════════
# 의미 유형(content_type) 태깅 기반 검색 보강 (#22)
#
# #18/#19/#21은 "실패 사례를 보고 그 타이틀을 손으로 찾아 규칙을 추가"하는
# 방식이었다 - round1엔 잘 맞았지만(93.7%) round2/3의 새 질문 유형("OO 기능이
# 뭐야"류 기능설명형, round2 6모델 전부 FALSE_DECLINE)엔 전혀 안 맞아서
# 72~76%에 그쳤다(실측 확인). 타이틀을 계속 추가하는 대신, 청크마다 미리
# 의미 유형(PROCEDURE/TROUBLESHOOTING/FEATURE_EXPLANATION/SPEC/GENERAL)을
# gpt-4o-mini로 소급 태깅해두고(tag_content_type.py), 질문 의도와 매칭되는
# 유형으로 where 필터를 넓혀서 검색을 보강한다 - 새 카테고리/새 모델이 추가돼도
# 타이틀 재조사 없이 자동으로 작동하는 게 핵심 차이.
#
# 기존 검색 경로를 "대체"하지 않고 "추가"만 한다(FAQ와 같은 패턴) - 태깅이
# 가끔 틀려도(애매한 GENERAL/PROCEDURE 경계 등) 회귀 위험이 없다.
# ════════════════════════════════════════════════════════════════════════════

_FEATURE_QUESTION_PATTERN = re.compile(
    r"(기능|모드|코스)\s*.{0,4}(이|가|은|는)?\s*(뭐야|뭐예요|무엇|알려줘|어떻게 써|어떻게 되나요)"
)


def is_feature_explanation_question(query: str) -> bool:
    """"OO 기능이 뭐야"/"OO 모드는 어떻게 써"류 기능 설명형 질문인지 판단.
    round2에서 "무풍 기능이 뭐야" 템플릿이 6개 모델 전부 FALSE_DECLINE이었던
    사례로 실측 확인된, #18/#19/#21이 커버 못 하던 새로운 실패 유형."""
    return bool(_FEATURE_QUESTION_PATTERN.search(query))


def detect_content_type(query: str) -> str | None:
    """질문 의도에 맞는 content_type 태그를 반환한다(없으면 None).
    기존 구조적 매칭 감지 함수들을 재사용해서 일관성을 유지한다."""
    if is_generic_troubleshooting_question(query):
        return "TROUBLESHOOTING"
    if is_panel_setting_question(query):
        return "PROCEDURE"
    if is_spec_question(query):
        return "SPEC"
    if is_feature_explanation_question(query):
        return "FEATURE_EXPLANATION"
    return None


def _merge_where(where: dict | None, extra: dict) -> dict:
    """기존 where 필터에 조건을 하나 더 AND로 추가한다."""
    if where is None:
        return extra
    if "$and" in where:
        return {"$and": where["$and"] + [extra]}
    return {"$and": [where, extra]}


# ════════════════════════════════════════════════════════════════════════════
# 응답 보류(withholding) 판단 — 거리 임계값 + LLM 관련성 검증 조합
#
# 왜 필요한가: 실제로 확인해보니 벡터검색은 관련 내용이 없어도 "그나마 가장
# 가까운 것"을 항상 반환한다. LLM의 자체 판단("문서에 없습니다")에만 맡기면
# 불안정하다 - 거짓 확신(무관한 내용으로 자신 있게 답변)도, 거짓 겸손(진짜
# 정답이 있는데도 "없다"고 함)도 실제로 관찰됐다(오늘 감사 참고).
#
# 그래서 임계값 2단계 + 애매한 구간만 LLM 검증하는 절차를 쓴다:
#   - distance <= DISTANCE_CONFIDENT: 확실히 관련 - LLM 검증 생략(비용 절약)
#   - distance >  DISTANCE_REJECT   : 확실히 무관 - LLM 호출 없이 바로 보류
#   - 그 사이(애매한 구간)          : LLM한테 "이 내용이 질문에 답할 수 있어?"
#     한 번 더 물어봐서 최종 결정
#
# 임계값은 오늘 감사에서 관찰된 실측치 기반 - 진짜 정답 매치는 대체로
# 0.4~0.7대, 무관한 내용은 1.1~1.3대였다. 엄밀한 정밀도/재현율 튜닝은
# 아니고 1차 추정치라, 더 많은 데이터가 쌓이면 재조정이 필요하다.
# ════════════════════════════════════════════════════════════════════════════

DISTANCE_CONFIDENT = 0.85
DISTANCE_REJECT = 1.25

NO_ANSWER_MESSAGE = (
    "죄송합니다. 제공된 문서에서 이 질문과 관련된 내용을 찾지 못했습니다. "
    "사용설명서를 직접 확인하시거나 제조사 서비스센터에 문의해 주세요."
)

RELEVANCE_CHECK_PROMPT = (
    "아래 [문서]가 [질문]에 실제로 답할 수 있는 내용을 담고 있는지 판단해. "
    "문서 내용이 질문과 명백히 무관하면(예: 질문은 특정 기능 사용법인데 문서는 "
    "안전 경고나 전혀 다른 증상 얘기) '아니오', 답할 수 있으면 '예'라고만 답해. "
    "다른 설명 없이 '예' 또는 '아니오'만 출력해."
)


def _is_relevant_enough(query: str, context: str, min_distance: float) -> bool:
    if min_distance <= DISTANCE_CONFIDENT:
        return True
    if min_distance > DISTANCE_REJECT:
        return False
    verdict = generate_openai(
        RELEVANCE_CHECK_PROMPT, f"[질문]\n{query}\n\n[문서]\n{context}",
        temperature=0.0,
    )
    return verdict.strip().startswith("예")


# ════════════════════════════════════════════════════════════════════════════
# 페이지 출처 표기 — metadata의 page를 답변 근거로 노출
# ════════════════════════════════════════════════════════════════════════════

def _format_context_with_pages(candidates: list[dict]) -> str:
    """청크마다 페이지 번호를 앞에 붙여서 LLM 컨텍스트를 만든다.

    SYSTEM_PROMPT는 수정 금지라 LLM한테 "페이지를 인용해서 답해"라고 직접
    지시할 수 없다 - 그래서 출처 표기는 LLM 답변에 맡기지 않고, 아래
    _page_citation()으로 답변 뒤에 프로그램이 직접 붙인다(결정적이고 안전함)."""
    parts = []
    for c in candidates:
        page = c.get("metadata", {}).get("page")
        prefix = f"[{page}페이지] " if page is not None else ""
        parts.append(f"{prefix}{c['text']}")
    return "\n\n".join(parts)


def _page_citation(candidates: list[dict]) -> str:
    pages = sorted({
        c["metadata"]["page"] for c in candidates
        if c.get("metadata", {}).get("page") is not None
    })
    if not pages:
        return ""
    return "\n\n(참고: " + ", ".join(f"{p}페이지" for p in pages) + ")"


# ════════════════════════════════════════════════════════════════════════════
# ★ my_answer() — RDB + 벡터 검색(decomposition+리랭커) 결합
# ════════════════════════════════════════════════════════════════════════════

def gather_candidates(query: str, state: dict) -> list[dict]:
    """RDB 정확매칭 + 구조적 챕터 매칭(트러블슈팅/패널/스펙, #18/#19/#21) +
    search_v2(원본+FAQ 검색, listwise LLM 리랭킹)를 전부 병합한 최종 후보 목록.

    my_answer()와 검증 스크립트(quant_llm_rerank.py 등)가 이 함수를 공유해야
    실제 프로덕션 로직과 동일한 걸 측정한다 - 한때 검증 스크립트가 search_v2()만
    호출해서 구조적 매칭 효과가 전혀 반영 안 된 숫자를 보고했던 적이 있었음."""
    candidates: list[dict] = []

    for code in find_error_codes_in_query(query, state):
        for r in lookup_code(state["conn"], code):
            candidates.append({
                "id": f"rdb_{r['solution_id']}",
                "text": f"[{r['brand']}/{r['category']}] {r['title']}\n{r['content']}",
                "distance": 0.0,
                "metadata": {},
            })

    model = next((m for m in state.get("known_models", []) if m in query), None)
    if model:
        if is_generic_troubleshooting_question(query):
            for c in find_troubleshooting_chapter(query, model, state):
                candidates.append({
                    "id": c["id"], "text": c["text"], "distance": c["distance"],
                    "metadata": c.get("metadata", {}),
                })
        if is_panel_setting_question(query):
            for c in find_panel_chapter(query, model, state):
                candidates.append({
                    "id": c["id"], "text": c["text"], "distance": c["distance"],
                    "metadata": c.get("metadata", {}),
                })
        # 스펙(#21) 하드코딩 타이틀 매칭은 제거됨 - search_v2()의
        # content_type=SPEC 검색 경로(아래 search_v2 호출 안에서 자동 적용)가
        # 대신 처리한다.

    seen_ids = {c["id"] for c in candidates}
    where_filter = detect_where_filter(query, state)
    for c in search_v2(query, state, top_k=3, where=where_filter):
        if c["id"] in seen_ids:
            continue
        seen_ids.add(c["id"])
        candidates.append({
            "id": c["id"],
            "text": c["text"],
            "distance": c["distance"],
            "metadata": c.get("metadata", {}),
        })
    return candidates


def my_answer(query: str) -> dict:
    state = _load_state()
    candidates = gather_candidates(query, state)

    def _public(cands: list[dict]) -> list[dict]:
        """반환 계약({"id","text","distance"})에 맞춰 내부용 metadata는 뺀다."""
        return [{"id": c["id"], "text": c["text"], "distance": c["distance"]} for c in cands]

    if not candidates:
        return {"answer": NO_ANSWER_MESSAGE, "candidates": []}

    context = _format_context_with_pages(candidates)
    min_distance = min(c["distance"] for c in candidates)

    if not _is_relevant_enough(query, context, min_distance):
        return {"answer": NO_ANSWER_MESSAGE, "candidates": _public(candidates)}

    answer = generate_openai(SYSTEM_PROMPT, f"[문서]\n{context}\n\n[질문]\n{query}")
    answer += _page_citation(candidates)

    return {"answer": answer, "candidates": _public(candidates)}


if __name__ == "__main__":
    run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)
