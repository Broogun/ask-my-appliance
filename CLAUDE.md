# Claude 지침서 — ask-my-appliance RAG 프로젝트

AI(Claude 등)에게 작업을 맡길 때 이 파일을 먼저 읽혀라.
프로젝트 구조, 네이밍 규칙, 금지 사항을 담고 있다.

---

## 프로젝트 개요

**한국 가전제품 사용설명서 RAG 시스템.**
LG/삼성 에어컨·냉장고·세탁기 PDF 매뉴얼(각 30개)을 청킹·임베딩해서
사용자 질문에 근거 문서 기반으로 답변한다.

- 임베딩: `BAAI/bge-m3` (로컬, 4.3GB)
- 벡터DB: ChromaDB (`chroma_db/`, 로컬 전용)
- LLM: OpenAI `o4-mini` (기본값) 또는 Ollama 로컬
- 에러코드 DB: SQLite (`data/error_codes.db`)

---

## 디렉토리 구조

```
pipeline/           ← 공통 핵심 코드 (팀원이 직접 수정 금지)
  pdf_chunker.py    ← PDF → 청크 (TOC 계층 기반)
  embed_store.py    ← 임베딩 + ChromaDB 저장/조회
  query_engine.py   ← 검색 + LLM 호출 + 라우팅 (RDB/벡터/복합)
  evaluate.py       ← 공통 평가 유틸리티 (COMMON_QUESTIONS 포함)
  build_documents.py
  figure_extractor.py

experiments/        ← 팀원 개인 실험 (여기서만 자유롭게)
  template_exp.py   ← 복사해서 시작
  RULES.md          ← 실험 규칙, 브랜치 전략
  results/          ← 결과 JSON (gitignore)

db/
  seed_error_codes.py   ← 에러코드 DB 초기화 스크립트

data/
  lg/               ← LG PDF (gitignore, 공유 드라이브에서 받기)
  samsung/          ← 삼성 PDF (gitignore)
  manual_figures/   ← 그림 추출 결과 (gitignore)
  error_codes.db    ← SQLite (gitignore, seed 스크립트로 재생성)

config.py           ← .env 로딩, 전역 설정 변수
app_streamlit.py    ← Streamlit 챗봇 UI
```

---

## 변수명 규칙 (반드시 준수)

| 역할 | 변수명 |
|---|---|
| 사용자 질문 | `query` |
| 검색 결과 목록 | `candidates` |
| 검색 결과 개수 | `top_k` |
| 제품 모델 필터 | `model_filter` |
| 청크 고유 ID | `chunk_id` |
| 섹션 제목 | `section_title` |
| 유사도 거리 | `distance` (낮을수록 좋음) |
| 임베딩 벡터 | `embedding` |

---

## answer() 반환 형식

`pipeline/query_engine.answer()` 및 실험 함수는 반드시 이 구조를 반환한다.

```python
{
    "answer": str,              # LLM이 생성한 최종 답변
    "candidates": [             # 검색된 근거 청크 목록
        {
            "id": str,
            "text": str,
            "distance": float,
            "metadata": {
                "product_model": str,
                "section_title": str,
                "source": str,
            }
        }
    ],
    "source": str,              # "vector" | "rdb" | "combined"
}
```

---

## 라우팅 로직

질문이 들어오면 `answer()`가 아래 순서로 처리한다.

```
질문
 ├─ 에러코드 감지(UE, F4 등) → RDB 조회
 │    └─ 벡터 검색도 병행 → 결과 있으면 "combined" 반환
 └─ 에러코드 없음 → 벡터 검색
       ├─ 모델명 감지 → 해당 모델 문서만 필터링
       └─ 모델명 없음 → 전체 컬렉션 검색
```

---

## 금지 사항

- `pipeline/` 파일을 실험 코드에서 직접 수정하지 않는다.
  → 기능을 바꾸고 싶으면 실험 파일 안에서 함수를 override한다.
- `.env` 파일을 커밋하지 않는다. API 키가 포함돼 있다.
- `chroma_db/`, `data/lg/`, `data/samsung/`을 커밋하지 않는다. 용량이 크다.
- 실험 결과 JSON(`experiments/results/`)을 커밋하지 않는다.

---

## 실험 시작 방법

```bash
# 1. 브랜치 생성
git checkout -b exp/{본인이름}

# 2. 템플릿 복사
cp experiments/template_exp.py experiments/{이름}_exp01.py

# 3. my_answer() 구현 후 실행
python experiments/{이름}_exp01.py

# 4. 결과 확인 후 PR
```

---

## 주의사항

- 로컬 임베딩(bge-m3) 기준 LG 30개 ingest는 약 1시간 소요.
  OpenAI 임베딩 사용 시 `.env`에서 `EMBEDDING_BACKEND=openai`로 변경.
- ChromaDB 컬렉션명: `appliance_manuals_local`
- 에러코드 DB가 없으면 `python db/seed_error_codes.py` 실행.
- Streamlit 실행: `streamlit run app_streamlit.py` → http://localhost:8501
