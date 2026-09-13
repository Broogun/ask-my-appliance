# Ask My Appliance

LG·삼성 가전제품 사용설명서 기반 RAG 챗봇.
에어컨·냉장고·세탁기 매뉴얼 PDF를 벡터 검색해서 사용법·고장 원인·에러코드를 답변한다.

---

## 기술 스택

| 역할 | 사용 기술 |
|---|---|
| PDF 파싱·청킹 | PyMuPDF (TOC 계층 기반) |
| 임베딩 | `BAAI/bge-m3` (로컬) / `text-embedding-3-small` (OpenAI) |
| 벡터 DB | ChromaDB |
| LLM | OpenAI `o4-mini` (기본값) / Ollama / 로컬 Qwen |
| 에러코드 DB | SQLite |
| UI | Streamlit |

---

## 디렉토리 구조

```
pipeline/               ← 공통 RAG 파이프라인 (수정 금지)
  pdf_chunker.py        ← PDF → 청크
  embed_store.py        ← 임베딩 + ChromaDB
  query_engine.py       ← 검색 + LLM 답변 생성
  evaluate.py           ← 공통 평가 유틸리티

experiments/            ← 팀원 개인 실험 공간
  template_exp.py       ← 실험 시작 템플릿 (복사해서 사용)
  RULES.md              ← 실험 규칙 및 브랜치 전략

db/                     ← 에러코드 SQLite
data/                   ← 에러코드 JSON (PDF는 gitignore)
docs/                   ← 설계 문서
```

---

## 빠른 시작

### 1. 클론 및 패키지 설치

```bash
git clone https://github.com/Broogun/ask-my-appliance.git
cd ask-my-appliance
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어서 아래 값을 채운다.

| 변수 | 설명 |
|---|---|
| `EMBEDDING_BACKEND` | `local` (기본, 무료) 또는 `openai` (빠름, 유료) |
| `CHAT_BACKEND` | `local` / `ollama` / `openai` |
| `OPENAI_API_KEY` | OpenAI 사용 시 팀장에게 문의 |

> 기본값(`local`)은 OpenAI 키 없이도 동작한다. 대신 bge-m3 모델 4.3GB 자동 다운로드.

### 3. PDF 매뉴얼 받기

PDF는 용량 문제로 Git에 포함되어 있지 않다.
팀 공유 드라이브에서 받아서 아래 경로에 넣는다.

```
data/
  lg/
    aircon/    ← LG 에어컨 PDF
    fridge/    ← LG 냉장고 PDF
    washer/    ← LG 세탁기 PDF
  samsung/
    aircon/    ← 삼성 에어컨 PDF
    fridge/    ← 삼성 냉장고 PDF
    washer/    ← 삼성 세탁기 PDF
```

### 4. DB 초기화 및 ingest

```bash
# 에러코드 DB 생성
python db/seed_error_codes.py

# 벡터DB 생성 (로컬 임베딩 기준 약 1시간)
python scripts_batch_ingest_all_lg.py
```

### 5. 챗봇 실행

```bash
streamlit run app_streamlit.py
```

→ http://localhost:8501

---

## 실험 참여 방법

각자 RAG 전 과정을 직접 구현하고 공통 지표로 비교한다.

```bash
# 1. 본인 브랜치 생성
git checkout -b exp/{이름}         # 예: exp/minsoo

# 2. 템플릿 복사
cp experiments/template_exp.py experiments/{이름}_exp01.py

# 3. my_answer() 함수 안에 본인 RAG 구현

# 4. 실험 실행 (공통 5개 질문 자동 평가)
python experiments/{이름}_exp01.py

# 5. PR 생성 → 팀장 리뷰
```

자세한 규칙은 [`experiments/RULES.md`](experiments/RULES.md) 참고.

---

## 브랜치 전략

| 브랜치 | 용도 |
|---|---|
| `main` | 검증된 코드만 (팀장 머지) |
| `exp/{이름}` | 개인 실험 브랜치 |

- 본인 브랜치에서만 커밋한다
- 실험 공유 시 PR → `main`
- PR 제목 형식: `[이름] exp{번호}: 한 줄 설명`

---

## AI 사용 가이드

Claude 등 AI 어시스턴트에게 작업을 맡길 때는 [`CLAUDE.md`](CLAUDE.md)를 먼저 읽혀라.
프로젝트 구조, 변수명 규칙, 금지 사항이 정리되어 있다.
