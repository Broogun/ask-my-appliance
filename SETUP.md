# 로컬 환경 세팅 가이드

## 1. 패키지 설치

```bash
pip install -r requirements.txt
```

## 2. .env 설정

`.env.example`을 복사해서 `.env`를 만들고 값을 채운다.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|---|---|
| `EMBEDDING_BACKEND` | `local` (무료) 또는 `openai` (유료) |
| `CHAT_BACKEND` | `local` / `ollama` / `openai` |
| `OPENAI_API_KEY` | OpenAI 사용 시 팀장에게 문의 |
| `CHAT_MODEL` | `o4-mini` 권장 |

## 3. PDF 매뉴얼 받기

Git에는 PDF가 포함되어 있지 않다. 팀 공유 드라이브에서 받아서 아래 경로에 넣는다.

```
data/
  lg/
    aircon/   ← LG 에어컨 PDF 10개
    fridge/   ← LG 냉장고 PDF 10개
    washer/   ← LG 세탁기 PDF 10개
  samsung/
    aircon/   ← 삼성 에어컨 PDF 10개
    fridge/   ← 삼성 냉장고 PDF 10개
    washer/   ← 삼성 세탁기 PDF 10개
```

## 4. 에러코드 DB 생성

```bash
python db/seed_error_codes.py
```

## 5. 벡터DB 생성 (ingest)

LG 데이터:
```bash
python scripts_batch_ingest_all_lg.py
```

삼성 데이터 (추후):
```bash
python scripts_batch_ingest_all_samsung.py
```

> ⚠️ 로컬 임베딩(bge-m3) 기준 LG 30개 기준 약 1시간 소요.
> OpenAI 임베딩 사용 시 1~2분으로 단축.

## 6. 앱 실행

```bash
streamlit run app_streamlit.py
```

→ http://localhost:8501

## 브랜치 규칙

| 브랜치 | 용도 |
|---|---|
| `main` | 검증된 코드만 (팀장 머지) |
| `dev` | 통합 개발 |
| `feat/samsung-data` | 삼성 데이터 담당자 |
| `feat/service-ui` | 서비스 UI 담당자 |
| `feat/rag-experiment` | RAG 실험 담당자 |

- PR은 `dev` 브랜치로 올리기
- `main` 머지는 팀장이 최종 확인 후 진행
