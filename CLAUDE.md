# Claude 지침서 — ask-my-appliance RAG 프로젝트

AI(Claude 등)에게 작업을 맡길 때 이 파일을 먼저 읽혀라.

---

## 프로젝트 개요

**한국 가전제품 사용설명서 RAG 실험 프로젝트.**  
LG·삼성 에어컨·냉장고·세탁기 PDF 매뉴얼을 데이터로 사용해  
팀원 4명이 각자 독립적인 RAG 파이프라인을 구현하고 성능을 비교한다.

---

## 이 레포에 있는 것

```
experiments/
  template_exp.py       ← 실험 시작 템플릿 (복사해서 사용)
  rag_tutorial.ipynb    ← 환경 세팅 확인 노트북
  planning_template.md  ← 사전 기획서 템플릿
  RULES.md              ← 실험 규칙 및 브랜치 전략
  compare_results.py    ← 팀 전체 결과 비교 보고서 생성
  results/              ← 실험 결과 JSON (커밋 대상)

.github/workflows/
  eval_report.yml       ← PR push 시 자동 보고서 생성 Action

README.md / CLAUDE.md / SETUP.md
.env.example / .gitignore / requirements.txt
```

---

## 실험 구조

### 통일 (수정 금지)

| 항목 | 내용 |
|---|---|
| 데이터 | `data/lg/` + `data/samsung/` LG·삼성 PDF |
| LLM | gpt-4o-mini |
| SYSTEM_PROMPT | 한국어, 근거 문서 기반 답변 |
| COMMON_QUESTIONS | 7개 고정 질문 |
| 반환 형식 | `{"answer": str, "candidates": list[dict]}` |

### 자유 (각자 결정)

청킹 방식 / 임베딩 모델 / 벡터 DB 구성 방식 / 검색 전략

---

## my_answer() 반환 형식

실험 함수는 반드시 이 구조를 반환해야 한다.

```python
{
    "answer": str,           # LLM이 생성한 최종 답변
    "candidates": [          # 검색된 근거 청크 목록
        {
            "id":       str,
            "text":     str,
            "distance": float,   # 낮을수록 유사 (없으면 0.0)
        },
    ],
}
```

---

## STRATEGY 딕셔너리

보고서에 자동 포함된다. 반드시 작성.

```python
STRATEGY = {
    "chunking":         "어떤 방식",
    "chunking_reason":  "왜 선택",
    "embedding":        "어떤 모델",
    "embedding_reason": "왜 선택",
    "db":               "어떤 DB",
    "db_reason":        "왜 선택",
    "retrieval":        "검색 전략",
    "retrieval_reason": "왜 선택",
}
```

---

## 변수명 규칙

| 역할 | 변수명 |
|---|---|
| 사용자 질문 | `query` |
| 검색 결과 목록 | `candidates` |
| 검색 결과 개수 | `top_k` |
| 청크 고유 ID | `chunk_id` |
| 유사도 거리 | `distance` (낮을수록 좋음) |
| 임베딩 벡터 | `embedding` |

---

## 금지 사항

- `COMMON_QUESTIONS`, `SYSTEM_PROMPT` 수정 금지 (팀 비교 기준이 무너짐)
- `.env` 커밋 금지 (API 키 포함)
- `data/` 커밋 금지 (용량 큼, 공유 드라이브에서 배포)

---

## 실험 시작 방법

```bash
# 1. 브랜치 생성
git checkout -b exp/{본인이름}

# 2. 템플릿 복사
cp experiments/template_exp.py experiments/{이름}_exp01.py

# 3. STRATEGY 채우고 my_answer() 구현

# 4. 실험 실행
python experiments/{이름}_exp01.py

# 5. 코드 + 결과 함께 커밋 후 PR
git add experiments/{이름}_exp01.py experiments/results/{이름}_exp01.json
git commit -m "[{이름}] exp01: 한 줄 설명"
git push origin exp/{본인이름}
```

코드 파일(`{이름}_exp01.py`)도 결과 JSON과 함께 커밋한다.  
팀장이 코드를 리뷰하고 전략과 결과가 일치하는지 확인한다.

---

## 보고서 자동 생성

`exp/*` 브랜치에 `results/*.json`이 push되면  
GitHub Actions가 PR 댓글에 비교 보고서를 자동으로 달아준다.

보고서 섹션:
1. 전략 비교 (청킹/임베딩/DB 선택 이유)
2. 성능 요약 (히트율/거리/응답시간)
3. 질문별 상세 비교
4. 결론 (최우수 실험)
