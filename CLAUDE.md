# Claude 지침서 — ask-my-appliance RAG 프로젝트

AI(Claude 등)에게 작업을 맡길 때 이 파일을 먼저 읽혀라.

---

## ⚠ AI 어시스턴트 git 사용 원칙 (중요)

**Claude(또는 다른 AI 어시스턴트)는 이 저장소에서 `git commit`/`git push`를
직접 실행하지 않는다.** 파일 수정·생성까지는 대신 해줘도 되지만, 커밋 메시지
확인과 실제 커밋/푸시는 **반드시 사용자 본인이 git bash에서 직접** 한다.

- AI는 변경 사항을 만들고 `git status`/`git diff`로 뭐가 바뀌었는지 보여주는
  것까지만 하고, "이제 본인이 커밋하세요"라고 안내로 마무리한다.
- 이유: 여러 팀원이 각자 AI 세션을 켜놓고 작업하는 구조라, AI가 자동으로
  커밋·푸시까지 해버리면 누가 언제 무엇을 올렸는지 사람이 놓치기 쉽고, 리뷰
  없이 `main`에 반영될 위험이 있다.
- 실험 브랜치(`exp/{이름}`)의 커밋도 예외 아님 — 실험 코드/결과 커밋도 본인이
  직접 한다.

---

## 프로젝트 개요

**한국 가전제품 사용설명서 RAG 실험 프로젝트.**  
LG·삼성 에어컨·냉장고·세탁기 PDF 매뉴얼을 데이터로 사용해  
팀원 4명이 각자 독립적인 RAG 파이프라인을 구현하고 성능을 비교한다.

**Python 3.12.14** 기준 (`.python-version`, GitHub Actions 둘 다 동일).

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

pipeline/
  evaluate.py            ← 공용 평가 하니스 (MODEL_QUESTIONS/SYSTEM_PROMPT/채점/
                            보고서 저장 - 실험마다 복붙하지 않고 import해서 씀)

.github/workflows/
  eval_report.yml       ← PR push 시 자동 보고서 생성 Action

README.md / CLAUDE.md / SETUP.md
.env.example / .gitignore / requirements.txt
```

`template_exp.py`는 이미 `from pipeline.evaluate import SYSTEM_PROMPT, run_and_save`로
이 모듈을 가져다 쓰고 있다 - 실험 파일에는 `my_answer()`만 구현하면 되고,
`MODEL_QUESTIONS`/`SYSTEM_PROMPT`/채점 로직은 건드릴 필요도, 복사할 필요도 없다.

---

## 실험 구조

### 통일 (수정 금지)

| 항목 | 내용 |
|---|---|
| 데이터 | `data/lg/` + `data/samsung/` LG·삼성 PDF 텍스트 + 기본 에러코드 데이터 |
| LLM | gpt-4o-mini |
| SYSTEM_PROMPT | 한국어, 근거 문서 기반 답변 |
| MODEL_QUESTIONS | 900개 (제품 모델 60개 x 3라운드 x 5문항, 계속 늘어남, 2026-09부터 - 이전 COMMON_QUESTIONS 22개 대체) |
| 반환 형식 | `{"answer": str, "candidates": list[dict]}` |

### 자유 (각자 결정)

청킹 방식 / 임베딩 모델 / 벡터 DB 구성 방식 / 검색 전략

### 이번 라운드 범위 밖 (제외)

이미지/그림(다이어그램) 추출과 OCR은 지금 안 한다 — 나중에 RAG 고도화 단계에서
별도로 진행할 예정. 지금은 PDF 텍스트와 에러코드 데이터만으로 실험한다.

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

- `MODEL_QUESTIONS`, `SYSTEM_PROMPT` 수정 금지 (팀 비교 기준이 무너짐 - 확장/변경은 관리자 승인 후에만)
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

## 추가 실험(exp02 이후) 진행하기

exp01 결과를 보고 개선하고 싶을 때:

```bash
# 1. 기존 브랜치에 그대로 이어서 작업 (새 브랜치 아님)
git checkout exp/{본인이름}

# 2. 번호를 올려서 새 파일로 - exp01을 덮어쓰지 않는다
cp experiments/{이름}_exp01.py experiments/{이름}_exp02.py

# 3. EXPERIMENT_NAME도 "{이름}_exp02"로 바꾸고, NOTES에 exp01과 뭘 다르게
#    했는지 적는다 (예: "청킹 크기 500→800, bge-m3로 임베딩 교체")

# 4. 실행 → 커밋 → 푸시는 exp01과 동일한 방식
python experiments/{이름}_exp02.py
git add experiments/{이름}_exp02.py experiments/results/{이름}_exp02.json
git commit -m "[{이름}] exp02: 한 줄 설명"
git push origin exp/{본인이름}
```

**exp01과 exp02, 뭐가 더 나은지 push 전에 로컬에서 먼저 확인하고 싶다면:**

```bash
python experiments/compare_results.py
```

`experiments/results/` 안의 모든 JSON을 한 번에 비교해서 터미널에 바로
출력해준다 — PR 안 열어도 로컬에서 바로 볼 수 있다.

**작업 시작 전 `main`을 한 번 당겨오는 습관도 들이면 좋다** (다른 팀원의
실험이 이미 `main`에 반영됐을 수 있어서, 같은 걸 또 시도하는 걸 피할 수 있다):

```bash
git checkout main
git pull origin main
git checkout exp/{본인이름}
```

---

## 보고서 자동 생성

`exp/*` 브랜치에 `results/*.json`이 push되면  
GitHub Actions가 PR 댓글에 비교 보고서를 자동으로 달아준다.

보고서 섹션:
1. 전략 비교 (청킹/임베딩/DB 선택 이유)
2. 성능 요약 (히트율/거리/응답시간)
3. 질문별 상세 비교
4. 결론 (최우수 실험)
