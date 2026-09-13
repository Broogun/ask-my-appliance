# Ask My Appliance

LG·삼성 가전제품 사용설명서 기반 RAG 실험 프로젝트.  
팀원 4명이 각자 독립적인 RAG 파이프라인을 구현하고 공통 기준으로 성능을 비교한다.

---

## 실험 구조

### 통일 (모든 팀원 동일)

| 항목 | 내용 |
|---|---|
| 데이터 | LG·삼성 에어컨·냉장고·세탁기 PDF (`data/lg/` + `data/samsung/`) |
| LLM | gpt-4o-mini |
| 답변 형식 | 한국어, 근거 문서 기반 |
| 평가 질문 | 5개 고정 (`COMMON_QUESTIONS`) |

### 자유 (각자 결정)

청킹 방식 / 임베딩 모델 / 벡터 DB / 검색 전략

---

## 레포 구조

```
experiments/
  template_exp.py       ← 실험 시작 템플릿
  rag_tutorial.ipynb    ← 청킹·임베딩 방식 비교 학습 노트북
  planning_template.md  ← 사전 기획서 템플릿
  RULES.md              ← 실험 규칙 및 브랜치 전략
  compare_results.py    ← 팀 결과 비교 보고서 생성
  results/              ← 실험 결과 JSON (커밋 대상)

.github/workflows/
  eval_report.yml       ← PR push 시 자동 보고서 생성

CLAUDE.md              ← AI 어시스턴트 사용 지침
SETUP.md               ← 환경 세팅 가이드
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

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | OpenAI 사용 시 팀장에게 문의 |

### 3. PDF 받기

PDF는 용량 문제로 Git에 포함되지 않는다.  
팀 공유 드라이브에서 받아서 아래 경로에 넣는다.

```
data/
  lg/
    aircon/       ← LG 에어컨 PDF
    fridge/       ← LG 냉장고 PDF
    washer/       ← LG 세탁기 PDF
  samsung/
    aircon/       ← 삼성 에어컨 PDF
    fridge/       ← 삼성 냉장고 PDF
    washer/       ← 삼성 세탁기 PDF
```

LG·삼성 PDF를 **같은 DB** 에 넣고, `brand` 메타데이터(`"lg"` / `"samsung"`)로 필터링하는 방식을 권장한다.

---

## 실험 참여 방법

### Step 1. 노트북으로 개념 익히기

`experiments/rag_tutorial.ipynb`를 열어서 청킹·임베딩 방식별 결과를 직접 비교해본다.

### Step 2. 사전 기획서 작성

`experiments/planning_template.md`를 복사해서 본인 전략을 먼저 정리한다.

### Step 3. 실험 파일 구현

```bash
git checkout -b exp/{이름}
cp experiments/template_exp.py experiments/{이름}_exp01.py
```

`{이름}_exp01.py` 안에서 **`my_answer()` 함수 하나만 구현**한다.  
청킹·임베딩·DB·검색 방식은 전부 자유.

```python
def my_answer(query: str) -> dict:
    # 본인 RAG 전 과정 구현
    ...
    return {"answer": str, "candidates": list}
```

### Step 4. 실험 실행 및 결과 커밋

```bash
python experiments/{이름}_exp01.py
# → experiments/results/{이름}_exp01.json 자동 저장

git add experiments/{이름}_exp01.py experiments/results/{이름}_exp01.json
git commit -m "[{이름}] exp01: 한 줄 설명"
git push origin exp/{이름}
```

### Step 5. PR → 자동 보고서

PR을 올리면 GitHub Actions가 팀 전체 결과 비교 보고서를 댓글로 자동 게시한다.

---

## 브랜치 전략

| 브랜치 | 용도 |
|---|---|
| `main` | 검증된 코드만 (팀장 머지) |
| `exp/{이름}` | 개인 실험 브랜치 |

---

## AI 사용 가이드

Claude 등 AI 어시스턴트에게 작업을 맡길 때는 `CLAUDE.md`를 먼저 읽혀라.
