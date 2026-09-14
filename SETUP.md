# 환경 세팅 가이드

## 1. 레포 클론

```bash
git clone https://github.com/Broogun/ask-my-appliance.git
cd ask-my-appliance
```

## 2. 패키지 설치

```bash
pip install -r requirements.txt
```

## 3. 환경변수 설정

`.env.example`을 복사해서 `.env`를 만들고 값을 채운다.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 — 팀장에게 문의 |

> `.env`는 절대 커밋하지 않는다. `.gitignore`에 포함되어 있음.

## 4. PDF 받기

Git에는 PDF가 포함되어 있지 않다. 아래 구글 드라이브에서 `ask-my-appliance-data.zip`을 받아서
압축을 풀면 나오는 `data/` 폴더 내용물을 레포 루트의 `data/` 밑에 그대로 넣는다.

**다운로드**: https://drive.google.com/file/d/1ePqOVFxh0t0qK-fjMoVuMGw4LbdHV-XF/view?usp=sharing

압축 풀면 아래 구조 그대로 들어있다:

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

기본 에러코드 데이터도 같은 공유 드라이브에서 같이 받는다.

> 이번 라운드는 PDF 텍스트 + 에러코드 데이터만 다룬다. 이미지/그림(다이어그램)
> 추출, OCR은 하지 않는다 — 나중에 RAG 고도화 단계에서 별도로 진행할 예정.

## 5. 세팅 확인

`experiments/rag_tutorial.ipynb`를 열어서 셀을 순서대로 실행한다.

- API 키 OK
- PDF 파일 목록 출력
- LLM 응답 출력

세 가지가 모두 정상이면 실험 시작 준비 완료.

## 6. 실험 시작

```bash
git checkout -b exp/{본인이름}
cp experiments/template_exp.py experiments/{이름}_exp01.py
```

이후 절차는 `experiments/RULES.md` 참고.

---

## 브랜치 전략

| 브랜치 | 용도 |
|---|---|
| `main` | 검증된 파일만 (팀장 머지) |
| `exp/{이름}` | 개인 실험 브랜치 |

PR 제목 형식: `[{이름}] exp{번호}: 한 줄 설명`
