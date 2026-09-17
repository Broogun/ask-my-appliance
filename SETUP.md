# 환경 세팅 가이드

## 1. 레포 클론

```bash
git clone https://github.com/Broogun/ask-my-appliance.git
cd ask-my-appliance
```

## 2. 패키지 설치

**Python 3.12.14 사용** (GitHub Actions 자동 보고서 생성도 3.12.14 기준. `.python-version`
파일에 명시돼 있음 - pyenv 쓰면 자동으로 맞춰짐).

```bash
python --version   # 3.12.14인지 확인
pip install -r requirements.txt
```

**GPU가 있으면 (선택, 강력 권장):** 위 명령은 CPU 전용 torch를 깔아서 검색기의 리랭커가 질문당 3초 걸린다.
NVIDIA GPU가 있는 노트북이면 CUDA 빌드로 바꾸면 0.3초가 되고, 자체 테스트셋 256문항 채점도 18분 → 5분.

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch    # 드라이버가 CUDA 12.6 이상이면
python -c "import torch; print(torch.cuda.is_available())"                                 # True 면 성공. 코드는 자동 감지
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

## 5-b. siyeon 검색기 / 웹 앱 처음 실행 (해당 브랜치를 받은 사람만)

`experiments/siyeon/README.md`, `web/README.md` 참고. 처음 한 번은 오래 걸린다:

- `data/appliance.sqlite` 가 없으면 자동 생성 (`experiments/siyeon/rdb/build_db.py`, 1분)
- `chroma_lg/` 벡터 인덱스가 없으면 청크 5천 개 + 예상질문 3.7만 개를 임베딩 — **CPU 30~40분, GPU 3~5분**.
  예상질문 자체는 `experiments/siyeon/chunk_questions.json` 에 캐시돼 있어 OpenAI 호출·과금은 없다.
- 그 다음부터는 서버 시작 1~2분(모델 로딩)

```bash
python -m uvicorn web.main:app --port 8000        # http://localhost:8000  로그인 admin1~7 / 1234
streamlit run experiments/siyeon/app.py            # 검색 결과·근거 조각을 뜯어보는 실험 GUI
python experiments/siyeon/eval_testset.py --tag {태그}   # 자체 테스트셋 256문항 채점
```

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
