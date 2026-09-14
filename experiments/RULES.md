# 실험 규칙

## 통일 vs 자유

| 항목 | 통일 (수정 금지) | 자유 (각자 결정) |
|---|---|---|
| 데이터 | `data/lg/` + `data/samsung/` LG·삼성 PDF 텍스트 + 기본 에러코드 데이터 | - |
| LLM | gpt-4o-mini | - |
| 답변 형식 | `SYSTEM_PROMPT` (한국어, 근거 기반) | - |
| 평가 질문 | `COMMON_QUESTIONS` 22개 | - |
| 반환 형식 | `{"answer": str, "candidates": list}` | - |
| 청킹 방식 | - | 고정크기 / 문단 / 슬라이딩 / 기타 |
| 임베딩 모델 | - | OpenAI / bge-m3 / 기타 |
| 벡터 DB | - | ChromaDB / FAISS / numpy / 기타 |
| 검색 전략 | - | top_k / 필터 / 중복 제거 등 |

**이번 라운드 범위 밖**: 이미지/그림 추출, OCR은 하지 않는다 — 나중에 RAG 고도화
단계에서 별도 진행 예정. PDF 텍스트와 에러코드 데이터만으로 실험한다.

---

## 실험 파일 구조

```
experiments/
  {이름}_exp{번호}.py     ← 실험 스크립트 (본인 브랜치에 커밋)
  results/
    {이름}_exp{번호}.json ← 평가 결과 (커밋 필수 — GitHub Actions가 사용)

pipeline/
  evaluate.py             ← 공용 평가 하니스 (COMMON_QUESTIONS/SYSTEM_PROMPT/채점/
                             보고서 저장). template_exp.py가 이미 import해서 쓰고
                             있으니 실험 파일에 복붙할 필요 없음 - my_answer()만
                             구현하면 된다.
```

- 번호는 01부터 시작, 개선할 때마다 번호 올리기 (덮어쓰지 말 것)
- 예: `minsoo_exp01.py`, `minsoo_exp02.py`

---

## 브랜치 전략

```
main
└── exp/{이름}   ← 각자 작업 브랜치
```

- 본인 브랜치에서만 커밋한다
- PR 제목 형식: `[{이름}] exp{번호}: {한 줄 설명}`
  - 예: `[minsoo] exp02: 문단 청킹 + bge-m3 적용`
- `main` 머지는 팀장이 최종 확인 후 진행

---

## 실험 작성 규칙

1. `template_exp.py`를 복사해서 시작한다.
2. `EXPERIMENT_NAME`, `NOTES`, `STRATEGY` 를 반드시 채운다.
3. `my_answer(query)` 하나만 구현한다. 청킹·임베딩·DB·검색 전부 이 안에서 자유롭게.
4. 반환 형식은 반드시 지킨다.
   ```python
   {
       "answer": str,
       "candidates": [
           {"id": str, "text": str, "distance": float},
           ...
       ],
   }
   ```
5. `COMMON_QUESTIONS`와 `SYSTEM_PROMPT`는 수정하지 않는다 (`pipeline/evaluate.py`에서
   import돼 있어서, 애초에 실험 파일 안에 이 값들이 없다 - 건드릴 곳도 없다).
6. 실험 실행 후 `results/{EXPERIMENT_NAME}.json`이 생성되면 함께 커밋한다.

---

## STRATEGY 작성 요령

보고서에 전략과 이유가 자동으로 포함된다. 성의 있게 작성할 것.

```python
STRATEGY = {
    "chunking":         "어떤 방식으로 청킹했는지",
    "chunking_reason":  "왜 그 방식을 선택했는지",
    "embedding":        "어떤 모델을 썼는지",
    "embedding_reason": "왜 그 모델을 선택했는지",
    "db":               "어떤 DB를 썼는지",
    "db_reason":        "왜 그 DB를 선택했는지",
    "retrieval":        "검색 전략 (top_k, 필터 등)",
    "retrieval_reason": "왜 그 전략을 선택했는지",
}
```

---

## 평가 지표

실험 실행 시 자동으로 계산된다.

| 지표 | 설명 |
|---|---|
| 키워드 히트율 | 기대 키워드가 답변/근거에 포함된 비율 (높을수록 좋음) |
| 평균 거리 | 검색된 청크의 유사도 거리 (낮을수록 좋음) |
| 평균 응답 시간 | 전체 파이프라인 지연 시간 |

팀장이 `python experiments/compare_results.py` 실행 시 전체 비교 보고서 자동 생성.  
PR push 시 GitHub Actions가 자동으로 PR 댓글에 보고서를 달아준다.

---

## 공통 테스트 질문

`pipeline/evaluate.py`의 `COMMON_QUESTIONS`가 원본이다 - 여기 문서에는 표로
복사해두지 않는다(복사해두면 딱 지금처럼 코드가 바뀌어도 문서가 그대로 남아
어긋나기 쉽다). 총 22개, LG·삼성 x 에어컨·냉장고·세탁기 6개 조합을 골고루
커버하고 에러코드 조회·증상 기반·설정/조작·브랜드 특화 기능·복합 질문·
브랜드 미언급 질문·안전 관련까지 다양한 유형이 섞여 있다. 질문 추가·수정은
팀장 승인 후 PR.

---

## 왜 코드를 합치지 않는가?

일반 개발은 기능을 합치는 게 목표지만, 이 실험은 **각자의 독립된 구현을 비교**하는 게 목표다.

- 코드를 합치면 실험 조건이 섞여서 비교가 불가능해진다
- 각자의 결과 JSON만 모아서 비교한다
- 실험이 모두 끝난 뒤 가장 성능이 좋은 구현을 `main`에 반영한다
