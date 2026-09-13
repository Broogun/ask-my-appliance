# 실험 규칙

## 파일 네이밍

```
experiments/
  {이름}_exp{번호}.py       # 실험 스크립트
  results/
    {이름}_exp{번호}.json   # 자동 저장 결과 (Git 제외)
```

예시: `minsoo_exp01.py`, `sujin_exp02.py`

번호는 01부터 시작하고, 같은 주제의 개선은 같은 번호에 덮어쓰지 말고 번호를 올린다.

---

## 브랜치 전략

```
main
└── exp/{이름}     # 각자 작업 브랜치
```

- 본인 브랜치(`exp/{이름}`)에서만 커밋한다.
- 실험 결과를 공유하고 싶을 때 PR → `main`.
- PR 제목 형식: `[{이름}] exp{번호}: {한 줄 설명}`
  - 예: `[minsoo] exp02: 리랭킹 적용 결과`
- `main` 머지는 팀장이 최종 확인 후 진행한다.

---

## 실험 작성 규칙

1. `template_exp.py`를 복사해서 시작한다.
2. `EXPERIMENT_NAME`과 `NOTES`는 반드시 채운다.
3. `my_answer()` 함수의 반환 형식을 바꾸지 않는다.
   ```python
   {"answer": str, "candidates": list[dict], "source": str}
   ```
4. 공통 평가(`COMMON_QUESTIONS`)로 반드시 결과를 뽑은 뒤 PR에 첨부한다.
5. `pipeline/` 코드를 직접 수정하지 않는다. 본인 실험 파일 안에서만 override한다.

---

## 변수명 규칙

| 역할 | 변수명 |
|---|---|
| 사용자 질문 | `query` |
| 검색 결과 목록 | `candidates` |
| 검색 결과 개수 | `top_k` |
| 제품 모델 필터 | `model_filter` |
| 청크 고유 ID | `chunk_id` |
| 섹션 제목 | `section_title` |
| 거리(유사도 역수) | `distance` (낮을수록 좋음) |

---

## 평가 지표

`pipeline/evaluate.py`의 `run_eval()` + `print_report()`를 사용한다.

| 지표 | 설명 |
|---|---|
| 키워드 히트율 | 기대 키워드가 답변/근거에 포함된 비율 |
| 평균 거리 | 검색된 청크의 유사도 (낮을수록 좋음) |
| 평균 응답 시간 | LLM 포함 전체 지연 |

최종 비교는 팀장이 JSON 결과를 모아서 진행한다.

---

## 공통 테스트 질문

`pipeline/evaluate.py`의 `COMMON_QUESTIONS` 참고.
질문 추가·수정은 팀장 승인 후 PR.
