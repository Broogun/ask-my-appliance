# siyeon — LG·삼성 가전 설명서 RAG

```
experiments/siyeon/
  exp/    실행 진입점       python experiments/siyeon/exp/exp06_all.py → results/siyeon_exp06.json  (exp06 = 최종 구성)
  rag/    구현 본체         chunking_lg.py  chunking_samsung.py  retrieval.py(SQLite 조회 + 벡터 검색)  understand.py(기능명·다른 제품, LLM 1회)
  rdb/    SQLite            schema.sql  build_db.py   (에러코드·모델별 기능 표·모델↔매뉴얼 매핑)
  docs/   설명 문서         아래 표
  testset.json              자체 테스트 256문항  (채점: python experiments/siyeon/eval_testset.py  · --ids a,b 로 일부만)
  eval_answers.py           답변 품질 LLM 심판 30문항 — 팀 프롬프트 vs 웹 프롬프트 (python experiments/siyeon/eval_answers.py --n 30)
  app.py                    Streamlit 실험 GUI  (streamlit run experiments/siyeon/app.py)
  ../../web/                FastAPI 웹 서비스 — 멀티턴은 답변 LLM에 최근 4턴을 그대로 (web/README.md)
  chunk_questions.json      청크별 예상 질문 캐시 v2 (커밋 대상 — 재생성에 API 비용). v1(4개씩)은 chunk_questions_v1.json
  understand_cache.json     질문 이해(LLM) 결과 캐시 (커밋 대상 — 채점 재현)
  notebook_lg_aircon.ipynb  탐색·시각화 노트북
```

| 먼저 읽을 것 | 내용 |
|---|---|
| `docs/overview_lg.md` | **여기서 시작.** 무엇을 만드는지, 청킹·검색 전략, 검색 4단계 상세, SQLite ERD |
| `docs/overview_samsung.md` | 삼성에서 무엇이 달랐고 어떻게 맞췄나 |
| `docs/code_map.md` | 코드 지도 — 어떤 함수가 어디 있고 어떤 순서로 읽나 |
| `docs/experiment_notes.md` | 실험 수치·시행착오 기록 (exp01~06, 리팩토링 라운드) |

## 실행

```bash
python experiments/siyeon/rdb/build_db.py --check    # data/appliance.sqlite 생성 (몇 초)
python experiments/siyeon/exp/exp06_all.py           # LG 30 + 삼성 30 팀 채점 → experiments/results/ (자체 채점 결과는 experiments/siyeon/results/)
python -m uvicorn web.main:app --port 8000            # 웹 서비스 (web/README.md)
```

처음 실행이면 임베딩(5,046청크 + 예상질문 20,180개)이 먼저 돌아 20~25분, 이후엔 `chroma_lg/` 디스크 인덱스를 재사용해 즉시 시작합니다.
`chroma_lg/`, `data/appliance.sqlite`, `data/answer_cache/` 는 생성물이라 git에서 제외됩니다.
