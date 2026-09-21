# web/ — 가전 도우미 웹 (FastAPI + HTML/CSS/JS)

```bash
# 프로젝트 루트에서
python -m uvicorn web.main:app --port 8000
# → http://localhost:8000   로그인: admin1~admin7 / 1234 (온보딩 화면을 다시 보려면 다른 계정으로)
```

서버가 뜨면 검색기(임베딩 모델·리랭커)를 백그라운드로 바로 로딩합니다 — GPU ~50초, CPU 1~2분. 그동안 채팅 화면엔 "준비 중" 배너가 뜨고,
그 사이 질문해도 준비되는 대로 답합니다. 준비 후 질문당 GPU 3~5초(이해 1초 + 검색 0.3초 + 답변 2~3초). `GET /api/health` → `{"ready": true}`.

> **`--reload`는 쓰지 마세요.** Windows에서 검색기를 든 워커가 재시작 신호에 멈춰 옛 코드로 계속 응답하고 포트를 물고 있는 일이 생깁니다.
> 코드 수정 중이라면 `--reload --reload-dir web` 으로 감시 범위를 web/ 에 한정하고, 이상하면 서버를 완전히 끄고 다시 켜세요.

## 한 턴의 흐름 (LLM 호출 2회로 고정)

```
질문 ──▶ ① LLM 질문 이해 (understand.py, ~1초)
              feature : 기능 질문이면 기능 표의 정식 이름으로 ("자동청소" → 클린봇)
              other_product: 등록 가전이 아닌 제품이면 그 이름 ("전자레인지") → 등록 안내로 분기
      ──▶ ② 라우터 (retrieval.py) — 질문 원문으로
              코드 있음 → SQLite 에러코드 / feature '없음' 확정 → 기능 없음 / 그 외 → 가전 필터 → 벡터×2 → RRF → 리랭커 → 관련도 컷
      ──▶ ③ LLM 답변 (WEB_SYSTEM_PROMPT + 번호 매긴 근거 + 최근 4턴 messages 그대로, 스트리밍)
```

멀티턴은 ③에서만 처리합니다 — LangChain `RunnableWithMessageHistory`처럼 최근 대화를 messages에 그대로 넣습니다(`HISTORY_MESSAGES=8`).
검색 전에 질문을 다시 쓰는 단계는 새 증상("냉장고에서 냄새나")을 직전 질문(소음)의 후속으로 오판해 합쳐 버려서 뺐습니다 (2026-09-18).

경로(route)는 `error_code / feature_absent / vector / none / unregistered` 다섯 가지. 화면 배지와 DB `messages.route`에 그대로 기록됩니다.
첫 화면의 기본 질문(제품군별 4개)은 그 제품군 모든 모델에서 근거가 나오는 것만 골랐습니다 (`tools/make_suggest.py`로 전수 검사).

## 구조

```
web/
  main.py              FastAPI 앱 — /api/* + 정적 페이지(/, /onboarding, /app). 기동 시 검색기 백그라운드 로딩
  db.py                SQLAlchemy 모델: users, user_appliances, conversations, messages(route, sources) → data/app.sqlite
  auth.py              로그인(admin1~7 / 1234 고정) · HMAC 토큰 · current_user 의존성
  rag_service.py       ★ RAG 연결 지점 — understand_query / retrieve / stream_answer / sources_of + WEB_SYSTEM_PROMPT
  routers/
    auth_router.py     POST /api/auth/login · /logout · GET /me
    catalog_router.py  GET /api/catalog/options · /models?product_type&brand&q · /support/{brand} · GET /api/manuals/{doc_id}/pdf
    appliance_router.py GET/POST/PATCH/DELETE /api/appliances
    chat_router.py     GET /api/conversations · POST /api/conversations/start(첫 질문) · POST /api/conversations/{id}/messages (NDJSON 스트리밍)
  tools/make_thumbs.py 설명서 PDF 제품 선화 → 썸네일 (실제 사진이 없을 때 대체)
  tools/make_suggest.py 기본 질문 후보를 60개 모델에 전수 검사 → 제품군별 4개 선정 (검색기·후보가 바뀌면 다시 실행, GPU ~10분)
  static/
    index.html         0. 로그인
    onboarding.html    1~3. 제품 찾기 → 확인 → 완료 (js/register.js)
    app.html + js/app.js  4~12. 새 대화(제품 선택) · 채팅+출처 패널 · 대화 기록 · 내 제품 · 모달(추가/수정/출처 PDF)
    css/base.css       와이어프레임 구조 그대로, 색만 입힘
    img/models/        제품 썸네일 — {manual_id}.png(실제 사진, LG 30개 있음) → auto/{id}.png(선화) → 아이콘 순 폴백 (README.md 참고)
```

## RAG를 바꾸고 싶을 때 (팀원용)

라우터·프론트·DB는 `rag_service.py`의 함수 4개만 봅니다. 이 형식만 지키면 검색기를 통째로 바꿔도 나머지는 그대로입니다.

| 함수 | 입력 | 출력 |
|---|---|---|
| `understand_query(query, appliance)` | 질문, 등록 가전 행 | `Understanding(query, feature, other_product)` |
| `retrieve(query, manual_id, feature)` | 질문, 등록 가전의 매뉴얼 id, 정규화된 기능명 | `RetrievalResult(hits, used, route, appliance)` — `route ∈ error_code/feature_absent/vector/none` |
| `stream_answer(query, res, history)` | 질문, 위 결과, 최근 메시지 `[{role, content}]` | 답변 토큰 제너레이터 |
| `sources_of(res)` | 위 결과 | 출처 패널용 `[{title, snippet, body, tag, doc_id, distance, chunk_id, page_start, page_end, pdf_url}]` |

LangChain으로 옮길 때: `retrieve` = `BaseRetriever` 어댑터 한 장, `stream_answer` = `RunnableWithMessageHistory` 답변 체인.

## 스트리밍 형식 (NDJSON)

```
{"type":"conv","conversation":{…}}                     ← /start(첫 질문)에서만 맨 처음
{"type":"status","text":"…"}                           ← 검색기 로딩 중일 때만
{"type":"sources","route":"vector","sources":[…],"unregistered":null}
{"type":"token","text":"LG 에어컨 "}                    ← 반복
{"type":"done","message_id":12}
```

출처 카드: `page_start/page_end`로 `p.21` 표시, 클릭하면 조각 전문 + `GET /api/manuals/{doc_id}/pdf#page=21` iframe.
대화 행은 첫 질문을 보낼 때(`/start`) 만들어지므로 질문 없이 나간 대화는 기록에 안 남습니다. 시각은 KST.

## 답변 품질

팀 채점용 `SYSTEM_PROMPT`(pipeline/evaluate.py)는 비교 기준이라 그대로 두고, 웹은 `WEB_SYSTEM_PROMPT`를 씁니다.
실사용 로그에서 나온 결함 3종(근거가 있는데 포기 / 다른 증상의 조치를 섞음 / 바꿔 쓰다 뜻이 뒤집힘)을 규칙으로 겨누고,
`python experiments/siyeon/eval_answers.py`(LLM 심판 30문항)로 두 프롬프트를 비교합니다.

## 아직 없는 것

- Postgres — `DATABASE_URL` 환경변수로 교체 (SQLAlchemy라 코드 변경 없음). 벡터도 옮기려면 pgvector
- 삼성 모델별 기능 표 (LG 부록 표만 있음) — 삼성 공용 설명서에서 "없는 기능"을 확정할 데이터가 없어 ②가 "모름"으로 떨어짐
