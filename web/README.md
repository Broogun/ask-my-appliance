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
    catalog_router.py  GET /api/catalog/options · /models?product_type&brand&q · /support/{brand} · GET /api/manuals/{doc_id}/pdf · GET /api/manuals/{doc_id}/page/{n}.png
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

출처 카드: `page_start/page_end`로 `p.21` 표시, 클릭하면 조각 전문 + `GET /api/manuals/{doc_id}/pdf#page=21` iframe. 카드에는 그 페이지의 렌더링 썸네일(`/page/21.png`, PyMuPDF로 렌더링해 `data/page_images/`에 캐시 — 벡터 그림 포함)이 붙고, 같은 페이지 출처는 첫 카드에만 표시.

**답변 속 설명서 그림** (`manual_figures.py`, LLM 없이 결정론): 답변 스트림이 끝나면 `{"type":"figures","items":[{"line":i,"figures":[{url,caption}]}]}` 를 보내고, 프런트가 답변의 i번째 줄 바로 뒤에 그림을 끼운다(`renderAnswerHtml`). 저장하지 않고 대화를 열 때마다 다시 계산한다(`chat_router._figures`).
- 그림 검출: 페이지를 렌더링해 글자를 지우고 남은 덩어리(표·구분선·글상자 제외). 드로잉 사각형 방식은 마스크 도형 때문에 이웃 그림과 합쳐져서 못 씀. 크롭은 `GET /api/manuals/{doc_id}/page/{n}/fig/{k}.png`.
- 그림↔답변 매칭: 그림마다 이름표 후보(2단 페이지는 앞 소제목 / 그림 바로 위 2줄 / 위 전체 / 좌우 같은 높이 글 / 아래 범례)를 만들고, 답변의 한 줄(범례는 목록 블록 전체)과 글자 4-gram이 충분히 겹칠 때만 붙인다. 못 찾으면 안 붙인다.
- `manual_pages.locate_pages`: 청크 메타데이터 page는 섹션 시작 쪽이라 31%가 1~5쪽 앞을 가리켜서, 조회 시점에 본문을 PDF 텍스트에서 찾아 보정한다(출처 카드 p.N과 PDF 링크도 이 값).
- 의미 유사도 보조 단계: 글자 매칭이 실패한 그림에 한해 로컬 임베딩(BGE-m3-ko)으로 답변 줄과 비교한다. 임베딩은 동작 방향(끄기/켜기, 넣기/꺼내기, 분리/조립)을 구분 못 해서 유사도 0.80↑ + 2위와 격차 0.15↑ + 반대 동작 쌍 충돌 시 거부의 세 조건을 모두 통과할 때만 붙인다(300문항 baseline에서 +10장, 눈으로 전부 맞음). 검색 엔진 로딩 중에는 건너뛴다.
- 매뉴얼 전체 보조 검색: 답변의 단계가 근거 페이지 밖의 절에서 온 내용이면(예: 문 경보음 답변의 "수평 조절 다리" 단계) 그림이 근거 페이지에 없다. 그래서 같은 매뉴얼 전체의 그림 이름표를 의미 유사도 0.78↑ + 글자 4-gram 3개↑ + 반대 동작 검사로 찾는다. 같은 양식 문구를 공유하는 동점 후보는 그림 내용(24x24 썸네일)을 비교해 같은 그림이면 하나만, 다른 그림이면 어느 쪽이 맞는지 모르므로 둘 다 뺀다. 300문항 baseline에서 +22장, 눈 검수 21/22 정확. 매뉴얼별 이름표+임베딩은 서버 시작 때 등록된 가전 것을 미리 계산(`prewarm_figures`, 매뉴얼당 ~4초).
- 예전 대화: 저장된 출처의 틀린 페이지는 열 때 `chat_router._fix_pages` 로 다시 보정한다(DB는 그대로).
- 측정(예전 300문항 baseline 답변에 적용, API 비용 없음, 최신 baseline 재계산 전이며 `docs/05` 5.7의 122/300과 값이 다름): 그림이 붙은 답변 116/300(39%). 그림 후보가 있는 페이지를 근거로 쓴 답변 223개 중 116개. 무작위 28쌍 눈 검수에서 26~27쌍이 맞음(약 93~96%), 확실한 오류 1건은 그림 위 글 수집이 문단 경계를 넘던 문제로 수정.
- 에러코드 사진: `scripts/crawl_error_images.py` 가 제조사 고객지원 페이지의 사진을 `data/error_images/`(gitignore, 저작권 때문에 저장소에 안 올림)에 받아 `error_entries.chunk_id` 에 연결한다(현재 LG 에어컨·세탁기(드럼/통돌이)·냉장고·김치냉장고 5개 페이지 → 26개 항목·사진 110장. 세탁기는 드럼/통돌이 페이지가 따로라 `variant` 로 기록하고 사용자 모델명(T로 시작=통돌이)으로 걸러 쓴다. 냉장고는 코드 표시부 사진만. 삼성 에러 안내 페이지는 확인한 범위에서 텍스트뿐이라 받을 사진이 없다). 대표 사진(표시부)은 답변 첫 본문 줄 뒤, 본문 사진은 설명(alt)이 답변 줄과 글자/의미로 맞을 때만 붙고, 화면에 "이미지 출처: LG전자 고객지원" 링크를 항상 표시한다. 서빙은 `GET /api/error-images/{folder}/{name}`.
- 한계: 옅은 색·조각난 그림(예: 구성품 사진 일부)은 놓치거나 일부만 잡힌다. 설명이 그림 위/아래 어느 쪽인지는 페이지 배치에 따라 다른데 위쪽 글을 우선한다(그림에 글이 없는 단계 사이의 그림은 다음 단계 것일 수 있음). 그림 수는 답변당 24장, 한 줄당 6장까지.

대화 행은 첫 질문을 보낼 때(`/start`) 만들어지므로 질문 없이 나간 대화는 기록에 안 남습니다. 시각은 KST.

## 답변 품질

팀 채점용 `SYSTEM_PROMPT`(experiments/harness/evaluate.py)는 비교 기준이라 그대로 두고, 웹은 `WEB_SYSTEM_PROMPT`를 씁니다.
실사용 로그에서 나온 결함 3종(근거가 있는데 포기 / 다른 증상의 조치를 섞음 / 바꿔 쓰다 뜻이 뒤집힘)을 규칙으로 겨누고,
`python experiments/siyeon/eval_answers.py`(LLM 심판 30문항)로 두 프롬프트를 비교합니다.

## 아직 없는 것

- (공유 DB는 구현됨) `DATABASE_URL` 이 postgres 면 앱 데이터·RDB·벡터(pgvector)·PDF(Storage)를 Supabase에서 읽는다 — 설정과 이전은 `supabase/README.md`
- 삼성 모델별 기능 표 (LG 부록 표만 있음) — 삼성 공용 설명서에서 "없는 기능"을 확정할 데이터가 없어 ②가 "모름"으로 떨어짐
