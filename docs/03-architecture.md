# 3. 시스템 아키텍처

## 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│  브라우저 (web/static)                                       │
│  로그인 → 제품 등록(온보딩) → 채팅 + 출처 패널               │
└───────────────────────────┬─────────────────────────────────┘
                            │ NDJSON 스트리밍
┌───────────────────────────▼─────────────────────────────────┐
│  FastAPI (web/main.py, web/routers/)                        │
│  auth · catalog · appliance · chat                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ 4함수 계약
┌───────────────────────────▼─────────────────────────────────┐
│  rag_service.py                                             │
│   understand_query → retrieve → stream_answer → sources_of  │
└──────┬──────────────────────────────────┬───────────────────┘
       │                                  │
┌──────▼─────────────┐          ┌─────────▼──────────────────┐
│ rag_myretriever.py │          │ manual_figures.py          │
│ ① 에러코드 RDB     │          │ 답변 줄 ↔ 설명서 그림 매칭  │
│ ② 기능 유무 표     │          │ manual_pages.py (쪽 보정)   │
│ ③ 벡터검색+리랭킹  │          └─────────┬──────────────────┘
└──────┬─────────────┘                    │
       │                                  │
┌──────▼──────────────────────────────────▼───────────────────┐
│  저장소                                                      │
│  Postgres+pgvector (Supabase)  ←→  SQLite + Chroma (로컬)    │
│  Supabase Storage (PDF)         ←→  data/ (로컬 PDF)         │
└─────────────────────────────────────────────────────────────┘
```

## 한 턴의 흐름

LLM 호출은 질문당 **2회로 고정**한다(질문 이해 1회 + 답변 생성 1회). 리랭킹이 필요한 경우만 1회 추가.

```
질문
 │
 ├─① understand_query()  LLM 1회 (~1초)
 │     feature       : 기능 질문이면 정식 명칭으로 정규화 ("자동청소" → 클린봇)
 │     other_product : 등록 가전이 아닌 제품이면 그 이름 → 등록 안내로 분기
 │
 ├─② retrieve()  라우팅 (아래 3경로 중 하나)
 │     error_code     : 질문에 코드가 있으면 RDB 정확 매칭 (LLM 0회, 0ms)
 │     feature_absent : 기능 표에서 '없음' 확정
 │     vector         : 벡터검색 top_k=40 → listwise LLM 리랭킹 → top_k=5
 │
 ├─③ stream_answer()  LLM 1회, 토큰 스트리밍
 │     WEB_SYSTEM_PROMPT 8개 규칙 + 번호 매긴 근거 + 최근 4턴 대화
 │     안전 경고는 코드가 결정론적으로 주입/제거 (LLM 판단에 맡기지 않음)
 │
 └─④ sources_of() + figures  출처 카드(쪽 번호·PDF·썸네일) + 답변 줄 사이 그림
```

경로(`route`)는 `error_code` / `feature_absent` / `vector` / `none` / `unregistered` 5가지이며,
화면 배지와 DB `messages.route`에 그대로 기록되어 나중에 분석할 수 있다.

## 모듈 구성

### 제품 코드

| 경로 | 역할 |
|---|---|
| `web/main.py` | FastAPI 앱, 정적 페이지, 시작 시 검색기 백그라운드 로딩 |
| `web/routers/` | `auth` · `catalog`(제품/PDF/이미지 서빙) · `appliance` · `chat`(스트리밍) |
| `web/rag_service.py` | **RAG 연결 지점** — 4함수 계약 + `WEB_SYSTEM_PROMPT` |
| `web/rag_myretriever.py` | 검색 엔진 — RDB 라우팅 + exp02 벡터검색·리랭킹 |
| `web/manual_figures.py` | 답변 줄 ↔ 설명서 그림 매칭 (LLM 없이 결정론) |
| `web/manual_pages.py` | PDF 경로 해석, 청크의 실제 쪽 번호 보정, Storage 폴백 |
| `web/rdb.py` | SQLite/Postgres 겸용 RDB 조회 계층 |
| `web/vecstore.py` | Chroma 인터페이스를 pgvector로 대체하는 어댑터 |
| `web/db.py` | SQLAlchemy 모델 (users / appliances / conversations / messages) |
| `scripts/` | 에러코드 사진 크롤러, 공유 DB 이전 스크립트 |
| `supabase/` | 공유 DB 스키마 + 설정 가이드 |

### 실험 코드

`experiments/`는 제품 코드가 아니라 **선택 근거를 남기는 기록**이다. 청킹·검색 전략을 비교한
실험 파일과 결과 JSON, 공용 평가 하니스가 들어 있다. 자세한 내용은
[`experiments/README.md`](../experiments/README.md).

## 검색 엔진 교체 가능한 설계

`rag_service.py`의 함수 4개가 **교체 지점**이다. 이 입출력만 지키면 검색기를 통째로 바꿔도
라우터·프런트·DB는 그대로 둘 수 있다.

| 함수 | 입력 | 출력 |
|---|---|---|
| `understand_query(query, appliance)` | 질문, 등록 가전 | `Understanding(query, feature, other_product)` |
| `retrieve(query, manual_id, feature)` | 질문, 매뉴얼 id, 기능명 | `RetrievalResult(hits, used, route, appliance)` |
| `stream_answer(query, res, history)` | 질문, 검색 결과, 최근 대화 | 답변 토큰 제너레이터 |
| `sources_of(res)` | 검색 결과 | 출처 패널 데이터 |

실제로 이 설계 덕분에 검색 엔진을 한 번 통째로 교체했다. 초기에는 CrossEncoder + RRF 기반
파이프라인이었는데, 실사용 로그에서 정답 청크를 놓치는 사례를 발견해 exp02 파이프라인
(TOC 청킹 + 넓은 후보 풀 + listwise LLM 리랭킹)으로 바꿨다. 이때 RDB와 제품 등록 흐름은
손대지 않았다.

## 저장소 이중화 (로컬 / 공유)

환경변수 `DATABASE_URL` 하나로 전환한다. 코드 분기는 조회 계층 안에만 있고 호출부는 동일하다.

| | 로컬 모드 (기본) | 공유 모드 (`DATABASE_URL`=postgres) |
|---|---|---|
| 앱 데이터 | `data/app.sqlite` | Supabase Postgres |
| RDB | `data/appliance.sqlite` | Supabase Postgres |
| 벡터 | Chroma (`chroma_db/`) | pgvector (`rag_chunks`) |
| PDF | `data/{brand}/{category}/` | Supabase Storage → 로컬 캐시 |

공유 모드는 팀원 4명이 각자 로컬 서버를 띄우면서도 같은 계정·대화·검색 데이터를 보게 하려는
목적이다. 임베딩 모델과 LLM 호출은 여전히 서버에서 일어나므로 Supabase가 서버를 대신하지는 않는다.

전환 시 검색 결과가 달라지지 않아야 하므로, pgvector 어댑터는 Chroma와 **거리 정의(제곱 L2)와
`get(ids)` 반환 순서**까지 맞췄다. 검증 결과는 [05-evaluation.md](05-evaluation.md#59-공유-db-전환-동등성)에 있다.

## 설계 결정 기록

| 결정 | 이유 | 근거 |
|---|---|---|
| 에러코드는 벡터가 아닌 RDB | 정확 매칭 문제를 유사도로 근사할 이유가 없음 | 조회 0ms, 정확도 100% |
| 리랭킹은 pointwise CrossEncoder → listwise LLM | 어휘만 겹치는 오답에 정답보다 높은 점수를 주는 사례 반복 확인 | 97케이스 중 94개 정확 매칭 |
| 안전 경고는 코드가 주입 | 프롬프트로 3차례 지시했으나 LLM이 계속 자체 생성 | 동일 질문 4회 재현 테스트 |
| 답변 그림은 LLM이 고르지 않음 | 위 이유와 동일 — 결정론적 매칭이 검증 가능 | 눈 검수 정밀도 93~96% |
| 쪽 번호는 조회 시점에 보정 | 청크 메타데이터의 쪽 번호가 31% 틀림 | 무작위 378개 중 117개 오류 |
| Self-RAG(문장 단위 자기검증)는 채택 안 함 | 막을 실패 사례가 측정되지 않음, 검색 단계 응답 보류로 이미 목표 달성 | FALSE_CONFIDENCE 0/300, `must_not_claim` 위반 0/9 |

공통 원칙은 하나다. **LLM의 판단을 신뢰해야만 성립하는 기능은, 검증 가능한 코드로 바꿀 수 있으면
바꾼다.**
