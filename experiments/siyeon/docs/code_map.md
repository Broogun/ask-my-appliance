# siyeon — 코드 지도

> 이 문서는 "코드 어디에 뭐가 있고, 어떤 순서로 읽으면 되는가"용입니다. (위치: `experiments/siyeon/docs/code_map.md`)
>
> ```
> experiments/siyeon/
>   exp/    실행 진입점 — python experiments/siyeon/exp/exp06_all.py (최종 구성)
>   rag/    구현: chunking_lg.py  chunking_samsung.py  understand.py  retrieval.py
>   rdb/    SQLite: schema.sql  build_db.py
>   docs/   overview_lg.md  overview_samsung.md  experiment_notes.md  code_map.md(이 문서)
>   testset.json  chunk_questions.json  understand_cache.json
>   eval_testset.py  eval_answers.py  app.py(Streamlit)
> ```
> "왜 이렇게 설계했나"는 `overview_lg.md`, 실험 수치와 실패 기록은 `experiment_notes.md`. 웹 서비스 쪽은 `web/README.md`.

---

## 0. 한 문장 요약

**PDF를 목차대로 잘라 두고(rag/chunking_*.py) → 질문이 오면 등록된 모델의 조각만 뒤져서 3개를 고른다(rag/retrieval.py).
검색 앞에 LLM 1회로 기능명 정규화·다른 제품 감지만 한다(rag/understand.py). 에러코드와 "이 모델에 그 기능 있나"는 검색 대신 SQLite에서 조회한다(rdb/).**

```
[오프라인, 1회]
  data/lg/**/*.pdf      ──chunking_lg.build_all_chunks()────────▶ 청크 2,8xx개
  data/samsung/**/*.pdf ──chunking_samsung.build_samsung_chunks()─▶ 청크 2,2xx개
                                                         │
                     LGAirconRetriever(chunks) ──────────┼──▶ chroma_lg/langchain        (청크 본문 임베딩, bge-m3)
                                                         └──▶ chroma_lg/chunk_questions (청크별 예상 질문 4~8개 임베딩)
  data/*_errors.json + PDF 부록 표 ──rdb/build_db.py──▶ data/appliance.sqlite

[질문마다 — LLM 2회]
  query + 등록 가전 ──understand()──▶ Understanding(feature, other_product)
                    ──retriever.retrieve(query, doc_id, feature=und.feature)──▶ RetrievalResult(hits, used, route, appliance)
                    ──답변 LLM (팀 채점: build_context + SYSTEM_PROMPT / 웹: web/rag_service.py — 최근 4턴을 messages에 그대로)
```

멀티턴은 답변 LLM에 최근 대화를 그대로 넣는 것으로만 처리한다 — 검색 전에 질문을 다시 쓰는 단계(규칙 버전, LLM 버전 둘 다)는 새 증상을 후속으로 오판해서 뺐다
(`experiment_notes.md` 리팩토링·슬림 라운드).

---

## 1. 읽는 순서 (전부 읽지 마세요)

뼈대는 세 함수 ~90줄입니다. 나머지는 이 셋이 부르는 부품이라 "이름만 알고 넘어가도" 됩니다.

| 순서 | 파일 · 함수 | 줄 수 | 읽으면 알게 되는 것 |
|---|---|---|---|
| ★1 | `rag/retrieval.py` → `LGAirconRetriever.retrieve()` + `_route()` | ~50 | 질문 하나가 처리되는 전 과정(①코드 ②기능 ③벡터 ④컷). **이것만 읽어도 절반** |
| 2 | `rag/retrieval.py` → `search_dense()` + `_candidates()` | ~30 | ③단계 내부: 벡터 두 갈래 → RRF → 리랭커 |
| 3 | `rag/understand.py` → `understand()` | ~40 | 검색 전 LLM 1회: 기능명 enum·다른 제품. 출력은 원문에 있는 단어만 인정 |
| ★4 | `rag/chunking_lg.py` → `parse_pdf()` | ~30 | PDF 한 개가 청크가 되는 과정. 챕터별로 파서 3개 중 하나를 고름 |
| 5 | `rag/chunking_lg.py` → `parse_by_toc()` / `parse_troubleshooting()` | ~95 | 목차 제목 줄에서 끊는 기본 파서 / 고장신고 표 → 증상 1개 = 청크 1개 |
| — | 그 외 전부 | | 아래 "함수 사전"에서 한 줄 설명만 확인 |

`retrieve()`를 읽을 땐 `docs/overview_lg.md`의 **"최종 구조 — 4단계 (상세)"** 절을 옆에 두세요. 같은 흐름을 말로 쓴 것입니다.

---

## 2. 함수 사전

### `rag/chunking_lg.py` — LG PDF → 청크

**청크 dict 형식** (모든 함수의 공통 통화):
```python
{"id": "AC_FQ18GC1EHN_관리하기_청소하기_>_필터_청소하기_7741f970",   # md5 기반, 재실행해도 동일
 "doc_id": "AC_FQ18GC1EHN",        # PDF 파일명 = 모델 식별자. 에러코드는 "LG-AC-COMMON" 등
 "brand": "lg", "category": "aircon", "product_type": "에어컨",
 "section": "관리하기", "subsection": "청소하기 > 필터 청소하기",
 "body": "일정한 주기마다 …",       # 본문
 "tag": "generic",                 # howto | symptom | safety | diagnosis | generic | error_code
 "page_start": 21, "page_end": 22,  # 출처 카드 'p.21' (assign_pages)
 "text": "관리하기 > 청소하기 > 필터 청소하기\n일정한 주기마다 …"}   # 임베딩되는 문자열 = 경로 + 본문
```

| 함수 | 한 줄 설명 | 언제 고치나 |
|---|---|---|
| **`build_all_chunks(include_legacy=True)`** | 진입점. `data/lg/{aircon,fridge,washer}/*.pdf` 전부 + 에러코드 JSON → 청크 목록 | 폴더가 늘면 `categories` 인자 |
| **`parse_pdf(pdf_path)`** | PDF 1개 → 청크. 챕터 범위 잡고, 챕터 종류별로 파서 선택. 목차 없으면 `parse_pdf_legacy`로 | 새 챕터 종류가 생기면 여기 분기 |
| `locate_sections(page_texts)` | 대챕터별 페이지 범위. "챕터명이 두 줄 연속 반복되는 페이지 = 챕터 시작" 규칙 | — |
| `toc_headings(doc)` / `_toc_pages(doc)` | PDF 북마크에서 2·3단계 소제목 목록 / 목차 페이지(제목 40%+가 줄로 있는 페이지)는 제목 탐색에서 제외 | — |
| `parse_by_toc(...)` | 기본 파서. 목차 제목과 같은 줄에서 청크를 끊는다. `detect_hidden=True`면 목차에 없는 4단계 소제목도 감지. `SPEC_HEADERS`(규격 표) → `제품 규격` 조각 분리 | — |
| `parse_troubleshooting(...)` | 고장신고 표(`증상｜원인 및 해결책`)를 읽어 증상 1개 = 청크 1개 (본문 첫 줄 `증상: …`) | 표 형식이 다른 브랜드가 오면 |
| `parse_pdf_legacy(pdf_path)` | 목차 없는 구형 PDF 3개용. 굵고 큰 글씨 또는 `7. 냉장실홈바` 줄을 소제목으로. 띄어쓰기 없는 PDF는 kiwi로 복원 | 구형 PDF가 더 오면 |
| `load_error_code_chunks()` | `data/lg_*_errors.json` → 1항목 = 1청크 (`tag=error_code`, `doc_id=LG-AC-COMMON` 등) | — |
| `strip_page_furniture(text)` | 페이지 번호·러닝헤더 제거 | — |
| `make_chunk(...)` / `split_oversized(...)` / `to_document(c)` | 청크 dict 생성 / 1,200자 초과 시 줄 경계 분할(100자 겹침) / LangChain `Document` 변환 | 청크 크기 바꿀 때 `MAX_CHUNK_CHARS` |
| `common_doc_id_for(category, brand)` | 카테고리·브랜드의 공통 에러코드 문서 id (`LG-AC-COMMON`) | — |

**상수**: `CHAPTER_BASES`(대챕터 11개), `HOWTO_BASES`(4단계 소제목 감지할 챕터), `SKIP_GROUPS`(색인 제외), `SPEC_HEADERS`, `MAX_CHUNK_CHARS=1200`.

### `rag/chunking_samsung.py` — 삼성 PDF → 청크 (LG 헬퍼 재사용)

| 함수 | 한 줄 설명 |
|---|---|
| **`build_samsung_chunks()`** | 진입점. `data/samsung/**/*.pdf` + `samsung_*_errors.json` |
| **`parse_pdf_samsung(pdf_path)`** | 목차 1단계 페이지로 챕터 범위 → 챕터 이름 키워드로 파서 선택 (`고장·문제` → 증상, `주의·안전` → safety, 나머지 → `parse_by_toc`) |
| `chapters_from_toc(doc)` | 목차 1단계를 본문 제목 줄로 검증·재배치 (목차가 깨진 문서 대응) |
| `parse_trouble_samsung(...)` | 표 5종(`증상｜확인/조치`, `코드｜진단｜해결방법` …) + 표 없는 텍스트형 증상 감지(글자 크기 / `~나요!`+불릿) |
| `strip_furniture_samsung(text, chapter_titles)` | 페이지 번호·러닝헤더 제거 |
| 상수 `TROUBLE_WORDS`, `SAFETY_WORDS`, `TABLE_HEADERS`, `SKIP_TITLES` | 챕터 역할 키워드, 표 헤더, 색인 제외 챕터 |

LG 파서와 나눠 둔 이유: 브랜드별로 달라지는 건 "챕터를 어떻게 찾나"와 "고장 부분이 표인가 글인가" 두 가지뿐이라 그 부분만 따로 둔다.
공통 부품(`make_chunk`, `split_oversized`, `parse_by_toc`, `parse_pdf_legacy`, `load_error_code_chunks`)은 import.

### `rag/understand.py` — 질문 이해 (LLM 1회, 검색 앞)

| 함수 · 형식 | 한 줄 설명 |
|---|---|
| **`understand(query, appliance)`** | gpt-4o-mini structured output 1회(타임아웃 4초, 실패하면 판단 없음) → `Understanding(query, feature, other_product)`. 결과는 `understand_cache.json`에 캐시 |
| `feature_vocab(category, brand)` | 그 브랜드·제품군 기능 표(model_features)의 기능 이름 → feature **enum**. 표가 없으면(삼성 전부, LG 세탁기·냉장고) 빈 목록 → feature는 항상 null |
| `_validate(raw, …)` | feature_evidence(근거 단어)·other_product가 원문에 없으면 무시 |

후속 질문 재작성은 없다. 멀티턴은 `web/rag_service.py`가 최근 4턴을 답변 LLM messages에 그대로 넣는다.

### `rag/retrieval.py` — 질문 → 청크 3개

| 함수 | 한 줄 설명 |
|---|---|
| **`LGAirconRetriever(chunks, use_reranker=True, verbose=True)`** | 생성자. 인덕스 2개(본문 벡터·예상질문 벡터) + SQLite 연결. 디스크에 있는 임베딩은 재사용, 없는 청크만 임베딩. 장치는 `DEVICE`(cuda 있으면 자동) |
| **`retrieve(query, doc_id=None, top_k=3, appliance=None, feature=None, context_cutoff=0.9)`** | **최종 API.** `_route()`(①②③) → ④ `apply_cutoff` → `RetrievalResult(hits, used, route, appliance)`. route ∈ error_code / feature_absent / vector / none. doc_id가 없으면 질문 속 모델명으로 가전 해석(팀 채점) |
| `_route(query, appliance, manual_id, …)` | ① 코드 → SQLite ② feature가 기능 표에서 '없음' → 확정 ③ `search_dense` + 약한 에러코드 청크 제거(≥ 0.5) |
| `search_dense(query, doc_id, top_k, n_candidates)` | 벡터(본문) + 벡터(예상질문→부모 청크) → RRF → 후보 8 → 리랭커(bge-reranker-v2-m3) → top-k. 거리 = 1 − 리랭커 확률 |
| `_candidates(query, doc_id, n)` | 위의 "후보 8" 부분 |
| `search_baseline(query, doc_id)` | 벡터(본문)만. **비교 실험용** |
| `lookup_error_codes(brand, category, query)` | 질문 속 코드 정규화 → `error_codes` 조회. 브랜드·제품군으로 범위 제한 (FF가 세탁기/냉장고에 다 있음) |
| `feature_available(appliance, feature)` | 기능 표 GLOB 조회 → True/False/None(표에 없음 → 검색으로) |
| `appliance_for(doc_id)` / `appliance_from_query(query)` | doc_id → 가전 dict / 질문 속 모델명(팀 채점 질문) → 가전 (manual_models 표 조회) |
| `make_answer_fn(doc_id, top_k, use_understanding=False, cache_path=None)` | 팀 채점 형식 `my_answer(query) → {answer, candidates}` |
| 모듈 함수 `apply_cutoff` / `build_context` / `appliance_label` | 관련도 컷 / 팀 채점용 [검색된 문서] 구성 / "LG 세탁기 FC4KC" |
| `_sync_collection` / `_ensure_metadata` / `_build_question_cache` | 인덕스 관리: 없는 id만 추가, 옛 문서 메타데이터 보강, 예상 질문 생성(gpt-4o-mini)·캐시 |
| `_as_filter(doc_id)` / `_scope_ids(doc_id)` | Chroma 필터 `{"doc_id": {"$in": [모델, 그 브랜드·카테고리 공통 에러코드 문서]}}` |

**상수**: `N_CANDIDATES=8`, `RERANKER_MAX_LENGTH=384`, `CONTEXT_CUTOFF=0.9`(관련확률 10% 미만은 LLM에 안 줌), `ERROR_CHUNK_MAX_DIST=0.5`, `PERSIST_DIR=chroma_lg/`, `DB_PATH`, `QUESTIONS_CACHE`.

**뺀 것 (2026-09-18, 실측·실사용 근거)**: BM25 · 후속 질문 재작성(규칙 5겹 → LLM 1회 → 제거) · 멀티쿼리 · listwise · 태그 가산 · `RERANK_WITH_QUESTION` · 형제 모델 경로 ·
브랜드·제품군 추정(단어 목록 → LLM → 제거) · 기능명 별칭 사전 · `"X 있어요?"` 정규식 · `spec_hint` · `suggestions` · `search`/`search_v2`/`search_v3` 3단. 이유와 수치는 `experiment_notes.md`.

### `rdb/` — SQLite

| 파일 · 함수 | 한 줄 설명 |
|---|---|
| `rdb/schema.sql` | 테이블 정의 + 각 테이블이 왜 있는지 주석 + 조회 예시. ERD는 `docs/overview_lg.md` |
| `rdb/build_db.py build()` | PDF 60개(md5, 부록 기능 표) + JSON 6개 → `data/appliance.sqlite` 재생성 |
| `normalize_code` / `code_variants` / `extract_query_codes` | 코드 표기 통일(`ch 05`→`CH5`), 혼동 문자 변형(`0E`↔`OE`), 질문에서 코드 추출. **retrieval.py도 이걸 import** |
| `parse_codes(title)` | `"Er(E) FF / Er(E) rF"` → `ERFF, EFF, FF, ERRF, ERF, RF` |
| `extract_feature_tables(doc)` | 부록 `모델명 | 기능…` 표 추출 (LG 에어컨만 있음) |
| `FEATURE_IMPLIED_BY` | `먼지통 → 클린봇` 처럼 부품이 기능을 함의하는 표 (retrieval의 기능 표 조회에 사용) |

---

## 3. 자주 하는 변경 → 어디를 건드리나

| 하고 싶은 것 | 위치 |
|---|---|
| 새 PDF 추가 | `data/{brand}/<category>/`에 넣기 → `build_db.py` 재실행 → 다음 실행 때 자동 청킹·임베딩 (없는 id만) |
| 청크 크기 조정 | `chunking_lg.MAX_CHUNK_CHARS`, `CHUNK_OVERLAP_CHARS` |
| 리랭커 속도/정확도 | `retrieval.N_CANDIDATES`, `RERANKER_MAX_LENGTH`; CPU만 있으면 `RAG_DEVICE=cpu` |
| 질문 이해(기능명·다른 제품) | `understand._SYSTEM` — 프롬프트 한 곳. 검증은 `_validate` |
| 기능 유무 판정을 넓히기 | 기능 표를 넣는다 — `build_db.extract_feature_tables` (지금은 LG 에어컨 부록 표만) |
| 멀티턴 | `web/rag_service.HISTORY_MESSAGES` (답변 LLM에 넣는 최근 메시지 수) |
| 에러코드 항목 수정 | `data/*_errors.json` → `build_db.py` 재실행 (벡터 청크는 다음 실행 때 자동 갱신) |
| 웹(FastAPI)에서 쓰기 | `web/rag_service.py` — `understand_query` / `retrieve` / `stream_answer` / `sources_of` 4개만 (web/README.md) |

---

## 4. 실행 파일과 데이터 파일

| 파일 | 역할 |
|---|---|
| `exp/exp01~02` | 에어컨 10개 시절 실험 (옛 22문항 채점 체계). 기록용 |
| `exp/exp03_lg.py` / `exp04_lg.py` / `exp05_all.py` | LG 27 → LG 30 → LG 30 + 삼성 30, 900문항 채점 |
| `exp/exp06_all.py` | **최종 구성** — 팀 채점은 보류. exp01~05는 옛 API로 돌린 기록(결과 JSON은 results/에 있음) |
| `eval_testset.py` | 자체 테스트셋 256문항 채점 (`--mode v3/dense/base`, `--no-understand`, `--ids a,b`) |
| `eval_answers.py` | 답변 품질 LLM 심판 30문항 — 팀 프롬프트 vs 웹 프롬프트 (근거 충실·맞는 근거 선택·방향 보존) |
| `app.py` | Streamlit 실험 GUI (모드 3개) |
| `testset.json` | 자체 테스트 256문항 (가전 미지정 10문항은 웹이 타지 않는 경로라 제외) |
| `chunk_questions.json` / `understand_cache.json` | LLM 산출 캐시 — **커밋 대상** (재생성에 API 비용, 채점 재현) |
| `chroma_lg/`, `data/appliance.sqlite` | 생성물, git 제외. 없으면 첫 실행 때 자동 생성 (임베딩 ~25분) |

---

## 5. 남은 정리

- 클래스명 `LGAirconRetriever` → `ManualRetriever` (에어컨 전용이 아니게 된 지 오래 — 호출부 6곳 같이)
- `make_answer_fn`의 OpenAI 호출을 평가 유틸로 분리 (검색기는 검색만)
- LangChain으로 옮길 때: `retrieve` = `BaseRetriever` 어댑터, 답변 = `RunnableWithMessageHistory` 체인 (지금 rag_service가 하는 것과 같음)
