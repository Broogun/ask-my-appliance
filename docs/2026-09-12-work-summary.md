# 2026-09-12 작업 정리

## 1. lg_data_PRO → rag_project 마이그레이션

`lg_data_PRO` 전체 파일을 `rag_project`로 복사 (`.git` 제외).

## 2. 경로 버그 수정

`scripts_batch_ingest_all_lg.py`, `scripts_batch_extract_figures_lg.py`, `scripts_batch_ingest_manuals.py`에서
`data/manuals_pdf/lg/` → `data/lg/`로 수정.

## 3. LG 청킹 개선 (pdf_chunker.py)

### 3-1. 깨진 헤딩 필터링 (`_is_heading`)
폰트 폴백(FONT) 방식에서 한글/영문/숫자가 전혀 없는 글자(`䤏`, `!` 등 cmap 손상 문자)를
헤딩으로 오인식하는 문제 수정. 정규식 `[가-힣a-zA-Z0-9]` 없으면 헤딩 제외.

### 3-2. TOC 최심 레벨 큰 섹션 문단 폴백 (`_split_by_paragraph`)
TOC에서 더 쪼갤 하위 레벨이 없는데 텍스트가 3,000자를 넘는 섹션을 빈 줄(`\n\n`) 기준
문단 단위로 재분할. 예: `WM_FC2521KX6C` "사용 관련" 8,159자 → 문단 단위 분할.

**개선 결과** (LG 30개 전체):
- 3,000자 초과 섹션: 기존 다수 → **1개** (3,215자, 문단 없어 더 분할 불가)
- 최대 섹션 길이: 19,946자 → **3,215자**

## 4. Docling 도입 가능성 점검

**결론: 도입 불필요.**
- LG 매뉴얼은 TOC가 있어 PyMuPDF 방식이 최적
- Docling은 PDF TOC/북마크 추출 미지원 → 현재 청킹 전략과 충돌
- 한국어 미검증, PyMuPDF 대비 15~35배 느림, 500MB+ 모델 필요

## 5. RAG 최적화 기법 점검

| 기법 | 판정 | 비고 |
|---|---|---|
| 리랭킹 (`bge-reranker-v2-m3-ko`) | ✅ 추천 | 30분, $0, +3~10% |
| 임베딩 파인튜닝 (bge-m3 + LoRA) | ✅ 추천 | 합성 데이터 생성 필요, +5~15% |
| HyDE | 💡 조건부 | 저신뢰도 쿼리에만 적용 |
| Parent-Child 청킹 | 💡 조건부 | 성능 측정 후 결정 |
| Graph RAG | ❌ 스킵 | ROI 없음, 단일 hop 쿼리 패턴 |

→ 메모리(`project_rag_optimization_plan.md`)에 저장 완료.

## 6. LG 30개 PDF ingest 완료

- 청크 수: **3,024개**
- 임베딩: `BAAI/bge-m3` (로컬)
- 벡터DB: ChromaDB (`appliance_manuals_local` 컬렉션)
- 그림 추출: `data/manual_figures/lg/` (벡터 드로잉 페이지 렌더링 + OCR)
- 에러코드 DB: `db/seed_error_codes.py` 실행 완료 (LG+삼성 51개 항목)

## 7. 중복 청크 이슈 실측 및 수정

**문제**: 오버랩 청크가 top-3을 같은 섹션으로 도배하는 현상 실측.
- `에어컨 필터 청소` 쿼리 → top-3이 `manual_19_1`, `_2`, `_3` (같은 섹션 3개)

**수정** (`query_engine.py` - `search()` 함수):
- `top_k * 3` 후보를 먼저 fetch
- 섹션 키(`{model}_manual_{section_idx}`) 기준 중복 제거
- 섹션당 가장 유사한 청크 1개만 유지 후 top_k 반환

**결과**: 5개 테스트 쿼리 전부 중복 없이 다양한 섹션 반환 확인.

## 8. 환경 및 설정

- `.env` 생성: `EMBEDDING_BACKEND=local`, `CHAT_BACKEND=openai`, `CHAT_MODEL=o4-mini`
- `.gitignore` 생성: PDF, chroma_db, manual_figures, .env 등 제외
- `SETUP.md` 생성: 팀원 환경 세팅 가이드

## 9. 남은 작업

- [ ] GitHub repo 세팅 및 팀 브랜치 전략 수립 (4명 각자 독립 실험)
  - 실험 통일 규칙 (테스트 질문 셋, 평가 지표)
  - 브랜치: `main` + `exp/member-{A~D}`
- [ ] 삼성 데이터 청킹 및 ingest (별도 담당자)
  - `TROUBLESHOOTING_KEYWORDS` 삼성 표현 추가 필요
  - 폰트 임계값 조정 필요 (삼성 매뉴얼 최대 폰트 11~12pt)
- [ ] 리랭킹 적용 (`dragonkue/bge-reranker-v2-m3-ko`)
- [ ] 임베딩 파인튜닝 (합성 질문 데이터 생성 후)
