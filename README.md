# Ask My Appliance

**내 가전제품 설명서만 근거로 답하는 RAG 챗봇.**
LG·삼성 60개 모델의 사용설명서를 학습 데이터가 아닌 **검색 대상**으로 두고, 답변의 모든 문장에
출처 쪽 번호와 설명서 그림을 붙인다.

```
"dE 에러가 나고 문이 안 닫혀요"
 → dE 에러는 세탁기 문이 제대로 닫히지 않았을 때 발생합니다.
   1. 세탁물이 고무 패킹과 도어 사이에 끼어있지 않은지 확인하세요.  [설명서 그림]
   2. 도어가 아래로 처져 있다면 위쪽으로 살짝 들어 올려 닫아보세요.  [설명서 그림]
   출처: WM_F21VDSK p.28 · LG전자 고객지원
```

---

## 문제

가전이 고장 나면 사용자는 세 가지 벽에 부딪힌다. 설명서는 평균 57쪽이라 원하는 쪽을 못 찾고,
검색 결과는 어느 모델 기준인지 불분명하며, 제조사 챗봇은 브랜드별로 나뉘어 있다.

범용 LLM에 물으면 그럴듯한 답이 나오지만 **내 모델 설명서에 실제로 적힌 내용인지 확인할 수 없다.**
가전 문제는 오답의 비용이 크고, 특히 안전 경고는 틀리면 위험으로 이어진다.

→ 자세한 내용: [docs/01-problem.md](docs/01-problem.md)

## 해결 방식

| | 내용 |
|---|---|
| **모델 한정 검색** | 사용자가 등록한 가전의 설명서만 검색 범위로 사용 |
| **확정 가능한 건 DB로** | 에러코드·모델별 기능 유무는 벡터 검색이 아닌 관계형 DB 정확 매칭 |
| **근거 제시** | 답변마다 출처 쪽 번호 + PDF 원문 + 해당 설명서 그림 |
| **LLM 판단 최소화** | 안전 경고 주입, 그림 선택, 쪽 번호는 모두 결정론적 코드가 담당 |

## 결과

실제 서비스 경로로 round1 300문항을 평가한 결과다(`experiments/web_baseline_eval.py`, 2026-09-23 최종 코드 기준).

| 라벨 | 개수 | 비율 |
|---|---|---|
| SUCCESS | 268 | **89.3%** |
| GENUINE_NO_ANSWER (설명서에 정말 없음) | 29 | 9.7% |
| FALSE_DECLINE (근거가 있는데 포기) | 3 | 1.0% |
| **FALSE_CONFIDENCE (근거 없이 확신)** | **0** | **0%** |

LLM 판정 기준으로 근거 없이 지어낸 답변은 관측되지 않았다. 다만 판정자가 사람 검수와 대조되지 않았고, GENUINE_NO_ANSWER 29건 중 20건은 판정 후처리로 강등된 건이라 이 0건은 확정 수치가 아니다([docs/05 §5.10](docs/05-evaluation.md#510-한계)). 같은 경로로 잰 round2는 SUCCESS 227 / GENUINE_NO_ANSWER 71 / FALSE_DECLINE 2 / FALSE_CONFIDENCE 0(75.7%), round3는 SUCCESS 240 / GENUINE_NO_ANSWER 56 / FALSE_DECLINE 4 / FALSE_CONFIDENCE 0(80.0%)이다.

검색 품질(질문 104건, 출처를 가린 다중 정답 판정·엄격 기준): 벡터 검색 Recall@40 93.3%, 최종 top-5 적중은 벡터만 84.6%에서 리랭킹 후 92.3%로 올라간다(관대 기준 96.2%→98.1%). 정답 판정은 Claude 1인이며 사람의 독립 검증은 아직 없다([docs/05 §5.5](docs/05-evaluation.md#55-검색-품질-오프라인-지표)).

→ 평가 설계와 전체 지표: [docs/05-evaluation.md](docs/05-evaluation.md)

## 시스템 구조

```
브라우저 ──► FastAPI ──► rag_service (4함수 계약)
                          ├─ 에러코드 RDB 정확 매칭        (LLM 0회)
                          ├─ 모델별 기능 표 '없음' 확정
                          └─ 벡터검색 top40 → listwise 리랭킹 top5 → 답변 생성
                                     │
                          Postgres+pgvector (Supabase) ⇄ SQLite+Chroma (로컬)
```

`DATABASE_URL` 하나로 로컬 모드와 팀 공유 DB 모드를 전환한다.

→ 자세한 내용: [docs/03-architecture.md](docs/03-architecture.md) · [docs/04-rag-pipeline.md](docs/04-rag-pipeline.md)

## 빠른 시작

```bash
git clone https://github.com/Broogun/ask-my-appliance.git
cd ask-my-appliance
pip install -r requirements.txt

cp .env.example .env          # OPENAI_API_KEY 입력 (+ 공유 DB를 쓰면 DATABASE_URL 등)
python -m uvicorn web.main:app --port 8000
# → http://localhost:8000   로그인 admin1~admin7
```

PDF·벡터 인덱스 준비, 공유 DB 연결, 원격 접속 설정은 [docs/06-setup.md](docs/06-setup.md) 참고.

## 저장소 구조

```
web/              웹 앱 (FastAPI + 정적 프런트) · RAG 연결 지점 · 그림 매칭
scripts/          에러코드 사진 크롤러 · 공유 DB 이전
supabase/         공유 DB 스키마 + 설정 가이드
docs/             프로젝트 문서 (문제 정의 → 평가)
experiments/      실험 기록 · 공용 평가 하니스 · 결과 JSON
data/             설명서 PDF · SQLite (Git 제외)
```

## 문서

| 문서 | 내용 |
|---|---|
| [01-problem.md](docs/01-problem.md) | 문제 정의, 설계 목표와 우선순위, 범위 |
| [02-data.md](docs/02-data.md) | 설명서 60개·에러코드·기능 표 구성과 수집 |
| [03-architecture.md](docs/03-architecture.md) | 시스템 구조, 모듈, 저장소 이중화, 설계 결정 |
| [04-rag-pipeline.md](docs/04-rag-pipeline.md) | 청킹·검색·리랭킹·생성·안전경고·그림 매칭 |
| [05-evaluation.md](docs/05-evaluation.md) | 평가 지표 설계와 결과, 한계 |
| [06-setup.md](docs/06-setup.md) | 설치·실행·트러블슈팅 |
| [07-lessons-learned.md](docs/07-lessons-learned.md) | 발견하고 고친 문제들 (원인·시도·결과) |
| [web/README.md](web/README.md) | 웹 앱 내부 구조와 API |
| [supabase/README.md](supabase/README.md) | 공유 DB 생성·이전·보안 |
| [experiments/README.md](experiments/README.md) | 실험 참여 방법과 평가 하니스 |

## 기술 스택

FastAPI · SQLAlchemy · ChromaDB / pgvector(Supabase) · sentence-transformers(`BGE-m3-ko`) ·
OpenAI gpt-4o-mini · PyMuPDF · RAGAS

## 팀

4인 프로젝트. 각자 독립적으로 RAG 파이프라인을 구현해 공통 질문셋으로 비교한 뒤, 가장 성능이 좋은
검색 엔진과 관계형 DB 설계를 통합해 하나의 서비스로 만들었다. 비교 과정과 실험 기록은
[`experiments/`](experiments/)에 남아 있다.
