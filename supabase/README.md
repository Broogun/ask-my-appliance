# 공유 DB (Supabase) 설정 가이드

팀원 모두가 **같은 계정·대화·검색 데이터**로 웹 앱을 테스트하도록, 로컬 파일(SQLite · Chroma · PDF)을 Supabase로 옮긴다.
서버(FastAPI)는 여전히 각자 로컬에서 실행한다 — 임베딩 모델(BGE-m3-ko)과 OpenAI 키가 서버에 필요해서 Supabase가 대신할 수 없다.

| 옮기는 것 | 원본 | Supabase |
|---|---|---|
| 앱 데이터 (계정·등록 가전·대화·메시지) | `data/app.sqlite` | Postgres 테이블 (서버가 시작할 때 자동 생성) |
| RDB (매뉴얼 60·에러코드·기능 표) | `data/appliance.sqlite` | Postgres 테이블 8개 |
| 벡터 (6,105청크 × 1024차원) | `chroma_db/` | Postgres `rag_chunks` + pgvector |
| PDF 원본 60개 (537MB) | `data/lg`, `data/samsung` | Storage 비공개 버킷 `manuals` → 서버가 필요할 때 받아 로컬 캐시 |

`DATABASE_URL` 을 설정하면 공유 DB 모드, 지우면 지금처럼 로컬 파일 모드다 (코드 변경 없이 전환).

용량은 무료 플랜 안이다: DB 약 34MB(한도 500MB) · PDF 537MB(한도 1GB) · 가장 큰 PDF 39MB(파일당 한도 50MB).

---

## 1. 프로젝트 만들기 (한 명만, 5분)

1. https://supabase.com 가입 → **New project**
2. 이름 `ask-my-appliance`, **Region: Northeast Asia (Seoul)**, Plan: Free
3. **Database Password** 를 강하게 만들고 비밀번호 관리자에 저장 (이후 대시보드에서 다시 볼 수 없다)
4. 프로젝트가 만들어질 때까지 1~2분 대기

## 2. 연결 정보 3가지 모으기

| 값 | 위치 | 용도 |
|---|---|---|
| `DATABASE_URL` | 상단 **Connect** 버튼 → **Session pooler** 문자열 | 서버·마이그레이션의 DB 접속 |
| `SUPABASE_URL` | Project Settings → API → Project URL | PDF Storage |
| `SUPABASE_SERVICE_KEY` | Project Settings → API Keys → **service_role** (secret) | PDF Storage |

- **반드시 Session pooler 문자열을 쓴다.** `Direct connection` 은 IPv6 전용이라 IPv4 망(대부분의 집·회사·학교)에서 접속이 멈춘다.
- 문자열의 `[YOUR-PASSWORD]` 를 1단계의 비밀번호로 바꾼다. 비밀번호에 `@ : / # %` 같은 문자가 있으면 URL 인코딩해야 하니, 처음부터 영문·숫자만으로 만드는 게 편하다.
- 형태: `postgresql://postgres.<project-ref>:<비밀번호>@aws-0-<region>.pooler.supabase.com:5432/postgres`

## 3. `.env` 에 넣기

프로젝트 루트의 `.env` (git에 안 올라간다) 에 추가:

```
DATABASE_URL=postgresql://postgres.xxxx:비밀번호@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

> **연결 문자열·키를 채팅, 이슈, 커밋, 스크린샷에 붙이지 않는다.** `service_role` 키는 모든 보안 규칙(RLS)을 무시하는 전권 키다.

## 4. 데이터 옮기기 (한 명만, 처음 한 번)

원본 로컬 데이터(`data/*.sqlite`, `chroma_db/`, PDF)를 가진 사람이 실행한다.

```bash
pip install -r requirements.txt
python scripts/migrate_to_postgres.py --all
```

스키마 생성 → RDB → 벡터 → 앱 데이터 → PDF 업로드 → RLS 활성화 순서로 진행된다 (전체 약 5~10분, 대부분 PDF 업로드).
여러 번 실행해도 안전하다: RDB·벡터는 비우고 다시 채우고, **앱 데이터는 대상에 이미 사용자가 있으면 건너뛰며**(공유 DB를 덮어쓰지 않는다), 이미 올라간 PDF도 건너뛴다.

단계별로 나눠 실행하려면: `--schema --rdb --vectors` / `--app` / `--storage` / `--rls`.

확인 (대시보드 SQL Editor):

```sql
select count(*) from rag_chunks;   -- 6105
select count(*) from manuals;      -- 60
select count(*) from users;        -- 7 이상
```

## 5. 팀원이 쓰는 법

각자 로컬에서:

1. 저장소를 받고 `pip install -r requirements.txt`
2. `.env` 에 위 3개 값 + 각자의 `OPENAI_API_KEY` (3개 값은 비밀번호 관리자나 DM으로 받는다)
3. `uvicorn web.main:app --port 8000` → http://localhost:8000

이제 **필요 없는 것**: `chroma_db/`, `data/*.sqlite`, PDF 폴더 — PDF는 처음 볼 때 Storage에서 자동으로 받아 `data/` 에 캐시된다.
**여전히 필요한 것**: OpenAI 키, 임베딩 모델(첫 실행 때 자동 다운로드 ~2GB, GPU가 없으면 느림).

계정은 기존 그대로 `admin1`~`admin7` / `1234`. 같은 DB를 보므로 누가 만든 대화든 같은 계정으로 로그인하면 보인다.

## 6. 데이터가 바뀌면

| 바뀐 것 | 다시 실행 |
|---|---|
| 청킹·임베딩을 바꿨거나 문서를 추가함 | `python scripts/migrate_to_postgres.py --vectors` |
| 에러코드·기능 표를 고침 (`build_db.py` 재실행 후) | `--rdb` |
| 새 PDF 추가 | `--storage` |

앱 데이터(대화·계정)는 이제 Supabase가 원본이다. 로컬 `data/app.sqlite` 는 더 이상 갱신되지 않는다.

## 7. 보안

- 마이그레이션이 **모든 테이블에 RLS를 켠다.** Supabase는 `public` 스키마 테이블을 API로 노출하므로 RLS가 없으면 `anon` 키만 있어도 `users.password_hash` 를 읽을 수 있다. 정책을 만들지 않았으니 API로는 아무것도 못 읽고, 서버는 DB에 직접 접속해서 RLS를 우회한다. 대시보드 Database → Tables 에서 각 테이블에 `RLS enabled` 가 보이는지 확인한다.
- PDF 버킷은 **비공개**다. 제조사 저작물이라 공개 URL로 열지 않는다.
- `service_role` 키와 DB 비밀번호는 팀 4명 안에서만 돌려 쓴다. 유출됐다면 대시보드에서 즉시 재발급/재설정한다.
- 시드 계정(`admin1`~`7` / `1234`)은 테스트용이다. 서버를 인터넷에 공개하면 반드시 바꾼다.

## 8. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 서버 시작이나 첫 질문에서 30초 멈췄다가 `PoolTimeout` | Direct connection(IPv6) 주소를 썼다 → **Session pooler** 주소로 교체 |
| `password authentication failed` | `DATABASE_URL` 의 비밀번호 오타/특수문자 → 대시보드 Database → Settings 에서 재설정 후 영문·숫자로 |
| `extension "vector" is not available` | 대시보드 Database → Extensions 에서 `vector` 를 켜고 재실행 |
| PDF 뷰어·그림이 안 나옴 | `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` 확인, `--storage` 로 업로드가 끝났는지 확인 (실패해도 앱은 죽지 않고 그 PDF만 안 보인다) |
| 검색 결과가 로컬과 다름 | `select count(*) from rag_chunks` 가 6105 인지 확인, 아니면 `--vectors` 재실행 |
| 공유 DB를 쓰기 싫다 | `.env` 에서 `DATABASE_URL` 줄을 지우면 로컬 SQLite/Chroma 모드로 돌아간다 |

## 9. 검증한 것 (로컬 Postgres+pgvector로, 실제 Supabase 이전에)

- 벡터 검색: 무작위 질문 80건에서 상위 40개 청크의 **순서가 Chroma와 100% 일치**, 거리 최대 오차 6e-7 (Chroma 기본 거리인 제곱 L2와 같은 정의). `where` 필터(`$and` 포함)·`get(ids)` 순서(Chroma는 요청 순서가 아니라 내부 순번 순으로 돌려준다)도 일치.
- RDB: 매뉴얼 조회 60/60, 에러코드 조회 72/72, 기능 유무(SQLite `GLOB` → Postgres `LIKE`) 132/132, 카탈로그 60건, 기능 목록 순서까지 동일.
- 앱 전체: 로그인, 카탈로그, 이전된 대화 열기(그림 계산 포함), 새 질문(에러코드 경로·벡터 경로), 메시지 저장과 시퀀스 이어짐.
- Storage: PDF 업로드(중복 건너뜀)·다운로드 캐시·잘못된 키·경로 조작 입력을 가짜 Storage 서버로 검증. **실제 Supabase Storage와의 연동은 프로젝트 생성 후 처음 확인한다** (`--storage` 실행 로그와 앱에서 PDF 열기).
