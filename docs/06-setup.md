# 6. 설치 및 실행

## 6.1 사전 준비

**Python 3.12.14** 기준(`.python-version`에 명시, pyenv를 쓰면 자동으로 맞춰진다).

```bash
git clone https://github.com/Broogun/ask-my-appliance.git
cd ask-my-appliance
python --version                  # 3.12.14 확인
pip install -r requirements.txt
```

**GPU (선택, 권장).** 위 명령은 CPU 전용 torch를 설치한다. NVIDIA GPU가 있으면 CUDA 빌드로 바꾸는 게
훨씬 빠르다(검색기 리랭커 질문당 3초 → 0.3초).

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch
python -c "import torch; print(torch.cuda.is_available())"     # True면 성공, 코드가 자동 감지
```

## 6.2 환경변수

`.env.example`을 복사해 `.env`를 만든다. **`.env`는 절대 커밋하지 않는다**(`.gitignore`에 포함).

```bash
cp .env.example .env
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | 답변 생성·리랭킹·평가에 사용 (gpt-4o-mini) |
| `DATABASE_URL` | | 공유 DB 모드용 Postgres 주소. 없으면 로컬 SQLite/Chroma 모드 |
| `SUPABASE_URL` | | 공유 DB 모드에서 PDF를 Storage에서 받을 때 |
| `SUPABASE_SERVICE_KEY` | | 위와 동일 (서버 전용 비밀 키) |
| `APP_SECRET` | | 로그인 토큰 서명 키. 서버를 외부에 노출할 때 필수 |
| `ADMIN_PASSWORD` | | 계정 공통 비밀번호. 기본값 `1234`는 로컬 전용 |

## 6.3 두 가지 실행 모드

| | 로컬 모드 | 공유 DB 모드 |
|---|---|---|
| 설정 | `DATABASE_URL` 없음 | `DATABASE_URL`에 Postgres 주소 |
| 데이터 | 각자 로컬 파일 | 팀이 같은 Supabase를 봄 |
| 필요한 것 | PDF·벡터 인덱스를 직접 준비 | `.env` 3줄만 |
| 용도 | 청킹·인덱싱 실험 | 팀 공동 테스트 |

### 로컬 모드 — PDF 받기

PDF는 용량(537MB)과 저작권 때문에 Git에 없다. 공유 드라이브에서 받아 `data/` 밑에 그대로 넣는다.

**다운로드**: https://drive.google.com/file/d/1ePqOVFxh0t0qK-fjMoVuMGw4LbdHV-XF/view?usp=sharing

```
data/
  lg/{aircon,fridge,washer}/         ← LG PDF
  samsung/{aircon,fridge,washer}/    ← 삼성 PDF
  *_errors.json                      ← 에러코드 원본
```

처음 실행 시 자동으로 만들어지는 것:

- `data/appliance.sqlite` — 없으면 `experiments/siyeon/rdb/build_db.py`가 생성 (약 1분)
- 벡터 인덱스 — 청크 임베딩. **CPU 30~40분, GPU 3~5분** (1회)

### 공유 DB 모드 — Supabase 연결

`.env`에 `DATABASE_URL`·`SUPABASE_URL`·`SUPABASE_SERVICE_KEY`만 넣으면 된다. PDF·벡터 인덱스를 따로
준비할 필요가 없고, PDF는 필요할 때 Storage에서 받아 로컬에 캐시한다.

프로젝트 생성부터 데이터 이전까지는 [`supabase/README.md`](../supabase/README.md)에 정리돼 있다.

```bash
python scripts/migrate_to_postgres.py --all   # 최초 1회, 데이터를 가진 사람만
```

## 6.4 웹 앱 실행

```bash
python -m uvicorn web.main:app --port 8000
# → http://localhost:8000
```

계정은 `admin1` ~ `admin7`, 비밀번호는 `ADMIN_PASSWORD`(기본 `1234`). 온보딩 화면은 계정당 한 번만
보이므로 테스트용으로 계정을 여러 개 뒀다.

서버가 뜨면 검색기(임베딩 모델)를 백그라운드로 로딩한다 — GPU 약 50초, CPU 1~2분. 그동안에도 질문은
받고, 준비되는 대로 답한다. `GET /api/health`로 상태를 볼 수 있다.

> **`--reload`는 쓰지 않는다.** Windows에서 검색기를 든 워커가 재시작 신호에 멈춰 옛 코드로 계속
> 응답하고 포트를 물고 있는 일이 생긴다. 코드 수정 중이라면 `--reload --reload-dir web`으로 감시 범위를
> 좁히고, 이상하면 서버를 완전히 껐다 켠다.

## 6.5 팀원이 원격에서 접속하게 하기 (선택)

서버 한 대만 띄우고 팀원은 URL로 접속하는 방식. 임베딩 모델과 OpenAI 키가 서버에만 있으면 된다.

**반드시 먼저 할 것** — 인터넷에 노출되므로 기본값을 그대로 두면 안 된다.

```bash
# .env 에 강한 값으로 설정 (예시 생성 명령)
python -c "import secrets; print('APP_SECRET=' + secrets.token_hex(32))"
python -c "import secrets; print('ADMIN_PASSWORD=' + secrets.token_urlsafe(9))"
```

`ADMIN_PASSWORD`를 바꾸면 다음 서버 시작 때 기존 계정 7개의 비밀번호가 함께 갱신된다.

```bash
python -m uvicorn web.main:app --host 0.0.0.0 --port 8000    # 외부 접속 허용
ngrok http 8000                                               # 터널 (별도 터미널)
```

- ngrok 무료 터널은 재시작할 때마다 주소가 바뀐다. 계속 쓰려면 대시보드에서 고정 도메인을 예약한다.
- 팀원 첫 접속 시 ngrok 경고 화면이 한 번 뜬다(정상).
- **비용은 서버를 띄운 사람의 OpenAI 키로 청구된다.**

## 6.6 자주 막히는 지점

| 증상 | 원인 / 해결 |
|---|---|
| 첫 질문이 2분 걸림 | 임베딩 모델 로딩 중. 서버 시작 후 1~2분 기다린다 |
| 서버 시작이 30초 멈추다 `PoolTimeout` | Supabase Direct connection(IPv6) 주소 사용 → **Session pooler** 주소로 교체 |
| `.env` 읽다가 `UnicodeDecodeError` | `.env`에 UTF-8이 아닌 바이트. 한글 주석을 넣을 때 인코딩 확인 |
| 검색 결과가 로컬과 다름 | 공유 DB의 `rag_chunks`가 6,105개인지 확인, 아니면 `--vectors` 재실행 |
| PDF·그림이 안 나옴 | `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` 확인. 실패해도 앱은 죽지 않고 그 PDF만 안 보인다 |
| GPU를 못 잡음 | `torch.cuda.is_available()` 확인, CUDA 빌드 재설치 |

## 6.7 관련 문서

| 문서 | 내용 |
|---|---|
| [`web/README.md`](../web/README.md) | 웹 앱 내부 구조, API, 스트리밍 형식 |
| [`supabase/README.md`](../supabase/README.md) | 공유 DB 생성·이전·보안 |
| [`experiments/README.md`](../experiments/README.md) | 실험 참여 방법, 평가 하니스 |
