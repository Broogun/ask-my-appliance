"""가전 도우미 웹 — FastAPI (API + 정적 프론트 서빙).

    python -m uvicorn web.main:app --port 8000     (프로젝트 루트에서. --reload 는 Windows에서 검색기 워커가 멈추는 문제가 있어 비권장)
    → http://localhost:8000        로그인: admin1~admin7 / 1234 (온보딩 화면을 다시 보려면 다른 계정으로)

서버 기동과 동시에 검색기(임베딩 모델·리랭커)를 백그라운드 로딩(1~2분). GET /api/health 의 ready 로 확인
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .routers import appliance_router, auth_router, catalog_router, chat_router

STATIC = Path(__file__).with_name("static")


init_db()   # 테이블 생성 + admin1~7 계정 (임포트 시점 — uvicorn/테스트 모두 동일)
app = FastAPI(title="가전 도우미")
for r in (auth_router, catalog_router, appliance_router, chat_router):
    app.include_router(r.router)
app.include_router(catalog_router.manuals_router)   # /api/manuals/{doc_id}/pdf
app.include_router(catalog_router.error_images_router)   # /api/error-images/{folder}/{name}  제조사 에러코드 페이지 사진


from .rag_service import is_ready, warm_in_background  # noqa: E402
warm_in_background()   # 검색기(임베딩 모델·리랭커) 로딩을 서버 시작과 동시에 — 첫 질문이 2분 걸리는 걸 막는다


@app.get("/api/health")
def health(warm: int = 0):
    if warm and not is_ready():
        warm_in_background()
    return {"ok": True, "ready": is_ready()}


# ── 정적 페이지 (HTML/CSS/JS) ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/onboarding")
def onboarding():
    return FileResponse(STATIC / "onboarding.html")


@app.get("/app")
def app_page():
    return FileResponse(STATIC / "app.html")
