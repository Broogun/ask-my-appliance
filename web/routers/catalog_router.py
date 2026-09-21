"""제품 찾기 — data/appliance.sqlite 의 manuals 테이블(60개)에서 카테고리·제조사·모델명으로 검색.
+ 설명서 PDF 원본 서빙 (/api/manuals/{doc_id}/pdf — 출처 카드에서 해당 페이지로 이동)."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..auth import current_user
from ..rag_service import SUPPORT, manual_pdf_path

router = APIRouter(prefix="/api/catalog", tags=["catalog"])
MANUAL_DB = Path(__file__).resolve().parents[2] / "data" / "appliance.sqlite"
BRAND_KO = {"lg": "LG", "samsung": "삼성"}


def _rows(sql, params=()):
    con = sqlite3.connect(MANUAL_DB); con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


_MODELS_SQL = """SELECT m.manual_id, m.brand, m.category, m.product_type,
                        (SELECT mm.model_pattern FROM manual_models mm
                          WHERE mm.manual_id = m.manual_id AND mm.source = 'filename'
                          ORDER BY length(mm.model_pattern) DESC LIMIT 1) AS model
                   FROM manuals m"""


@router.get("/options")
def options(_=Depends(current_user)):
    rows = _rows(_MODELS_SQL)
    return {
        "product_types": sorted({r["product_type"] for r in rows}),
        "brands": [{"id": b, "name": BRAND_KO[b]} for b in ("lg", "samsung")],
    }


@router.get("/models")
def models(product_type: str = "", brand: str = "", q: str = "", _=Depends(current_user)):
    rows = _rows(_MODELS_SQL)
    qn = q.replace("-", "").replace(" ", "").upper()
    out = []
    for r in rows:
        if product_type and r["product_type"] != product_type: continue
        if brand and r["brand"] != brand: continue
        if qn and qn not in (r["model"] or "").replace("-", "").upper(): continue
        out.append({**r, "brand_name": BRAND_KO.get(r["brand"], r["brand"])})
    return out[:30]


@router.get("/support/{brand}")
def support(brand: str, _=Depends(current_user)):
    return SUPPORT.get(brand, SUPPORT["lg"])


# ── 설명서 PDF 원본 ────────────────────────────────────────────────────────────
manuals_router = APIRouter(prefix="/api/manuals", tags=["manuals"])


@manuals_router.get("/{doc_id}/pdf")
def manual_pdf(doc_id: str, _=Depends(current_user)):
    """출처 카드의 'PDF에서 보기' — 브라우저 PDF 뷰어가 #page=N 으로 해당 페이지를 연다.
    새 탭/iframe 에서는 Authorization 헤더가 없으므로 로그인 쿠키(token)로 인증된다 (auth.current_user)."""
    path = manual_pdf_path(doc_id)
    if path is None:
        raise HTTPException(404, "설명서 PDF가 없습니다")
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline")
