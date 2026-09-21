"""내 제품 — 등록(여러 개 일괄)·조회·수정·삭제."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import Appliance, User, get_db, iso_kst
from .catalog_router import BRAND_KO, _MODELS_SQL, _rows

router = APIRouter(prefix="/api/appliances", tags=["appliances"])


class ApplianceIn(BaseModel):
    manual_id: str
    nickname: str = ""
    location: str = ""


class ApplianceUpdate(BaseModel):
    manual_id: str | None = None
    nickname: str | None = None
    location: str | None = None


def _dump(a: Appliance) -> dict:
    return {"id": a.id, "manual_id": a.manual_id, "brand": a.brand, "brand_name": BRAND_KO.get(a.brand, a.brand),
            "category": a.category, "product_type": a.product_type, "model": a.model,
            "nickname": a.nickname or f"{a.product_type} {a.model}", "location": a.location, "created_at": iso_kst(a.created_at)}


def _manual(manual_id: str) -> dict:
    rows = [r for r in _rows(_MODELS_SQL) if r["manual_id"] == manual_id]
    if not rows:
        raise HTTPException(404, f"등록할 수 없는 모델입니다: {manual_id}")
    return rows[0]


@router.get("")
def list_appliances(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [_dump(a) for a in db.query(Appliance).filter_by(user_id=user.id).order_by(Appliance.id).all()]


@router.post("", status_code=201)
def add_appliances(items: list[ApplianceIn], user: User = Depends(current_user), db: Session = Depends(get_db)):
    created = []
    for it in items:
        m = _manual(it.manual_id)
        a = Appliance(user_id=user.id, manual_id=m["manual_id"], brand=m["brand"], category=m["category"],
                      product_type=m["product_type"], model=m["model"], nickname=it.nickname.strip(), location=it.location.strip())
        db.add(a); created.append(a)
    db.commit()
    return [_dump(a) for a in created]


@router.patch("/{aid}")
def update_appliance(aid: int, body: ApplianceUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = db.get(Appliance, aid)
    if not a or a.user_id != user.id:
        raise HTTPException(404)
    if body.manual_id and body.manual_id != a.manual_id:
        m = _manual(body.manual_id)
        a.manual_id, a.brand, a.category, a.product_type, a.model = m["manual_id"], m["brand"], m["category"], m["product_type"], m["model"]
    if body.nickname is not None: a.nickname = body.nickname.strip()
    if body.location is not None: a.location = body.location.strip()
    db.commit()
    return _dump(a)


@router.delete("/{aid}", status_code=204)
def delete_appliance(aid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = db.get(Appliance, aid)
    if not a or a.user_id != user.id:
        raise HTTPException(404)
    for c in list(a.conversations):
        db.delete(c)
    db.delete(a); db.commit()
