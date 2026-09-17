from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import current_user, hash_password, make_token
from ..db import User, get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginIn, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=body.username).first()
    if not user or user.password_hash != hash_password(body.password):
        raise HTTPException(401, "아이디 또는 비밀번호가 맞지 않습니다")
    token = make_token(user.id)
    response.set_cookie("token", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
    return {"token": token, "user": {"id": user.id, "username": user.username}}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("token")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "username": user.username}
