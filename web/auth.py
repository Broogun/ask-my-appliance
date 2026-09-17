"""로그인 — 고정 계정(admin1~admin7 / 1234). 토큰은 표준 라이브러리 HMAC 서명(외부 의존성 없음)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import User, get_db

SECRET = os.getenv("APP_SECRET", "dev-secret-change-me").encode()
TOKEN_TTL = 60 * 60 * 24 * 7   # 7일


def hash_password(pw: str) -> str:
    return hashlib.sha256(("salt::" + pw).encode()).hexdigest()


def make_token(user_id: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"uid": user_id, "exp": int(time.time()) + TOKEN_TTL}).encode()).decode().rstrip("=")
    sig = hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{sig}"


def parse_token(token: str) -> int | None:
    try:
        payload, sig = token.split(".")
        if not hmac.compare_digest(hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32], sig):
            return None
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return data["uid"] if data["exp"] > time.time() else None
    except Exception:
        return None


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() or request.cookies.get("token", "")
    uid = parse_token(token) if token else None
    user = db.get(User, uid) if uid else None
    if not user:
        raise HTTPException(401, "로그인이 필요합니다")
    return user
