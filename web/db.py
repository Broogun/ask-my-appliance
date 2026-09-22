"""앱 DB (사용자·등록 가전·대화). SQLite로 시작 — Postgres 전환은 DATABASE_URL 만 바꾸면 된다.

    data/app.sqlite            ← 이 파일 (신규)
    data/appliance.sqlite      ← 매뉴얼·에러코드 (기존, siyeon/rdb/build_db.py 가 만듦. 읽기만)
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
def _normalize_url(url: str) -> str:
    """Supabase 가 주는 'postgres://...' / 'postgresql://...' 를 SQLAlchemy 가 psycopg3 드라이버로 열 수 있게 바꾼다."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


DATABASE_URL = _normalize_url(os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'data' / 'app.sqlite').as_posix()}"))

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:   # Postgres(Supabase): 끊긴 연결 자동 복구, pgbouncer(transaction 모드)와 호환되게 prepared statement 끔
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args={"prepare_threshold": None})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    """저장용 현재 시각 — 한국 시간, tz 없는 naive (SQLite DateTime은 tz를 안 남긴다).
    예전엔 datetime.utcnow 를 썼는데 isoformat()에 'Z'가 안 붙어 브라우저가 로컬 시간으로 읽는 바람에
    대화 기록 일시가 9시간 전으로 보였다."""
    return datetime.now(KST).replace(tzinfo=None)


def iso_kst(dt: datetime | None) -> str:
    """API 응답용 — '2026-09-17T17:09:24+09:00'. 오프셋을 붙여 보내야 JS new Date()가 어느 시간대인지 안다."""
    return dt.replace(tzinfo=KST).isoformat() if dt else ""


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_kst)


class Appliance(Base):
    """사용자가 등록한 가전. manual_id 가 siyeon.rag 검색 범위(doc_id)가 된다."""
    __tablename__ = "user_appliances"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    manual_id: Mapped[str] = mapped_column(String(50))          # 예: AC_FQ18GC1EHN
    brand: Mapped[str] = mapped_column(String(20))              # lg | samsung
    category: Mapped[str] = mapped_column(String(20))           # aircon | fridge | washer
    product_type: Mapped[str] = mapped_column(String(20))       # 에어컨 | 냉장고 | 세탁기 | 건조기 …
    model: Mapped[str] = mapped_column(String(50))
    nickname: Mapped[str] = mapped_column(String(50), default="")
    location: Mapped[str] = mapped_column(String(30), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_kst)
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="appliance")


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    appliance_id: Mapped[int] = mapped_column(ForeignKey("user_appliances.id"))
    title: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_kst)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_kst, onupdate=now_kst)
    appliance: Mapped["Appliance"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", order_by="Message.id",
                                                     cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))               # user | assistant
    content: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(20), default="")  # error_code | feature_absent | vector | none | unregistered
    sources: Mapped[list] = mapped_column(JSON, default=list)   # [{title, snippet, tag, doc_id, distance}]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_kst)
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


def init_db() -> None:
    Base.metadata.create_all(engine)
    from .auth import hash_password
    # 회원가입 없음 — 고정 계정 admin1~admin7. 비번은 ADMIN_PASSWORD(.env)로 - 기본값은 로컬 전용 "1234",
    # 서버를 팀 밖에 노출할 때(ngrok 등)는 반드시 강한 값으로 바꾼다. 바뀌면 기존 계정 비번도 같이 갱신한다
    # (공유 DB라 한 번 바꾸면 모든 팀원에게 적용됨).
    password_hash = hash_password(os.getenv("ADMIN_PASSWORD", "1234"))
    with SessionLocal() as s:
        existing = {u.username: u for u in s.query(User).all()}
        changed = False
        for n in range(1, 8):
            name = f"admin{n}"
            if name not in existing:
                s.add(User(username=name, password_hash=password_hash)); changed = True
            elif existing[name].password_hash != password_hash:
                existing[name].password_hash = password_hash; changed = True
        if changed:
            s.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
