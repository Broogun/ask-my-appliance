"""대화 — 새 대화(제품 선택) · 메시지 전송(스트리밍) · 기록.

대화 행(conversations)은 **첫 질문을 보낼 때** 만들어진다. 제품을 고르고 "대화 시작"만 누른 건 아직 대화가 아니라서
POST /api/conversations 는 저장 없이 초안(id=null)만 돌려주고, 첫 질문은 POST /api/conversations/start 로 보낸다
(스트림 첫 줄 {"type":"conv"} 에 만들어진 대화가 실려 온다). 예전엔 "대화 시작"에서 바로 행을 만들어서 질문 없이
나간 빈 대화가 DB에 쌓였다.

메시지 응답은 줄 단위 JSON 스트림(NDJSON):
    {"type":"conv",    "conversation": {...}}             (/start 로 새 대화를 만들었을 때만, 맨 처음)
    {"type":"status",  "text": "…"}                      (검색기 로딩 중일 때만)
    {"type":"sources", "route": "...", "sources":[...], "unregistered": "전자레인지"|null}
    {"type":"token", "text": "..."}   ×N
    {"type":"done", "message_id": 12}
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import Appliance, Conversation, Message, SessionLocal, User, get_db, iso_kst, now_kst
from ..rag_service import get_retriever, is_ready, retrieve, sources_of, stream_answer, understand_query
from .appliance_router import _dump as dump_appliance

router = APIRouter(prefix="/api/conversations", tags=["chat"])


class NewConv(BaseModel):
    appliance_id: int


class MessageIn(BaseModel):
    content: str


class StartIn(BaseModel):
    appliance_id: int
    content: str


def _conv(db: Session, cid: int, user: User) -> Conversation:
    c = db.get(Conversation, cid)
    if not c or c.user_id != user.id:
        raise HTTPException(404, "대화를 찾을 수 없습니다")
    return c


def _appliance(db: Session, aid: int, user: User) -> Appliance:
    a = db.get(Appliance, aid)
    if not a or a.user_id != user.id:
        raise HTTPException(404, "등록된 제품이 아닙니다")
    return a


def _dump_conv(c: Conversation, with_messages=False) -> dict:
    d = {"id": c.id, "title": c.title or "새 대화", "appliance": dump_appliance(c.appliance),
         "created_at": iso_kst(c.created_at), "updated_at": iso_kst(c.updated_at), "message_count": len(c.messages)}
    if with_messages:
        d["messages"] = [{"id": m.id, "role": m.role, "content": m.content, "route": m.route, "sources": m.sources,
                          "created_at": iso_kst(m.created_at)} for m in c.messages]
    return d


@router.get("")
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    convs = db.query(Conversation).filter_by(user_id=user.id).order_by(Conversation.updated_at.desc()).all()
    return [_dump_conv(c) for c in convs if c.messages]


@router.post("", status_code=200)
def new_conversation(body: NewConv, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """초안 — DB에 저장하지 않는다. 프론트는 id=null 이면 첫 질문을 /start 로 보낸다."""
    a = _appliance(db, body.appliance_id, user)
    return {"id": None, "title": "새 대화", "appliance": dump_appliance(a), "created_at": "", "updated_at": "",
            "message_count": 0, "messages": []}


@router.get("/{cid}")
def get_conversation(cid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _dump_conv(_conv(db, cid, user), with_messages=True)


@router.delete("/{cid}", status_code=204)
def delete_conversation(cid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.delete(_conv(db, cid, user)); db.commit()


@router.post("/start")
def start_conversation(body: StartIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """첫 질문 — 여기서 대화 행을 만들고 바로 답변 스트리밍. 스트림 첫 줄에 만들어진 대화를 보낸다."""
    query = body.content.strip()
    if not query:
        raise HTTPException(400, "질문이 비었습니다")
    a = _appliance(db, body.appliance_id, user)
    conv = Conversation(user_id=user.id, appliance_id=a.id, title=query[:60])
    db.add(conv); db.commit()
    return _respond(db, conv, query, first_line={"type": "conv", "conversation": _dump_conv(conv, with_messages=True)})


@router.post("/{cid}/messages")
def send_message(cid: int, body: MessageIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    query = body.content.strip()
    if not query:
        raise HTTPException(400, "질문이 비었습니다")
    return _respond(db, _conv(db, cid, user), query)


def _respond(db: Session, conv: Conversation, query: str, first_line: dict | None = None) -> StreamingResponse:
    """사용자 메시지 저장 → LLM 질문 이해(기능명·다른 제품) → (다른 제품 안내 | 검색 + 답변 스트리밍) → 어시스턴트 메시지 저장."""
    appliance = conv.appliance
    history = [{"role": m.role, "content": m.content} for m in conv.messages[-8:]]      # 최근 4턴
    user_msg = Message(conversation_id=conv.id, role="user", content=query)
    if not conv.title:
        conv.title = query[:60]
    db.add(user_msg); db.commit()
    conv_id = conv.id

    def gen():
        def line(obj): return json.dumps(obj, ensure_ascii=False) + "\n"
        if first_line:
            yield line(first_line)
        # ① 질문 이해 (LLM 1회, ~1초): 기능명 정규화 · 다른 제품 감지. 실패하면 판단 없이 진행
        und = understand_query(query, appliance)
        if und.other_product:
            # 등록 가전이 아닌 다른 제품 질문 (와이어프레임 6-1) — 검색 없이 안내
            text = (f'"{und.other_product}"는 등록된 제품 목록에 없어서, 설명서를 근거로 정확하게 답변드리기 어려워요. '
                    f"제품을 등록하시면 이 대화에서 바로 이어서 안내해드릴 수 있어요.")
            yield line({"type": "sources", "route": "unregistered", "sources": [], "unregistered": und.other_product})
            yield line({"type": "token", "text": text})
            mid = _save(conv_id, text, "unregistered", [])
            yield line({"type": "done", "message_id": mid})
            return
        # ② 검색 → 출처 먼저 보내고 → 답변 스트리밍. 검색기가 아직 로딩 중이면 상태를 먼저 알린다
        if not is_ready():
            yield line({"type": "status", "text": "설명서 검색 엔진을 준비하고 있어요 (처음 한 번, 1분)…"})
            get_retriever()
            yield line({"type": "status", "text": "관련 설명서를 찾고 있어요…"})
        res = retrieve(query, appliance.manual_id, und.feature)
        sources = sources_of(res)
        yield line({"type": "sources", "route": res.route, "sources": sources, "unregistered": None})
        buf = []
        try:
            for tok in stream_answer(query, res, history):
                buf.append(tok)
                yield line({"type": "token", "text": tok})
        except Exception as e:  # API 한도·네트워크 — 근거는 이미 보냈으니 오류만 알린다
            err = f"답변 생성 중 오류가 났어요 ({type(e).__name__}). 오른쪽 출처 패널의 설명서 내용을 참고해 주세요."
            buf.append(err); yield line({"type": "token", "text": err})
        mid = _save(conv_id, "".join(buf), res.route, sources)
        yield line({"type": "done", "message_id": mid})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _save(conv_id: int, content: str, route: str, sources: list) -> int:
    with SessionLocal() as s:
        m = Message(conversation_id=conv_id, role="assistant", content=content, route=route, sources=sources)
        s.add(m)
        conv = s.get(Conversation, conv_id)
        if conv is not None:
            conv.updated_at = now_kst()   # (m.created_at 은 flush 전이라 None — 예전엔 그래서 갱신이 안 됐다)
        s.commit()
        return m.id
