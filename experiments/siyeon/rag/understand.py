"""질문 이해 — 검색 앞에서 LLM(gpt-4o-mini) 1회로 두 가지만 판단한다.

    und = understand("자동청소 켜는 법", appliance)
    und.feature        # 질문이 존재·사용법을 묻는 기능의 정식 이름 — 그 브랜드·제품군 기능 표(model_features)에 있는 이름만 (enum). 아니면 None
    und.other_product  # 등록 가전이 아닌 다른 제품(전자레인지 …)을 묻는 것이면 그 이름, 아니면 None

둘 다 와이어프레임의 시연 동작("이 모델엔 클린봇 기능이 없습니다", 미등록 제품 안내)에 쓰인다. 기능 표가 없는 브랜드·제품군(삼성 전부,
LG 세탁기·냉장고)은 feature가 항상 None — 그 경우 "없는 기능"은 관련도 컷이 "설명서에 없음"으로 처리한다.
후속 질문 재작성은 하지 않는다(2026-09-18 제거: 새 증상을 직전 질문의 후속으로 오판해 합쳐 버림). 멀티턴은 답변 LLM에 최근 대화를 그대로 넣는다.

안전장치: structured output(enum) / feature_evidence(근거 단어)가 원문에 없으면 feature 무시 / other_product도 원문에 있어야 인정 /
호출 실패·타임아웃 4초 → 판단 없음(답은 항상 나온다) / 결과 캐시(understand_cache.json → 채점 재현, 비용 0).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = ROOT / "experiments" / "siyeon" / "understand_cache.json"
DB_PATH = ROOT / "data" / "appliance.sqlite"
MODEL = "gpt-4o-mini"


@dataclass
class Understanding:
    query: str
    feature: str | None = None
    other_product: str | None = None
    source: str = "llm"            # llm | cache | passthrough(호출 실패)


_SYSTEM = """당신은 가전제품 사용설명서 챗봇의 '질문 이해' 단계입니다. 사용자의 질문을 읽고 JSON만 출력합니다.

1. feature: 질문이 아래 [기능 목록] 중 어떤 기능의 존재 여부나 사용법을 묻는 것이면 그 정식 이름을, 아니면 null.
   사용자는 다른 말로 부릅니다 (자동청소·필터 자동 청소 → 클린봇, 말로 켜기·음성 명령 → 음성인식, 사람 없으면 꺼짐 → 활동부재).
   목록에 없는 기능이면 null. 냉방·제습·송풍·건조·세탁 같은 기본 운전 모드, 필터 청소 같은 일반 관리, 증상·고장 질문은 기능이 아니므로 null.
   확실하지 않으면 null (없는 기능을 있다고/있는 기능을 없다고 하는 것보다 null이 안전). feature_evidence에는 판단 근거가 된 질문 속 단어를 원문 그대로 적습니다.
2. other_product: 질문이 등록 가전이 아닌 다른 제품(전자레인지, 식기세척기, TV …)에 대한 것이면 질문에 나온 그 제품 이름을 원문 그대로, 아니면 null.
   등록 가전을 같은 제품군의 다른 이름으로 부른 경우(건조기 사용자의 "세탁기", 김치냉장고 사용자의 "냉장고")는 null입니다.
출력은 JSON 하나만."""


def feature_vocab(category: str, brand: str, db_path: Path = DB_PATH) -> list[str]:
    """그 브랜드·제품군 기능 표의 기능 이름 → feature enum. 표가 없으면 빈 목록. (브랜드 필수 — LG 표 이름을 삼성 질문에 주면 LLM이 억지로 고른다)"""
    if not db_path.exists():
        return []
    con = sqlite3.connect(db_path)
    try:
        return [f for (f,) in con.execute("SELECT DISTINCT feature FROM model_features WHERE category=? AND brand=? ORDER BY feature",
                                          (category, brand))]
    finally:
        con.close()


def _schema(features: list[str]) -> dict:
    feat = {"anyOf": [{"type": "string", "enum": features}, {"type": "null"}]} if features else {"type": "null"}
    props = {"feature": feat, "feature_evidence": {"type": ["string", "null"]}, "other_product": {"type": ["string", "null"]}}
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


_cache: dict | None = None


def understand(query: str, appliance: dict | None, use_cache: bool = True, timeout: float = 4.0) -> Understanding:
    """appliance: {brand, category, product_type, model, …}. 가전이 없으면 판단할 게 없어 그대로 통과."""
    global _cache
    query = query.strip()
    if not appliance or not os.getenv("OPENAI_API_KEY"):
        return Understanding(query=query, source="passthrough")
    features = feature_vocab(appliance.get("category", ""), appliance.get("brand", ""))
    label = " ".join(x for x in ({"lg": "LG", "samsung": "삼성"}.get(appliance.get("brand", ""), ""),
                                 appliance.get("product_type") or "", appliance.get("model") or "") if x)
    key = hashlib.md5(json.dumps([query, label, features, "v4"], ensure_ascii=False).encode()).hexdigest()
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    if use_cache and key in _cache:
        return _validate(_cache[key], query, features, "cache")
    user = f"[등록 가전] {label}\n[기능 목록] {', '.join(features) if features else '(없음 — feature는 null)'}\n[질문] {query}"
    try:
        from openai import OpenAI
        resp = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=timeout).chat.completions.create(
            model=MODEL, temperature=0, max_tokens=120,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": "understanding", "strict": True, "schema": _schema(features)}})
        raw = json.loads(resp.choices[0].message.content)
    except Exception as e:      # 타임아웃·한도·스키마 오류 → 판단 없이 진행
        return Understanding(query=query, source=f"passthrough:{type(e).__name__}")
    if use_cache:
        _cache[key] = raw
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return _validate(raw, query, features, "llm")


def _validate(raw: dict, query: str, features: list[str], source: str) -> Understanding:
    """LLM 출력을 코드가 좁힌다 — 근거 단어·제품명이 원문에 실제로 있어야 인정."""
    qn = _norm(query)
    feature, ev = raw.get("feature") or None, raw.get("feature_evidence") or ""
    if feature and (feature not in features or not ev or _norm(ev) not in qn):
        feature = None
    other = raw.get("other_product") or None
    if other and _norm(other) not in qn:
        other = None
    return Understanding(query=query, feature=feature, other_product=other, source=source)
