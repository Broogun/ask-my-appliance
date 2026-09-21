"""가전 설명서 검색기 (LG·삼성 공용) — 등록 모델 필터 → 벡터(본문 + 예상질문) → RRF → 리랭커 → 관련도 컷. 앞에 SQLite 조회 2개.

    from siyeon.rag.chunking_lg import build_all_chunks
    from siyeon.rag.chunking_samsung import build_samsung_chunks
    from siyeon.rag.retrieval import LGAirconRetriever

    r = LGAirconRetriever(build_all_chunks() + build_samsung_chunks())
    res = r.retrieve("필터 청소는 얼마나 자주 해요?", doc_id="AC_FQ18GC1EHN")   # → RetrievalResult(hits, used, route, appliance)

경로(route)
    error_code      질문에 코드(CH05, UE …)가 있어 SQLite error_codes에서 찾음        (결정적)
    feature_absent  질문이 묻는 기능이 모델별 기능 표(model_features)에서 '없음'        (결정적, LG 에어컨 부록 표. feature는 understand.py가 준다)
    vector          등록 모델 필터 → 벡터×2 → RRF → 리랭커 8→3 → 관련도 컷 통과
    none            컷을 통과한 근거가 없음 → "설명서에 없음"

남긴 것은 전부 실측 근거가 있다: 목차 청킹 83→90%, 예상질문+리랭커 →88%, 컷 0.9("세탁기 질문에 에어컨 답" 차단), 에러코드 DB 코드 질문 100%,
약한 에러코드 조각 제거(CH237이 증상 질문을 가로채던 것). 2026-09-18 슬림: 후속 질문 재작성·멀티쿼리·listwise·BM25·태그 가산·브랜드 추정·
spec_hint·suggestions·호환 별칭을 뺐다 (실사용에서 오판을 만들거나 웹이 한 번도 안 타던 경로). 멀티턴은 답변 LLM에 최근 대화를 그대로 넣는다(web/rag_service.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking_lg import Document, common_doc_id_for, to_document

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "siyeon" / "rdb"))
from build_db import FEATURE_IMPLIED_BY, code_variants, extract_query_codes  # noqa: E402  (코드 정규화는 DB 빌드와 같은 함수)

# ── 설정 ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 384        # 512보다 빠르고 정확도도 더 좋았음
N_CANDIDATES = 8                 # 리랭커에 넘길 후보 수. 12보다 8이 더 좋았음
RRF_K = 60
CONTEXT_CUTOFF = 0.9             # 리랭커 distance(=1-관련확률) 이 값 이상이면 LLM 컨텍스트에서 제외
ERROR_CHUNK_MAX_DIST = 0.5       # 에러코드 조각은 리랭커가 강하게 지지할 때만 남긴다
QUESTIONS_PER_CHUNK = 4
QUESTIONS_BY_TAG = {"symptom": 8, "howto": 8, "diagnosis": 8, "generic": 8, "safety": 4, "error_code": 4}   # 사용자 표현이 다양한 태그는 8개

PERSIST_DIR = ROOT / "chroma_lg"                                          # 인덕스 (이름은 옛것 — LG+삼성 전부 들어 있음)
DB_PATH = ROOT / "data" / "appliance.sqlite"                               # build_db.py 산출물
QUESTIONS_CACHE = ROOT / "experiments" / "siyeon" / "chunk_questions.json"    # 청크별 예상 질문 캐시 (커밋 대상)


def _pick_device() -> str:
    """GPU가 있으면 cuda (리랭커 질문당 3초 → 0.2초). RAG_DEVICE=cpu 로 강제 가능."""
    if os.getenv("RAG_DEVICE"):
        return os.getenv("RAG_DEVICE")
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _pick_device()

_QGEN_PROMPT = """아래는 {product} 사용설명서의 한 부분이다. 이 내용으로 답할 수 있는, 실제 사용자가 챗봇에 칠 법한 질문을 {n}개 만들어라.

규칙:
- 설명서 용어를 그대로 쓰지 말고 일반 사용자의 구어체로. 특히 설명서의 전문 용어·부품 이름은 반드시 일상 표현으로 바꿔 쓴 질문을 섞어라
  (예: 거름망 → 필터, 수분이 남음 → 덜 말랐어요, 제품이 갑자기 작동 → 혼자 켜져요, 산막 → 하얀 막, 누수 → 물이 새요, 배수 → 물이 안 빠져요)
- 같은 뜻을 서로 다른 말로 묻는 변형을 포함하라 (부품 이름 없이 현상만 말하는 질문, 짧은 질문, 반말 질문)
- 증상/문제 상황이면 사용자가 겪는 현상 중심으로 (예: "혼자 켜졌어요", "물이 뚝뚝 떨어져요", "옷이 덜 말랐어요")
- 위 예시 문장을 그대로 복사하지 마라. [내용]에 실제로 있는 현상·부품에 맞게 새로 써라
- 반드시 아래 [내용]에 실제로 있는 것만 물어라. 내용에 없는 에러코드·기능·상황을 지어내지 마라
{code_rule}- 짧게, 한 문장씩. JSON 배열만 출력: ["질문1", "질문2", ...]

[섹션] {label}
[내용]
{body}"""


# ── 결과 형식 ─────────────────────────────────────────────────────────────────
ROUTES = ("error_code", "feature_absent", "vector", "none")


@dataclass
class RetrievalResult:
    hits: list                       # [(chunk, distance)] top_k. 거리 오름차순
    route: str                       # ROUTES 중 하나
    used: list = field(default_factory=list)          # 관련도 컷을 통과해 LLM 컨텍스트에 들어갈 조각
    appliance: dict | None = None                     # 등록 가전 {brand, category, product_type, model, manual_id}

    @property
    def grounded(self) -> bool:
        return bool(self.used)

    def candidates(self) -> list[dict]:
        """팀 채점 형식 [{"id", "text", "distance"}]"""
        return [{"id": c["id"], "text": c["text"], "distance": round(d, 4)} for c, d in self.hits]


def apply_cutoff(hits, context_cutoff: float | None = CONTEXT_CUTOFF):
    """관련도 컷. distance ≥ cutoff(관련확률 10% 미만)는 LLM에 넘기지 않는다. SQLite 조회 결과(source=rdb)는 그 자체가 답이므로 항상 통과."""
    return [(c, d) for c, d in hits if context_cutoff is None or d < context_cutoff or c.get("source") == "rdb"]


def appliance_label(appliance: dict | None) -> str:
    """{brand: lg, product_type: 세탁기, model: FC4KC} → 'LG 세탁기 FC4KC'"""
    if not appliance:
        return ""
    brand = {"lg": "LG", "samsung": "삼성"}.get(appliance.get("brand"), appliance.get("brand", ""))
    return " ".join(x for x in (brand, appliance.get("product_type") or "", appliance.get("model") or "") if x)


def build_context(used, appliance: dict | None = None) -> str:
    """LLM 프롬프트의 [검색된 문서] — 팀 채점(make_answer_fn)용. 웹은 web/rag_service.py 가 번호 매긴 근거 형식을 따로 쓴다."""
    head = (f"[사용자 가전] {appliance_label(appliance)} — 이 모델 기준으로 답하세요. 수치는 아래 문서에 적힌 값만 그대로 쓰고(추정 금지), "
            "여러 모델 공용 설명서라 값이 여러 개면 모델별로 그대로 나열하세요. 질문한 내용이 아래 문서에 없으면 '설명서에 해당 내용이 없다'고만 답하고 "
            "일반 상식으로 채우지 마세요.\n\n" if appliance else "")
    context = "\n\n".join(f"[{c['section']} > {c['subsection']}]\n{c['body']}" for c, _ in used)
    if not context:
        return head + "(이 모델의 설명서에서 질문과 관련 있는 내용을 찾지 못했습니다. 설명서에 없다고 간단히 안내하고, 추측으로 답하지 마세요.)"
    return head + context


# ── 검색기 ───────────────────────────────────────────────────────────────────
class LGAirconRetriever:
    """이름은 에어컨 10개 시절 것 — 지금은 LG·삼성 60개 매뉴얼 전부를 다룬다."""

    def __init__(self, chunks: list[dict], persist_dir: Path = PERSIST_DIR, db_path: Path = DB_PATH,
                 questions_cache: Path = QUESTIONS_CACHE, use_reranker: bool = True, verbose: bool = True):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import chromadb

        self.chunks = chunks
        self.chunk_by_id = {c["id"]: c for c in chunks}
        self.doc_category = {c["doc_id"]: c.get("category", "") for c in chunks}          # doc_id → aircon|fridge|washer
        self.doc_brand = {c["doc_id"]: c.get("brand", "lg") for c in chunks}                 # doc_id → lg|samsung
        self.known_doc_ids = sorted(d for d in self.doc_category if not d.endswith("-COMMON"))   # 사용자가 고를 수 있는 모델
        self.log = print if verbose else (lambda *a, **k: None)
        self.use_reranker = use_reranker

        # 1) 임베딩 + 본문 컬렉션 (디스크 영속화. 청크 id가 md5라 있는 것 재사용, 없는 것만 임베딩)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE},
                                                encode_kwargs={"normalize_embeddings": True})
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.vectorstore = Chroma(client=self.client, collection_name="langchain", embedding_function=self.embeddings)
        docs_by_id = {c["id"]: to_document(c) for c in chunks}
        self._sync_collection(self.vectorstore, docs_by_id, "본문 청크")
        self._ensure_metadata(self.vectorstore, docs_by_id)

        # 2) 청크별 예상 질문 컬렉션 (오프라인 생성 → 캐시 → 임베딩). 질문 doc id = 청크 id + 질문 텍스트 해시
        self.question_cache = self._build_question_cache(questions_cache)
        self.question_store = Chroma(client=self.client, collection_name="chunk_questions", embedding_function=self.embeddings)
        qdocs = {f'{c["id"]}__q{hashlib.md5(q.encode("utf-8")).hexdigest()[:8]}':
                 Document(page_content=q, metadata={"parent_id": c["id"], "doc_id": c["doc_id"]})
                 for c in chunks for q in self.question_cache.get(c["id"], [])}
        self._sync_collection(self.question_store, qdocs, "예상 질문")

        # 3) 리랭커(지연 로딩), 4) SQLite (에러코드·모델별 기능 표·모델↔매뉴얼)
        self._reranker = None
        if not db_path.exists():
            self.log(f"[retrieval] {db_path} 없음 → build_db.py 로 생성")
            from build_db import build
            build(verbose=verbose)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._model_to_manual = {r["model_pattern"].replace("-", ""): r["manual_id"] for r in
                                 self.db.execute("SELECT model_pattern, manual_id FROM manual_models WHERE source='filename'")}

    # ── 인덕스 구축 ──────────────────────────────────────────────────────────
    def _sync_collection(self, store, docs_by_id: dict[str, Document], label: str) -> None:
        """컬렉션의 id 집합을 docs_by_id에 맞춘다: 없는 것만 추가 임베딩, 남는 것은 삭제.
        (주의: 채점이 도는 동안 다른 프로세스가 이걸 실행하면 채점 쪽이 'Error finding id'로 죽는다 — 인덕스 갱신은 채점 중 금지)"""
        existing = set(store.get(include=[])["ids"])
        wanted = set(docs_by_id)
        extra, missing = sorted(existing - wanted), sorted(wanted - existing)
        if extra:
            store.delete(ids=extra)
        if missing:
            self.log(f"[retrieval] {label} {len(missing)}개 임베딩 (재사용 {len(existing & wanted)}개) ...")
            for k in range(0, len(missing), 2000):          # Chroma 1회 upsert 상한(5,461개) 회피
                batch = missing[k:k + 2000]
                store.add_documents([docs_by_id[i] for i in batch], ids=batch)
        else:
            self.log(f"[retrieval] {label} {len(wanted)}개 디스크 인덕스 재사용")

    def _ensure_metadata(self, store, docs_by_id: dict[str, Document]) -> None:
        """예전 인덕스에 없던 메타데이터(brand/category/product_type)를 재임베딩 없이 채운다."""
        got = store.get(include=["metadatas"])
        stale = [(i, m) for i, m in zip(got["ids"], got["metadatas"]) if i in docs_by_id and "category" not in (m or {})]
        if stale:
            store._collection.update(ids=[i for i, _ in stale], metadatas=[docs_by_id[i].metadata for i, _ in stale])
            self.log(f"[retrieval] 메타데이터 보강 {len(stale)}개 (재임베딩 없음)")

    def _build_question_cache(self, cache_path: Path) -> dict[str, list[str]]:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        todo = [c for c in self.chunks if c["id"] not in cache]
        if not todo:
            return cache
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            self.log(f"[retrieval] OPENAI_API_KEY 없음 - 예상 질문 {len(todo)}개 생성 건너뜀 (캐시 {len(cache)}개만 사용)")
            return cache
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        self.log(f"[retrieval] 예상 질문 생성: {len(todo)}개 청크 ...")

        def gen(c: dict) -> list[str]:
            label = f'{c["section"]} > {c["subsection"]}' if c["subsection"] else c["section"]
            code_rule = "- 에러코드(CH04 등)를 그대로 포함한 질문 1개 이상\n" if c["tag"] == "error_code" else ""
            n = QUESTIONS_BY_TAG.get(c.get("tag", ""), QUESTIONS_PER_CHUNK)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.7,
                messages=[{"role": "user", "content": _QGEN_PROMPT.format(
                    n=n, label=label, body=c["body"][:1500], code_rule=code_rule,
                    product=("삼성 " if c.get("brand") == "samsung" else "") + (c.get("product_type") or "가전"))}],
            )
            text = resp.choices[0].message.content.strip()
            qs = json.loads(text[text.find("["): text.rfind("]") + 1])
            return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:n]

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(gen, c): c["id"] for c in todo}
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    cache[futures[fut]] = fut.result()
                except Exception:
                    pass   # 다음 실행 때 재시도
                if i % 100 == 0 or i == len(todo):
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return cache

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH, device=DEVICE)
        return self._reranker

    # ── 가전 · 필터 ─────────────────────────────────────────────────────────
    def appliance_for(self, doc_id: str) -> dict:
        """청크 doc_id(=manual_id) → 등록 가전 dict. 웹에서는 user_appliances 행이 이 역할."""
        row = self.db.execute("SELECT brand, category, product_type FROM manuals WHERE manual_id = ?", (doc_id,)).fetchone()
        if row is None:
            raise ValueError(f"manuals 테이블에 없는 매뉴얼: {doc_id}")
        return {"brand": row["brand"], "category": row["category"], "product_type": row["product_type"],
                "model": doc_id.split("_", 1)[1].upper(), "manual_id": doc_id if doc_id in self.doc_category else None}

    def appliance_from_query(self, query: str) -> dict | None:
        """질문 문장에 모델명이 있으면(팀 채점 질문 'FQ18GC1EHN 에어컨 …') 등록 가전처럼 해석한다. manual_models 표 조회 — 규칙 아님."""
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{4,}", query):
            manual_id = self._model_to_manual.get(tok.upper().replace("-", ""))
            if manual_id:
                return self.appliance_for(manual_id)
        return None

    def _scope_ids(self, doc_id: str) -> list[str]:
        """등록 모델 doc_id + 그 브랜드·카테고리의 공통 에러코드 문서 (LG 에어컨 → LG-AC-COMMON)"""
        if doc_id not in self.known_doc_ids:
            raise ValueError(f"알 수 없는 모델 {doc_id} (선택 가능 {len(self.known_doc_ids)}개)")
        common = common_doc_id_for(self.doc_category.get(doc_id, ""), self.doc_brand.get(doc_id, "lg"))
        return [doc_id] + ([common] if common else [])

    def _as_filter(self, doc_id: str | None) -> dict | None:
        return None if doc_id is None else {"doc_id": {"$in": self._scope_ids(doc_id)}}

    # ── 벡터 검색 ───────────────────────────────────────────────────────────
    def search_baseline(self, query: str, doc_id=None, top_k: int = 3) -> list[tuple[dict, float]]:
        """벡터(본문)만 — 비교용 baseline. distance = Chroma L2(정규화 벡터라 2×(1−코사인))."""
        res = self.vectorstore.similarity_search_with_score(query, k=top_k, filter=self._as_filter(doc_id))
        return [({**d.metadata, "text": d.page_content}, float(s)) for d, s in res]

    def _candidates(self, query: str, doc_id, n: int) -> list[str]:
        """벡터(본문) + 벡터(예상질문→부모 조각) → RRF(두 검색기에서 동시에 상위인 조각이 위로) → 후보 id n개."""
        filt = self._as_filter(doc_id)
        hits = self.vectorstore.similarity_search_with_score(query, k=n, filter=filt)
        ranked = [[d.metadata["id"] for d, _ in hits]]
        qhits = self.question_store.similarity_search_with_score(query, k=n * 2, filter=filt)
        seen, parents = set(), []
        for d, _ in qhits:
            pid = d.metadata["parent_id"]
            if pid not in seen and pid in self.chunk_by_id:
                seen.add(pid); parents.append(pid)
        ranked.append(parents[:n])
        score: dict[str, float] = {}
        for lst in ranked:
            for rank, cid in enumerate(lst, start=1):
                score[cid] = score.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        return [cid for cid in sorted(score, key=score.get, reverse=True) if cid in self.chunk_by_id][:n]

    def search_dense(self, query: str, doc_id=None, top_k: int = 3, n_candidates: int = N_CANDIDATES) -> list[tuple[dict, float]]:
        """벡터×2 → RRF → 리랭커 → top_k. distance = 1 − 리랭커 관련확률."""
        fused = self._candidates(query, doc_id, n_candidates)
        if not fused:
            return []
        if not self.use_reranker:
            return [(self.chunk_by_id[cid], 1.0 + i * 0.01) for i, cid in enumerate(fused[:top_k])]
        probs = np.asarray(self.reranker.predict([(query, self.chunk_by_id[cid]["text"]) for cid in fused]))
        order = np.argsort(-probs)[:top_k]
        return [({**self.chunk_by_id[fused[i]], "rerank_score": float(probs[i])}, float(1.0 - probs[i])) for i in order]

    # ── SQLite 조회 ─────────────────────────────────────────────────────────
    def lookup_error_codes(self, brand: str, category: str, query: str) -> list[dict]:
        """질문 속 코드(CH05, ch 05, 0E/OE …) → error_codes 표. 브랜드·제품군으로 범위를 좁힌다 (FF가 세탁기/냉장고에 다 있음)."""
        variants = sorted({v for tok in extract_query_codes(query) for v in code_variants(tok)})
        if not variants:
            return []
        rows = self.db.execute(
            f"SELECT e.entry_id, e.brand, e.category, e.title, e.summary, e.content, e.chunk_id, c.code_display, c.kind "
            f"FROM error_codes c JOIN error_entries e USING (entry_id) "
            f"WHERE c.brand = ? AND c.category = ? AND c.code_norm IN ({','.join('?' * len(variants))}) "
            f"ORDER BY c.kind = 'primary' DESC, e.entry_id", (brand, category, *variants)).fetchall()
        seen, out = set(), []
        for r in rows:
            if r["entry_id"] not in seen:
                seen.add(r["entry_id"]); out.append(dict(r))
        return out

    def feature_available(self, appliance: dict, feature: str) -> bool | None:
        """모델별 기능 표(model_features, LG 에어컨 부록) 조회 — True 있음 / False 없음 / None 표에 없음(모름 → 검색으로)."""
        target = FEATURE_IMPLIED_BY.get(feature, feature)
        row = self.db.execute(
            "SELECT has FROM model_features WHERE brand=? AND category=? AND feature=? AND ? GLOB replace(model_pattern,'*','?')",
            (appliance["brand"], appliance["category"], target, appliance.get("model") or "")).fetchone()
        return None if row is None else bool(row["has"])

    def _error_entry_as_chunk(self, e: dict) -> dict:
        c = self.chunk_by_id.get(e["chunk_id"])
        if c:
            return {**c, "source": "rdb"}
        label = f"에러코드 ({e.get('brand', '')} {e.get('category', '')})".strip()
        return {"id": e["chunk_id"] or f"rdb_error_{e['entry_id']}", "doc_id": "RDB", "section": label,
                "subsection": e["title"], "body": e["content"], "tag": "error_code",
                "text": f"{label} > {e['title']}\n{e['content']}", "source": "rdb"}

    # ── 라우터 ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, doc_id: str | None = None, top_k: int = 3, appliance: dict | None = None,
                 n_candidates: int = N_CANDIDATES, context_cutoff: float | None = CONTEXT_CUTOFF,
                 feature: str | None = None) -> RetrievalResult:
        """① 에러코드 SQLite → ② 기능 표 → ③ 벡터 검색 → ④ 관련도 컷. feature는 understand.py가 정규화한 기능 이름(없으면 None)."""
        if appliance is None:
            appliance = self.appliance_for(doc_id) if doc_id else self.appliance_from_query(query)
        manual_id = (appliance or {}).get("manual_id")
        hits, route = self._route(query, appliance, manual_id, top_k, n_candidates, feature)
        used = apply_cutoff(hits, context_cutoff)
        if not used and route == "vector":
            route = "none"
        return RetrievalResult(hits=hits, route=route, used=used, appliance=appliance)

    def _route(self, query: str, appliance, manual_id, top_k, n_candidates, feature):
        if appliance:
            # ① 에러코드 — 코드는 문자 그대로 매칭이라 정규식(extract_query_codes)이 맞는 도구
            entries = self.lookup_error_codes(appliance["brand"], appliance["category"], query)
            if entries:
                hits = [(self._error_entry_as_chunk(e), 0.0) for e in entries[:top_k]]
                if len(hits) < top_k and manual_id:                      # 남는 자리는 매뉴얼의 관련 증상으로 보강
                    got = {h[0]["id"] for h in hits}
                    extra = [(c, d) for c, d in self.search_dense(query, manual_id, top_k + 3, n_candidates)
                             if c["tag"] != "error_code" and c["id"] not in got]
                    hits += extra[: top_k - len(hits)]
                return hits, "error_code"
            # ② 기능 유무 — 표가 '없음'이라고 할 때만 검색을 막는다 (있음/표에 없음은 검색으로). 공용 설명서엔 없는 기능 사용법도 실려 있어서
            if feature and manual_id and self.feature_available(appliance, feature) is False:
                body = f"이 모델({appliance['model']})에는 '{feature}' 기능이 없습니다. (모델별 기능 표 기준)"
                return [({"id": f"feature_absent_{feature}", "doc_id": manual_id, "section": "기능 없음", "subsection": feature,
                          "body": body, "tag": "feature_absent", "text": body, "source": "rdb"}, 1.0)], "feature_absent"
        # ③ 벡터 검색 (manual_id 없으면 전체 인덕스). 에러코드 조각은 리랭커가 강하게 지지할 때만 남긴다 (순서는 리랭커 그대로)
        hits = self.search_dense(query, manual_id, top_k + 5, n_candidates)
        kept = [(c, d) for c, d in hits if c["tag"] != "error_code" or d < ERROR_CHUNK_MAX_DIST]
        return (kept or hits)[:top_k], "vector"

    # ── 팀 공용 채점 형식 ───────────────────────────────────────────────────
    def make_answer_fn(self, doc_id=None, top_k: int = 3, context_cutoff: float | None = CONTEXT_CUTOFF,
                       system_prompt: str | None = None, cache_path: Path | None = None, use_understanding: bool = False):
        """my_answer(query) -> {"answer", "candidates"} (pipeline.evaluate.run_and_save 형식).
        use_understanding: 검색 전에 understand()로 기능명을 정규화한다. cache_path: 질문별 {검색 결과, 답변} 캐시 — 429 등으로 중단돼도 재실행 때 성공분은 LLM을 다시 부르지 않는다."""
        from openai import OpenAI
        if system_prompt is None:
            sys.path.insert(0, str(ROOT))
            from pipeline.evaluate import SYSTEM_PROMPT as system_prompt   # 팀 공용, 수정 금지
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else {}

        def save_cache():
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        def my_answer(query: str) -> dict:
            entry = cache.get(query, {})
            if "hits" in entry:
                hits, route, appliance = [(c, d) for c, d in entry["hits"]], entry.get("route", "vector"), entry.get("appliance")
            else:
                appliance = self.appliance_for(doc_id) if doc_id else self.appliance_from_query(query)
                feature = None
                if use_understanding and appliance:
                    from .understand import understand
                    feature = understand(query, appliance).feature
                res = self.retrieve(query, doc_id=doc_id, top_k=top_k, appliance=appliance, context_cutoff=context_cutoff, feature=feature)
                hits, route = res.hits, res.route
                entry.update(hits=[({k: v for k, v in c.items() if k != "rerank_score"}, d) for c, d in hits], route=route, appliance=appliance)
            candidates = [{"id": c["id"], "text": c["text"], "distance": round(d, 4)} for c, d in hits]
            if entry.get("answer") and not entry["answer"].startswith("[LLM 오류]"):
                return {"answer": entry["answer"], "candidates": candidates, "route": route}
            context = build_context(apply_cutoff(hits, context_cutoff), appliance)
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": f"[검색된 문서]\n{context}\n\n[질문]\n{query}"}])
                answer = resp.choices[0].message.content
            except Exception as e:                       # 429(한도)·네트워크 등 — 검색 결과는 살리고 답변만 오류 표시
                answer = f"[LLM 오류] {type(e).__name__}: {str(e)[:120]}"
            entry["answer"] = answer
            cache[query] = entry
            save_cache()
            return {"answer": answer, "candidates": candidates, "route": route}

        return my_answer
