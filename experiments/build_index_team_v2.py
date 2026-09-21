"""팀 4명 RAG 비교(RAG 4종 구성 비교 — 발표용.md) 기반 통합 청킹 재구축.

4명 중 3명(영훈/지민/시연)이 독립적으로 "오버랩 없음 + 구조 정보(제목/경로) 반복"을
선택했고, 나만 700자/15% 오버랩 슬라이딩 윈도우를 썼다. 오늘 실측(NOTES
§27)에서도 오버랩 슬라이싱이 같은 섹션 조각이 최종 답변 근거 3개 중 78.3%
확률로 중복 포함되는 원인이 될 수 있다는 상관관계를 확인했다 - 3명의 선택과
방향이 일치한다.

이 스크립트는 기존 청킹 파이프라인(TOC 기반, `rag_tutorial.ipynb` 검증됨)은
그대로 두고, **마지막 슬라이싱 단계만** 문자 수 고정 오버랩 → 문단(줄바꿈)
경계 기준 무오버랩 슬라이싱으로 바꾼다. 나머지(리스트와이즈 리랭킹, 3단계
확신 게이트, RDB 정확매칭)는 exp02가 이미 팀 내에서 가장 많이 검증된 방식
(900문항)이라 그대로 유지한다.

⚠ 범위 제한(정직하게 명시): TOC 없는 5개 PDF(RF_S825AW35, RF_S835S31,
RF_RP20C3111S9, RF_SRS705IC, WM_WF21T6500KW)의 원본 폴백 파싱 코드는 저장소
정리 과정에서 삭제되어 재현 불가 - 이 5개는 기존 컬렉션의 청크를 그대로
복사해서 옮긴다(재청킹 안 함). "고장 신고 전 확인하기" 표 전용 파서(증상 단위
청크화)도 이 노트북엔 없어서(원본 배치 스크립트에만 있었던 것으로 추정) 이번
재구축엔 포함하지 않았다 - TOC+문제해결 키워드 강제 세분화로 대체됨.

사용법: python experiments/build_index_team_v2.py
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import chromadb
import pymupdf
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
OLD_CHROMA_PATH = ROOT / "chroma_db"
NEW_CHROMA_PATH = ROOT / "chroma_db_team_v2"

MAX_CHUNK_CHARS = 700  # exp02와 동일 - 이 값 자체는 팀 비교표에서 큰 이견 없었음
TROUBLESHOOTING_KEYWORDS = ["문제 해결", "고장 진단", "고장 신고", "고장신고", "에러 메시지", "트러블", "고장", "서비스를 요청하기 전에"]
MAX_SECTION_SPAN_PAGES = 20

NO_TOC_MODELS = {"S825AW35", "S835S31", "RP20C3111S9", "SRS705IC", "WF21T6500KW"}

PDF_SOURCES = [
    ("data/lg/aircon", "LG", "에어컨"),
    ("data/lg/fridge", "LG", "냉장고"),
    ("data/lg/washer", "LG", "세탁기"),
    ("data/samsung/aircon", "삼성", "에어컨"),
    ("data/samsung/fridge", "삼성", "냉장고"),
    ("data/samsung/washer", "삼성", "세탁기"),
]


# ════════════════════════════════════════════════════════════════════════════
# TOC 파싱 (rag_tutorial.ipynb 검증된 로직 그대로 - build_toc_tree ~ extract_sections_v3)
# ════════════════════════════════════════════════════════════════════════════

def build_toc_tree(toc):
    root = []
    stack = [(0, root)]
    for lvl, title, page in toc:
        node = {"level": lvl, "title": title, "page": page, "children": []}
        while len(stack) > 1 and stack[-1][0] >= lvl:
            stack.pop()
        stack[-1][1].append(node)
        stack.append((lvl, node["children"]))
    return root


def page_text(doc, start_page_1based, end_page_1based_exclusive):
    start_idx = max(start_page_1based - 1, 0)
    end_idx = min(max(end_page_1based_exclusive - 1, start_idx + 1), len(doc))
    return "\n".join(doc[p].get_text() for p in range(start_idx, end_idx)).strip()


def _subtree_has_keyword(node):
    if any(kw in node["title"] for kw in TROUBLESHOOTING_KEYWORDS):
        return True
    return any(_subtree_has_keyword(c) for c in node["children"])


def select_nodes_with_ancestors_v3(nodes, max_depth, ancestors=()):
    result = []
    for node in nodes:
        child_ancestors = ancestors + (node["title"],)
        force_expand = _subtree_has_keyword(node)
        if node["children"] and (node["level"] < max_depth or force_expand):
            result.extend(select_nodes_with_ancestors_v3(node["children"], max_depth, child_ancestors))
        else:
            result.append({**node, "_ancestors": ancestors})
    return result


def disambiguate_titles(sections):
    depth = 1
    while True:
        counts = Counter(s["heading"] for s in sections)
        if all(c == 1 for c in counts.values()):
            return
        changed = False
        for s in sections:
            if counts[s["heading"]] > 1:
                ancestors = s.get("_ancestors", ())
                if len(ancestors) < depth:
                    continue
                prefix = " > ".join(ancestors[-depth:])
                base = s.get("_base_heading", s["heading"])
                s["_base_heading"] = base
                s["heading"] = f"[{prefix}] {base}"
                changed = True
        depth += 1
        if not changed or depth > 6:
            return


def extract_sections_v3(pdf_path: str, max_depth: int = 2):
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    if not toc:
        return None
    tree = build_toc_tree(toc)
    boundaries = select_nodes_with_ancestors_v3(tree, max_depth=max_depth)

    sections = []
    for i, node in enumerate(boundaries):
        next_page = boundaries[i + 1]["page"] if i + 1 < len(boundaries) else len(doc) + 1
        if next_page <= node["page"]:
            next_page = node["page"] + 1
        if next_page - node["page"] > MAX_SECTION_SPAN_PAGES:
            next_page = node["page"] + 1
        text = page_text(doc, node["page"], next_page)
        if text:
            sections.append({
                "heading": node["title"], "text": text, "page": node["page"],
                "_ancestors": node["_ancestors"],
            })

    disambiguate_titles(sections)
    for s in sections:
        s.pop("_ancestors", None)
        s.pop("_base_heading", None)
    return sections


# ════════════════════════════════════════════════════════════════════════════
# ★ 변경 지점: 오버랩 슬라이딩 윈도우 → 문단(줄바꿈) 경계 무오버랩 슬라이싱
# ════════════════════════════════════════════════════════════════════════════

def sections_to_chunks_no_overlap(sections: list[dict], product_model: str) -> list[dict]:
    """기존(exp02): 700자씩 15%(105자) 겹치게 문자 위치 그대로 자름 - 문장 중간이
    잘리는 걸 오버랩으로 땜빵했지만, 그 결과 인접 조각끼리 내용이 겹쳐서 검색
    후보 슬롯을 나눠 쓰는 문제가 있었다(NOTES §27, 최종 근거 3개 중 78.3%가
    같은 섹션 중복).

    변경(팀 3명 방식과 동일한 방향): 오버랩을 아예 없애는 대신, 자르는 지점을
    임의 문자 위치가 아니라 **줄바꿈(문단) 경계**로 옮겼다 - 문장이 중간에서
    끊기는 문제를 오버랩 없이도 완화한다. 제목 반복(`[섹션명] 본문`)은 기존과
    동일하게 유지 - 이게 팀 3명이 쓰는 "구조 정보 반복"에 해당하는 부분이라
    이미 있던 설계를 살렸다.
    """
    chunks = []
    for i, sec in enumerate(sections):
        lines = sec["text"].split("\n")
        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for line in lines:
            line_len = len(line) + 1  # 줄바꿈 복원 비용 포함
            if buf and buf_len + line_len > MAX_CHUNK_CHARS:
                pieces.append("\n".join(buf))
                buf, buf_len = [], 0
            buf.append(line)
            buf_len += line_len
        if buf:
            pieces.append("\n".join(buf))
        if not pieces:
            pieces = [sec["text"]]

        for j, piece in enumerate(pieces):
            chunks.append({
                "id": f"{product_model}_{i}_{j}",
                "text": f"[{sec['heading']}] {piece}",
                "metadata": {"section_title": sec["heading"], "page": sec["page"]},
            })
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# 색인 빌드
# ════════════════════════════════════════════════════════════════════════════

def product_model_from_path(pdf_path: Path) -> str:
    stem = pdf_path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"임베딩 device: {device}", flush=True)
    embed_model = SentenceTransformer("dragonkue/BGE-m3-ko", device=device)

    new_client = chromadb.PersistentClient(path=str(NEW_CHROMA_PATH))
    try:
        new_client.delete_collection("appliance_manuals")
    except Exception:
        pass
    new_col = new_client.create_collection("appliance_manuals")

    old_client = chromadb.PersistentClient(path=str(OLD_CHROMA_PATH))
    old_col = old_client.get_collection("appliance_manuals")

    t0 = time.time()
    total_chunks = 0
    reused_chunks = 0

    for folder, brand, category in PDF_SOURCES:
        pdf_dir = ROOT / folder
        if not pdf_dir.exists():
            continue
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            product_model = product_model_from_path(pdf_path)

            if product_model in NO_TOC_MODELS:
                # TOC 없는 5개 - 재청킹 안 하고 기존 컬렉션에서 그대로 복사
                old_data = old_col.get(
                    where={"product_model": product_model},
                    include=["documents", "metadatas", "embeddings"],
                )
                if old_data["ids"]:
                    new_col.upsert(
                        ids=old_data["ids"], documents=old_data["documents"],
                        embeddings=old_data["embeddings"], metadatas=old_data["metadatas"],
                    )
                    reused_chunks += len(old_data["ids"])
                    print(f"{pdf_path.name:22s} [TOC없음-기존재사용] {len(old_data['ids'])}개", flush=True)
                else:
                    print(f"{pdf_path.name:22s} [경고] 기존 컬렉션에도 없음", flush=True)
                continue

            sections = extract_sections_v3(str(pdf_path))
            if not sections:
                print(f"{pdf_path.name:22s} [건너뜀] TOC 파싱 실패", flush=True)
                continue

            chunks = sections_to_chunks_no_overlap(sections, product_model)
            embeddings = embed_model.encode([c["text"] for c in chunks], normalize_embeddings=True)
            new_col.upsert(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=embeddings.tolist(),
                metadatas=[
                    {
                        "brand": brand, "category": category, "product_model": product_model,
                        "section_title": c["metadata"]["section_title"], "page": c["metadata"]["page"],
                    }
                    for c in chunks
                ],
            )
            total_chunks += len(chunks)
            print(f"{pdf_path.name:22s} 청크 {len(chunks)}개", flush=True)

    elapsed = time.time() - t0
    print(f"\n재청킹: {total_chunks}개, 기존재사용(TOC없음): {reused_chunks}개", flush=True)
    print(f"컬렉션 전체: {new_col.count()}개, 소요 시간: {elapsed:.1f}초", flush=True)
    print(f"저장 경로: {NEW_CHROMA_PATH}", flush=True)


if __name__ == "__main__":
    main()
