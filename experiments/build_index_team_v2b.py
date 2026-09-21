"""v2(무오버랩+문단경계)가 기존보다 MRR/Recall이 떨어진 원인 후보 중 하나인
"문맥 반복 부재"만 분리 검증하는 가벼운 실험.

v2와 동일하게 오버랩 0% + 문단 경계 슬라이싱을 쓰되, 청크 접두사를 직속
섹션 제목 하나("[소음 문제]")가 아니라 TOC 조상 경로 전체("[에어컨 > 문제
해결 > 소음 문제]")로 바꿨다 - 지민이 쓰는 "문맥 반복"에 해당하는 부분만
추가해서, 이게 v2의 regression을 만회하는지만 본다. 나머지(크기 상한 700자,
짧은 조각 병합 없음)는 v2와 동일하게 유지 - 변수 하나만 바꿔서 원인을
분리하기 위함.

사용법: python experiments/build_index_team_v2b.py
"""

from __future__ import annotations

import time
from pathlib import Path

import chromadb
import pymupdf
import torch
from sentence_transformers import SentenceTransformer

import build_index_team_v2 as v2

ROOT = Path(__file__).resolve().parent.parent
OLD_CHROMA_PATH = ROOT / "chroma_db"
NEW_CHROMA_PATH = ROOT / "chroma_db_team_v2b"

MAX_CHUNK_CHARS = v2.MAX_CHUNK_CHARS
NO_TOC_MODELS = v2.NO_TOC_MODELS
PDF_SOURCES = v2.PDF_SOURCES


def extract_sections_with_breadcrumb(pdf_path: str, max_depth: int = 2):
    """v2.extract_sections_v3와 동일하지만 _ancestors를 지우지 않고 유지."""
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    if not toc:
        return None
    tree = v2.build_toc_tree(toc)
    boundaries = v2.select_nodes_with_ancestors_v3(tree, max_depth=max_depth)

    sections = []
    for i, node in enumerate(boundaries):
        next_page = boundaries[i + 1]["page"] if i + 1 < len(boundaries) else len(doc) + 1
        if next_page <= node["page"]:
            next_page = node["page"] + 1
        if next_page - node["page"] > v2.MAX_SECTION_SPAN_PAGES:
            next_page = node["page"] + 1
        text = v2.page_text(doc, node["page"], next_page)
        if text:
            sections.append({
                "heading": node["title"], "text": text, "page": node["page"],
                "_ancestors": node["_ancestors"],
            })

    v2.disambiguate_titles(sections)
    # _ancestors는 유지 - disambiguate가 만든 "_base_heading"(원래 제목)이 있으면
    # 그걸 쓰고, 없으면(중복 없어서 disambiguate 안 건드린 경우) heading 그대로 사용
    for s in sections:
        s["_base_heading"] = s.get("_base_heading", s["heading"])
    return sections


def sections_to_chunks_context_repeat(sections: list[dict], product_model: str) -> list[dict]:
    chunks = []
    for i, sec in enumerate(sections):
        ancestors = sec.get("_ancestors", ())
        base = sec["_base_heading"]
        breadcrumb = " > ".join([*ancestors, base]) if ancestors else base

        lines = sec["text"].split("\n")
        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for line in lines:
            line_len = len(line) + 1
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
                "text": f"[{breadcrumb}] {piece}",
                # section_title은 기존 라벨셋(product_model+section_title 매칭)과 호환되도록
                # disambiguate된 heading(= v2/기존과 동일한 값) 그대로 유지
                "metadata": {"section_title": sec["heading"], "page": sec["page"]},
            })
    return chunks


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
            product_model = v2.product_model_from_path(pdf_path)

            if product_model in NO_TOC_MODELS:
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

            sections = extract_sections_with_breadcrumb(str(pdf_path))
            if not sections:
                print(f"{pdf_path.name:22s} [건너뜀] TOC 파싱 실패", flush=True)
                continue

            chunks = sections_to_chunks_context_repeat(sections, product_model)
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
