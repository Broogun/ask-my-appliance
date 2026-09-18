"""오버랩 보강 실험 - 기존(700자/15%오버랩) 방식이 v2/v2b(무오버랩)보다 검색
지표는 나았지만, round3 감사 중 발견한 AM145JNPDBH1 "예약 기능" 케이스처럼
정답 문장 바로 앞에서 청크가 잘려 15% 오버랩으로도 못 막는 경우가 실제로
있었다(...예약기 에서 잘리고 다음 조각으로 넘어감). 슬라이딩 윈도우 자체는
그대로 두고 오버랩 비율만 15%->25%로 올려서 경계 누락이 줄어드는지, 그리고
MRR/Recall이 유지되는지(v2처럼 오히려 나빠지지 않는지) 확인한다.

사용법: python experiments/build_index_team_v3.py
"""

from __future__ import annotations

import time
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

import build_index_team_v2 as v2

ROOT = Path(__file__).resolve().parent.parent
OLD_CHROMA_PATH = ROOT / "chroma_db"
NEW_CHROMA_PATH = ROOT / "chroma_db_team_v3"

MAX_CHUNK_CHARS = 700
CHUNK_OVERLAP_CHARS = 175  # 700의 25% (기존 15%=105자에서 상향)
NO_TOC_MODELS = v2.NO_TOC_MODELS
PDF_SOURCES = v2.PDF_SOURCES


def sections_to_chunks_overlap25(sections: list[dict], product_model: str) -> list[dict]:
    """rag_tutorial.ipynb 원본 sections_to_chunks()와 동일한 문자 위치 슬라이딩
    윈도우 - CHUNK_OVERLAP_CHARS만 105->175로 올림."""
    step = MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS
    chunks = []
    for i, sec in enumerate(sections):
        text = sec["text"]
        pieces = [text[j:j + MAX_CHUNK_CHARS] for j in range(0, len(text), step)] or [text]
        for j, piece in enumerate(pieces):
            chunks.append({
                "id": f"{product_model}_{i}_{j}",
                "text": f"[{sec['heading']}] {piece}",
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

            sections = v2.extract_sections_v3(str(pdf_path))
            if not sections:
                print(f"{pdf_path.name:22s} [건너뜀] TOC 파싱 실패", flush=True)
                continue

            chunks = sections_to_chunks_overlap25(sections, product_model)
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
