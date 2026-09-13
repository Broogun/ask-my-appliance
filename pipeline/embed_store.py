"""문서를 임베딩해서 ChromaDB에 저장하는 범용 유틸 (도메인 무관).

EMBEDDING_BACKEND=local  (기본값, 무료): sentence-transformers 로컬 모델
EMBEDDING_BACKEND=openai (유료): OpenAI Embeddings API

TODO: 가전제품 설명서/FAQ를 {"id", "text", "metadata"} 형태의 문서 리스트로
변환하는 도메인 전용 빌더가 아직 없다 (예전 build_documents.py는 무신사 상품
JSON 전용이라 삭제함). 새 빌더가 만들어지면 ingest_documents()에 넘기면 된다.
"""
import argparse
import json

import chromadb

import config

_local_model = None  # sentence-transformers 모델은 무겁기 때문에 한 번만 로드해서 재사용


def _embed_local(texts: list[str]) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"[embed] 로컬 임베딩 모델 로딩 중: {config.LOCAL_EMBEDDING_MODEL} (최초 1회는 다운로드 필요)")
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _local_model.encode(texts, normalize_embeddings=True).tolist()


OPENAI_EMBED_BATCH_SIZE = 100  # 한 번에 너무 많이 보내면 300,000 토큰 제한에 걸림


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    embeddings = []
    for i in range(0, len(texts), OPENAI_EMBED_BATCH_SIZE):
        batch = texts[i:i + OPENAI_EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(d.embedding for d in resp.data)
    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    if config.EMBEDDING_BACKEND == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def get_collection():
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)


_known_models_cache = None


def get_known_models() -> list[str]:
    """벡터DB에 실제로 들어있는 product_model 값 목록 (중복 제거). 질문에 특정
    모델명이 언급됐는지 판별할 때 쓴다. 컬렉션이 자주 안 바뀌므로 캐싱한다."""
    global _known_models_cache
    if _known_models_cache is not None:
        return _known_models_cache
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    models = {m.get("product_model") for m in result["metadatas"] if m.get("product_model")}
    _known_models_cache = sorted(models)
    return _known_models_cache


def ingest_documents(docs: list[dict]):
    """docs: [{"id": str, "text": str, "metadata": dict}, ...] 형태를 받아 저장한다."""
    if not docs:
        print("[ingest] 문서가 없습니다.")
        return

    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)

    collection = get_collection()
    collection.upsert(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=texts,
        metadatas=[d.get("metadata", {}) for d in docs],
    )
    print(
        f"[ingest] {len(docs)}개 문서를 '{config.COLLECTION_NAME}' 컬렉션에 저장했습니다. "
        f"(embedding backend: {config.EMBEDDING_BACKEND})"
    )


def ingest(input_path: str):
    """이미 {"id", "text", "metadata"} 형태로 만들어진 문서 JSON 파일을 읽어서 저장한다."""
    docs = json.loads(open(input_path, encoding="utf-8").read())
    ingest_documents(docs)


def main():
    parser = argparse.ArgumentParser(description="문서를 임베딩하여 Chroma에 저장")
    parser.add_argument("--input", required=True, help="{id, text, metadata} 문서 JSON 파일 경로")
    args = parser.parse_args()
    ingest(args.input)


if __name__ == "__main__":
    main()
