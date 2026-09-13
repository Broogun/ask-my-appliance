"""사용설명서 청킹 테스트용 - 백그라운드로 도는 에러코드 평가와 겹치지 않게
별도의 Chroma 경로/컬렉션을 씀."""
import config

config.CHROMA_DIR = "chroma_db_manual_test"
config.COLLECTION_NAME = "manual_test"

from pipeline.embed_store import ingest_documents
from pipeline.pdf_chunker import build_documents_from_pdf

docs = build_documents_from_pdf(
    "data/manuals_pdf/AC_FQ18GU1BHN.pdf",
    product_model="FQ18GU1BHN",
    product_name="LG 벽걸이 에어컨 FQ18GU1BHN",
    category="에어컨",
)
print(f"[build] {len(docs)}개 문서(청크) 생성")
ingest_documents(docs)
