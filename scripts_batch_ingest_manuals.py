# -*- coding: utf-8 -*-
"""data/manuals_pdf/AC_*.pdf 10개를 전부 청킹 + 임베딩해서 벡터DB에 넣는다."""
import glob
import os

import config  # noqa: F401
from pipeline.embed_store import ingest_documents
from pipeline.pdf_chunker import build_documents_from_pdf

paths = sorted(glob.glob("data/lg/aircon/AC_*.pdf"))
print(f"대상 PDF {len(paths)}개")

all_docs = []
for path in paths:
    model = os.path.splitext(os.path.basename(path))[0].replace("AC_", "")
    print(f"[{model}] 청킹 중...")
    docs = build_documents_from_pdf(
        path,
        product_model=model,
        product_name=f"LG 에어컨 {model}",
        category="에어컨",
    )
    print(f"  -> {len(docs)}개 청크")
    all_docs.extend(docs)

print(f"\n총 {len(all_docs)}개 문서 임베딩 중...")
ingest_documents(all_docs)
print("완료")
