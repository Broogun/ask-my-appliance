# -*- coding: utf-8 -*-
"""data/manuals_pdf/lg/{aircon,fridge,washer}/*.pdf 를 전부 청킹+임베딩한다.
이미 넣은 에어컨 10개도 다시 넣지만 upsert라 중복 문제 없음."""
import glob
import os

import config  # noqa: F401
from pipeline.embed_store import ingest_documents
from pipeline.pdf_chunker import build_documents_from_pdf

CATEGORY_MAP = {"aircon": "에어컨", "fridge": "냉장고", "washer": "세탁기"}

all_docs = []
for folder, category in CATEGORY_MAP.items():
    paths = sorted(glob.glob(f"data/lg/{folder}/*.pdf"))
    print(f"[{category}] 대상 PDF {len(paths)}개")
    for path in paths:
        model = os.path.splitext(os.path.basename(path))[0].split("_", 1)[-1]
        print(f"  [{model}] 청킹 중...")
        docs = build_documents_from_pdf(
            path,
            product_model=model,
            product_name=f"LG {category} {model}",
            category=category,
            extract_figures=True,
            figures_out_dir=f"data/manual_figures/lg/{folder}/{model}",
        )
        print(f"    -> {len(docs)}개 청크")
        all_docs.extend(docs)

print(f"\n총 {len(all_docs)}개 문서 임베딩 중...")
ingest_documents(all_docs)
print("완료")
