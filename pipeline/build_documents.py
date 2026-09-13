"""가전제품 설명서 JSON(제품별 섹션 목록)을 {id, text, metadata} 문서로 변환한다.

입력 형식 (data/sample_manuals.json 참고):
    [
      {
        "product_model": "WM-1000",
        "product_name": "...",
        "category": "세탁기",
        "sections": [
          {"type": "사용법"|"오류코드"|"FAQ", "title": "...", "content": "..."},
          ...
        ]
      },
      ...
    ]

설명서 문단은 상품 설명보다 길어질 수 있어서, 일정 길이를 넘는 섹션은 문단
단위로 나눠 청킹한다 (짧으면 그대로 1개 문서).
"""
import json
from pathlib import Path

CHUNK_SIZE = 300  # 문자 수 기준. 이보다 짧은 섹션은 청킹하지 않음.
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def section_to_documents(product: dict, section: dict, section_index: int) -> list[dict]:
    header = f"[{product['product_name']} / {section['type']} / {section['title']}] "
    chunks = _chunk_text(section["content"])

    docs = []
    for chunk_index, chunk in enumerate(chunks):
        doc_id = f"{product['product_model']}_{section_index}_{chunk_index}"
        docs.append(
            {
                "id": doc_id,
                "text": header + chunk,
                "metadata": {
                    "product_model": product["product_model"],
                    "product_name": product["product_name"],
                    "category": product.get("category", ""),
                    "section_type": section["type"],
                    "section_title": section["title"],
                },
            }
        )
    return docs


def build_documents(input_path: str) -> list[dict]:
    products = json.loads(Path(input_path).read_text(encoding="utf-8"))
    docs = []
    for product in products:
        for i, section in enumerate(product["sections"]):
            docs.extend(section_to_documents(product, section, i))
    return docs


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config  # noqa: E402,F401  (Windows 콘솔 UTF-8 출력 설정을 위해 import)

    src = sys.argv[1] if len(sys.argv) > 1 else "data/sample_manuals.json"
    for d in build_documents(src):
        print(f"[{d['id']}] {d['text']}")
