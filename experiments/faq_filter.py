"""FAQ 생성 대상 청크 선별 (블록리스트 + 파편 병합). LLM 호출 없음."""
import re
from collections import defaultdict

# 법정 고지/행정 보일러플레이트 - 실사용 질문과 무관, 모델 안 가리고 거의 동일 문구.
# FAQ 생성에서만 제외하고 벡터 검색 대상으로는 그대로 남긴다.
BLOCK_TITLE_PATTERNS = [
    r"보증서", r"폐가전", r"폐\s*전자", r"소비자\s*분쟁", r"유상\s*서비스",
    r"전기안전\s*캠페인", r"오픈소스", r"제품\s*사용설명서$", r"^목차$", r"^차례$",
    r"^찾아보기$", r"서비스\s*센터\s*안내", r"^제품\s*규격$", r"에너지\s*효율\s*등급",
]
BLOCK_RE = re.compile("|".join(BLOCK_TITLE_PATTERNS))

# 페이지 폴백으로 생긴 의미 없는 제목("14페이지", "15" 등)
MEANINGLESS_TITLE_RE = re.compile(r"^\d+(페이지)?$")

MIN_TEXT_LEN = 150  # 이보다 짧으면 같은 섹션 앞 조각과 병합


def _section_key(chunk_id: str) -> str:
    """'GC-B40BSCQ_12_3' -> 'GC-B40BSCQ_12' (같은 섹션 조각끼리 묶는 키)"""
    return chunk_id.rsplit("_", 1)[0]


def build_faq_targets(ids, documents, metadatas):
    """FAQ 생성 대상 리스트를 만든다.

    반환: [{"source_chunk_id", "text", "section_title", "product_model",
            "category", "brand"}] - text는 짧은 파편이면 같은 섹션 이웃과 합친 것.
    """
    by_section = defaultdict(list)
    for cid, doc, meta in zip(ids, documents, metadatas):
        by_section[_section_key(cid)].append((cid, doc, meta))

    targets, blocked, merged_count = [], 0, 0
    for sec_key, items in by_section.items():
        items.sort(key=lambda x: int(x[0].rsplit("_", 1)[1]))
        for idx, (cid, doc, meta) in enumerate(items):
            title = meta.get("section_title", "")
            if BLOCK_RE.search(title) or MEANINGLESS_TITLE_RE.match(title):
                blocked += 1
                continue
            text = doc
            if len(doc) < MIN_TEXT_LEN and idx > 0:
                text = items[idx - 1][1] + "\n" + doc  # 같은 섹션 앞 조각과 병합
                merged_count += 1
            targets.append({
                "source_chunk_id": cid,
                "text": text,
                "section_title": title,
                "product_model": meta.get("product_model"),
                "category": meta.get("category"),
                "brand": meta.get("brand"),
            })
    return targets, blocked, merged_count


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(".").resolve()))
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp02", Path("experiments/박형건_exp02.py").resolve())
    exp02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp02)
    state = exp02._load_state()
    all_docs = state["collection"].get(include=["documents", "metadatas"])

    targets, blocked, merged = build_faq_targets(
        all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
    )
    total = len(all_docs["ids"])
    print(f"전체 청크: {total}")
    print(f"블록리스트로 제외: {blocked} ({blocked/total:.1%})")
    print(f"짧은 파편이라 앞 조각과 병합: {merged}")
    print(f"FAQ 생성 대상: {len(targets)} ({len(targets)/total:.1%})")

    avg_len = sum(len(t["text"]) for t in targets) / len(targets)
    print(f"생성 입력 평균 길이: {avg_len:.0f}자")

    print("\n제외된 제목 샘플:")
    seen = set()
    for cid, doc, meta in zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]):
        t = meta.get("section_title", "")
        if (BLOCK_RE.search(t) or MEANINGLESS_TITLE_RE.match(t)) and t not in seen:
            seen.add(t)
            print(f"  {t!r}")
        if len(seen) >= 15:
            break

    for m in ("GC-B40BSCQ", "DF90H24R5C"):
        n = sum(1 for t in targets if t["product_model"] == m)
        print(f"\n프로토타입 대상 {m}: FAQ 생성 대상 {n}개 청크")
