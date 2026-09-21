"""설명서 PDF 경로 + 청크가 실제로 실린 페이지 찾기.

벡터DB의 청크 page 메타데이터는 '섹션이 시작한 페이지'라서 청크가 뒤쪽 페이지에 있으면 1~5쪽 앞을 가리킨다
(무작위 청크 378개 중 117개가 틀렸고 틀린 방향은 항상 실제보다 앞쪽). 출처 카드의 p.N, PDF 링크, 그림 매칭이 모두
정확한 페이지를 필요로 해서, 조회 시점에 청크 본문을 PDF 페이지 텍스트에서 찾아 보정한다(벡터DB는 건드리지 않음).
"""
import re
from functools import lru_cache
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
_FORWARD_WINDOW = 15     # 틀린 값은 항상 실제보다 앞쪽 - claimed 이후 이 범위에서만 찾는다
_PROBE = 18


def manual_pdf_path(doc_id: str) -> Path | None:
    """doc_id(PDF 파일명 stem) → data/{brand}/{category}/{doc_id}.pdf. 에러코드 공통 문서('LG-AC-COMMON')는 None."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id) or doc_id.endswith("-COMMON"):
        return None
    return next(iter((ROOT / "data").glob(f"*/*/{doc_id}.pdf")), None)


def _nz(t: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", t)


@lru_cache(maxsize=64)
def _page_texts(pdf_path: str) -> tuple[str, ...]:
    with pymupdf.open(pdf_path) as d:
        return tuple(_nz(p.get_text()) for p in d)


def locate_pages(doc_id: str, body: str, claimed: int | None) -> tuple[int | None, int | None]:
    """(page_start, page_end). 본문의 앞/뒤 조각이 실린 페이지를 찾고, 못 찾으면 claimed 그대로."""
    path = manual_pdf_path(doc_id)
    b = _nz(body)
    if path is None or not claimed or len(b) < 40:
        return claimed, claimed
    pages = _page_texts(str(path))

    def where(off: int) -> int | None:
        probe = b[off:off + _PROBE]
        if len(probe) < _PROBE:
            return None
        for n in range(claimed, min(len(pages), claimed + _FORWARD_WINDOW) + 1):
            if probe in pages[n - 1]:
                return n
        return None

    first, last = where(10), where(max(len(b) - _PROBE - 10, 10))
    if first is None and last is None:
        return claimed, claimed
    ps, pe = first or last, last or first
    return ps, max(ps, pe)
