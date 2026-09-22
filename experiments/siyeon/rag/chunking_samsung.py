"""삼성 가전 매뉴얼 청킹 — 목차(TOC) 기반. 에어컨·냉장고·세탁기·건조기·스타일러·슈드레서 공용.

    from siyeon.rag.chunking_samsung import build_samsung_chunks
    chunks = build_samsung_chunks()      # data/samsung/{aircon,fridge,washer}/*.pdf + data/samsung_*_errors.json

청크 dict 형식은 LG(chunking_lg.py)와 같다. LG 파서의 헬퍼(make_chunk, split_oversized, parse_by_toc, parse_pdf_legacy …)를
그대로 쓰고, 삼성 매뉴얼(InDesign)에서 달라지는 것만 여기 있다:

  1) 챕터 범위 — LG는 고정 챕터명 + "두 줄 반복" 패턴이지만 삼성은 챕터명이 제품군·세대마다 달라서(에어컨만 3계열)
     목차 1단계의 페이지 번호를 그대로 챕터 범위로 쓴다. 30개 중 27개가 목차 있음.
  2) 챕터 역할 — 이름 정확 일치 대신 키워드: '고장·문제·점검' → 문제해결 파서, '주의·안전' → safety, 나머지 → 목차 파서
  3) 고장신고 — 세탁기·건조기·스타일러는 표(`증상|확인/조치`, `코드|진단|해결방법`), 에어컨·냉장고는 대부분 표 없는 텍스트.
     텍스트형은 증상 줄을 (a) 본문보다 큰 글씨(에어컨 11pt vs 9.7pt) 또는 (b) '~나요!' 로 끝나고 다음 줄이 불릿 으로 감지해
     증상 1개 = 청크 1개로 만든다.
  4) `코드|진단|해결방법` 표의 코드 셀은 이미지라 글자가 안 나온다 → 진단·해결 텍스트만 증상 청크로. 코드 조회는 RDB가 담당.
  5) 목차 없는 3개(RP20C3111S9, SRS705IC(띄어쓰기 없음), WF21T6500KW)는 LG와 같은 폴백 파서. 삼성은 굵은 글씨를 안 써서
     크기 차이만으로 소제목을 잡는다.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

from .chunking_lg import (ROOT, CATEGORY_KO, make_chunk, split_oversized, toc_headings, parse_by_toc, _restore_spacing,
                       _tidy_cell, load_error_code_chunks, assign_pages)

SAMSUNG_DIR = ROOT / "data" / "samsung"
SAMSUNG_ERROR_JSONS = sorted((ROOT / "data").glob("samsung_*_errors.json"))

TROUBLE_WORDS = ("고장", "문제", "점검", "해결")
SAFETY_WORDS = ("주의", "안전")
# 고장신고 표 헤더 (앞 2~3열). BOM(﻿)·공백 제거 후 비교
TABLE_HEADERS = {("증상", "확인/조치"), ("증상", "원인및해결책"), ("카테고리", "증상", "원인및해결책"),
                 ("증상", "확인", "조치사항"), ("코드", "진단", "해결방법")}
SKIP_TITLES = ("찾아보기", "오픈소스", "전자 제품을 버리려면", "폐 전자 제품", "제품보증서", "확인동의서", "메모")
_SYMPTOM_END_RE = re.compile(r"(?<!세)(요|까|죠|네)\s*[!?.]?\s*(\([^()]*\))?\s*$")   # '~나요! (뚝, 딱 …)' 는 증상, '~하세요.' 는 안내문이라 제외


def _norm_cell(c) -> str:
    return re.sub(r"\s+", "", (c or "").replace("﻿", ""))


def _product_type_map() -> dict[str, str]:
    try:
        sys.path.insert(0, str(ROOT))
        from experiments.harness.evaluate import MODEL_LIST
        return {model: category for brand, category, model in MODEL_LIST if brand == "삼성"}
    except Exception:
        return {}


# ── 챕터 범위: 목차 1단계 ─────────────────────────────────────────────────────
_PAREN_SUFFIX_RE = re.compile(r"\s*\([^()]*\)\s*$")   # '스마트 케어 (해당 모델만 참조하세요.)' → '스마트 케어'


def _title_variants(title: str) -> set[str]:
    t = title.strip()
    return {t, _PAREN_SUFFIX_RE.sub("", t).strip()} - {""}


def _toc_pages(doc, page_lines: list[set[str]]) -> set[int]:
    """목차(차례)가 인쇄된 페이지들. 목차 제목의 40% 이상이 줄로 등장하는 페이지로 본다.
    제목을 본문에서 찾을 때 이 페이지들은 건너뛴다 — 안 그러면 목차가 깨진 문서(RF_RH82M9152SL: 챕터 4개가 0페이지)에서
    챕터 시작을 목차 페이지로 잡고, 그 안의 목차 줄들을 소제목으로 오인해 같은 소제목 3개가 챕터 6개에 반복됐다."""
    titles = {v for _, t, _ in doc.get_toc() for v in _title_variants(t)}
    if len(titles) < 5:
        return set()
    return {i for i, lines in enumerate(page_lines) if len(titles & lines) >= max(5, 0.4 * len(titles))}


def _find_title_page(title: str, page_lines: list[set[str]], hint: int | None, skip: set[int] = frozenset()) -> int | None:
    """제목이 줄 하나로 등장하는 페이지. 목차가 가리킨 페이지(hint)와 그 다음 페이지를 먼저, 없으면 문서 전체에서.
    skip(목차 페이지)은 제외."""
    variants = _title_variants(title)
    order = ([hint, hint + 1] if hint is not None else []) + list(range(len(page_lines)))
    for k in order:
        if 0 <= k < len(page_lines) and k not in skip and variants & page_lines[k]:
            return k
    return None


def chapters_from_toc(doc, page_lines: list[set[str]] | None = None) -> list[tuple[str, int, int]]:
    """[(챕터명, 시작 idx, 끝 idx)]. 표지·차례·찾아보기 등은 제외.

    목차의 페이지 번호를 그대로 믿지 않고 본문에서 제목 줄을 찾아 검증한다 — RF_RH82M9152SL은 목차 항목 3개가 0페이지를
    가리키고 '고장신고 전 확인하기'가 실제로는 치수 표 페이지를 가리키는 등 목차가 깨져 있다."""
    if page_lines is None:
        page_lines = [{ln.strip() for ln in pg.get_text().split("\n")} for pg in doc]
    skip = _toc_pages(doc, page_lines)
    l1 = []
    for l, t, pg in doc.get_toc():
        t = t.strip()
        if l != 1 or any(k in t for k in SKIP_TITLES) or t.lower().endswith(".pdf"):
            continue
        p = _find_title_page(t, page_lines, pg - 1 if pg >= 1 else None, skip)
        if p is not None:
            l1.append((t, p))
    l1.sort(key=lambda x: x[1])
    out = []
    for i, (t, p) in enumerate(l1):
        end = l1[i + 1][1] if i + 1 < len(l1) else len(doc)
        if end <= p:                       # 같은 페이지에서 다음 챕터가 시작 → 이 챕터는 그 페이지 하나
            end = p + 1
        out.append((t, p, end))
    return out


def strip_furniture_samsung(text: str, chapter_titles: set[str]) -> str:
    """페이지 앞머리 '21 / 사용하기' (페이지 번호 + 러닝헤더) 제거. LG와 같은 패턴, 챕터명만 문서 목차에서."""
    lines = text.split("\n")
    k = 0
    while k < len(lines) and k < 4:
        s = lines[k].strip()
        if s == "" or s.isdigit() or s in chapter_titles:
            k += 1
        else:
            break
    return "\n".join(lines[k:])


# ── 고장신고: 표 + 텍스트 ─────────────────────────────────────────────────────
def _body_font_size(page) -> float:
    sizes = Counter()
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if len(s["text"].strip()) > 5:
                    sizes[round(s["size"], 1)] += len(s["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _lines_with_size(page) -> list[tuple[str, float, float]]:
    """[(텍스트, 글자크기, x)] 줄 단위."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            sp = [s for s in ln["spans"] if s["text"].strip()]
            if sp:
                out.append(("".join(s["text"] for s in sp).strip(), max(s["size"] for s in sp), sp[0]["bbox"][0]))
    return out


def parse_trouble_samsung(doc, toc, doc_id, start, end, section, chapter_titles, meta) -> list[dict]:
    chunks: list[dict] = []
    table_pages: set[int] = set()

    # (1) 표 — 증상 1개 = 청크 1개 (LG와 같은 방식). '코드|진단|해결방법'은 코드 셀이 이미지라 진단 문장을 증상으로 씀
    current: dict | None = None
    def flush():
        if current and current["parts"]:
            body = f"증상: {current['symptom']}\n" + "\n".join(current["parts"])
            chunks.append(make_chunk(doc_id, section, current["category"], body, tag="symptom", **meta))
    for i in range(start, end):
        for table in sorted(doc[i].find_tables().tables, key=lambda t: t.bbox[1]):
            rows = table.extract()
            if not rows or len(rows[0]) < 2:
                continue
            head = tuple(_norm_cell(c) for c in rows[0][:3])
            if head not in TABLE_HEADERS and head[:2] not in TABLE_HEADERS:
                continue
            table_pages.add(i)
            is_code = head[0] == "코드"
            has_cat = head[0] == "카테고리"
            category = "점검 코드" if is_code else "문제 해결"
            for row in rows[1:]:
                cells = [(c or "") for c in row]
                if has_cat:
                    if cells[0].strip():
                        category = re.sub(r"\s+", " ", cells[0].replace("﻿", "")).strip()
                    cells = cells[1:]
                cells = [c.replace("\ufeff", "") for c in cells]
                if is_code:
                    diag, fix = (cells[1] if len(cells) > 1 else ""), (cells[2] if len(cells) > 2 else "")
                    symptom, cause = re.sub(r"\s+", " ", diag).strip(), _tidy_cell(fix)
                else:
                    symptom, cause = re.sub(r"\s+", " ", cells[0]).strip(), _tidy_cell(cells[1] if len(cells) > 1 else "")
                if not cause:
                    continue
                if symptom and not (current and current["symptom"] == symptom):
                    flush(); current = {"symptom": symptom, "category": category, "parts": []}
                if current is None:
                    current = {"symptom": "(증상 미상)", "category": category, "parts": []}
                current["parts"].append(cause)
    flush()

    # (2) 텍스트 — 표가 없는 페이지: 증상 줄(큰 글씨 또는 '~나요!' + 다음 줄 불릿) 단위로 끊는다
    text_pages = [i for i in range(start, end) if i not in table_pages]
    if text_pages:
        buf: list[str] = []
        category = ""
        symptom = ""
        def flush_text():
            body = "\n".join(buf).strip()
            if len(body) > 30:
                label = " > ".join(x for x in (category, symptom) if x)
                chunks.append(make_chunk(doc_id, section, label or "문제 해결", body, tag="symptom" if symptom else "diagnosis", **meta))
        for i in text_pages:
            body_size = _body_font_size(doc[i])
            raw = _lines_with_size(doc[i])
            # 홀로 선 불릿 기호 줄('ꞏ')은 다음 줄 앞에 붙인다
            lines: list[tuple[str, float, float]] = []
            for text, size, x in raw:
                if lines and lines[-1][0] in ("•", "ꞏ", "·"):
                    lines[-1] = (lines[-1][0] + " " + text, size, lines[-1][2])
                else:
                    lines.append((text, size, x))
            titles_here = {h["title"] for h in toc if h["page_idx"] in (i - 1, i, i + 1)}
            # 글자 크기로 증상을 구분하는 페이지인가 — 절 제목(목차 제목)은 제외해야 한다.
            # (냉장고 문제해결 첫 페이지는 절 제목 '안심하세요! 고장이 아닙니다'만 11pt고 증상·본문은 둘 다 10pt인데,
            #  제목을 증상으로 오판하면 말투 규칙이 꺼져서 그 페이지 증상 7개가 통째로 누락됐다)
            page_has_mid = any(body_size + 0.8 <= s < body_size + 2.5 for txt, s, _ in lines
                               if txt not in titles_here and txt not in chapter_titles)
            for j, (text, size, x) in enumerate(lines):
                if text.isdigit() or text in chapter_titles:
                    continue                                        # 페이지 번호·러닝헤더
                nxt = lines[j + 1][0] if j + 1 < len(lines) else ""
                is_bullet = text.startswith(("•", "ꞏ", "·", "-", "→"))
                big = size >= body_size + 2.5                         # 20pt 절 제목 / 13pt 카테고리
                mid = body_size + 0.8 <= size < body_size + 2.5       # 11pt 증상 (에어컨)
                looks_symptom = (not page_has_mid and not is_bullet and len(text) <= 60 and _SYMPTOM_END_RE.search(text)
                                 and nxt.startswith(("•", "ꞏ", "·")))   # 냉장고: 크기 같고 '~나요!' + 다음 줄 불릿
                if (text in titles_here and _SYMPTOM_END_RE.search(text)) or ((mid or looks_symptom) and text not in titles_here):
                    flush_text(); buf = []; symptom = text                      # RT42/RT53: 증상 문장이 목차 2단계 제목으로 들어 있음
                elif text in titles_here or big:
                    flush_text(); buf = []; category, symptom = text, ""
                elif False:
                    flush_text(); buf = []; symptom = text
                else:
                    buf.append(text)
        flush_text()
    return chunks


# ── PDF 1개 ───────────────────────────────────────────────────────────────────
def parse_pdf_samsung(pdf_path: Path, category: str = "", product_type: str = "") -> list[dict]:
    from .chunking_lg import parse_pdf_legacy
    doc_id = pdf_path.stem
    doc = pymupdf.open(str(pdf_path))
    meta = {"brand": "samsung", "category": category, "product_type": product_type}
    page_lines = [{ln.strip() for ln in pg.get_text().split("\n")} for pg in doc]
    chapters = chapters_from_toc(doc, page_lines)
    toc = toc_headings(doc)
    if len(chapters) < 3 or not toc:
        doc.close()
        chunks = parse_pdf_legacy(pdf_path, category, product_type, allow_unbold=True)   # 삼성은 굵은 글씨로 소제목을 안 표시
        for c in chunks:
            c["brand"] = "samsung"
        return chunks
    chapter_titles = {t for t, _, _ in chapters} | {t.strip() for l, t, pg in doc.get_toc() if l == 1}
    page_texts = [strip_furniture_samsung(pg.get_text(), chapter_titles) for pg in doc]
    # 2·3단계 소제목도 실제 등장 페이지로 보정하고, '(해당 모델만 참조하세요.)' 같은 괄호 접미사를 뗀 변형을 함께 등록
    fixed = []
    skip = _toc_pages(doc, page_lines)
    for h in toc:
        p = _find_title_page(h["title"], page_lines, h["page_idx"], skip)
        if p is not None:
            for v in _title_variants(h["title"]):
                fixed.append({"level": h["level"], "title": v, "page_idx": p})
    toc = fixed
    # parse_by_toc는 LG용 strip_page_furniture를 내부에서 부르지만 삼성 챕터명엔 반응하지 않으므로 여기서 먼저 떼고 넘긴다
    raw_page_texts = [pg.get_text() for pg in doc]     # 페이지 위치 사후 탐색용 (furniture 떼기 전 원문)
    chunks: list[dict] = []
    for title, start, end in chapters:
        if any(k in title for k in TROUBLE_WORDS):
            mine = parse_trouble_samsung(doc, toc, doc_id, start, end, title, chapter_titles, meta)
        elif any(k in title for k in SAFETY_WORDS):
            mine = parse_by_toc(page_texts, toc, doc_id, start, end, title, meta, tag="safety")
        else:
            mine = parse_by_toc(page_texts, toc, doc_id, start, end, title, meta, detect_hidden=True, tag="howto")
        chunks += assign_pages(mine, raw_page_texts, start, end)
    doc.close()
    return chunks


def build_samsung_chunks(categories: tuple[str, ...] = ("aircon", "fridge", "washer")) -> list[dict]:
    ptype = _product_type_map()
    chunks: list[dict] = []
    for cat in categories:
        for pdf_path in sorted((SAMSUNG_DIR / cat).glob("*.pdf")):
            model = pdf_path.stem.split("_", 1)[1] if "_" in pdf_path.stem else pdf_path.stem
            chunks += parse_pdf_samsung(pdf_path, cat, ptype.get(model, CATEGORY_KO.get(cat, "")))
    errs = load_error_code_chunks([p for p in SAMSUNG_ERROR_JSONS if p.stem.replace("samsung_", "").replace("_errors", "") in categories])
    for c in errs:
        c["brand"] = "samsung"
    return chunks + errs


if __name__ == "__main__":
    cs = build_samsung_chunks()
    print(f"\n총 {len(cs)}개 청크  태그별: {Counter(c['tag'] for c in cs)}")
    for doc_id in sorted({c["doc_id"] for c in cs}):
        mine = [c for c in cs if c["doc_id"] == doc_id]
        L = sorted(len(c["body"]) for c in mine)
        sym = Counter(c["subsection"].split(" > ")[0] for c in mine if c["tag"] == "symptom")
        print(f"  {doc_id:18s} {mine[0].get('product_type',''):6s} {len(mine):4d}개 (중간 {L[len(L)//2]:4d}자)  증상: {dict(sym) if sym else '-'}")
