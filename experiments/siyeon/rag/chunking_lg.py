"""LG 가전 매뉴얼 청킹 — PDF 목차(TOC) 기반. 에어컨·냉장고·세탁기·건조기·스타일러 공용.

    from siyeon.rag.chunking_lg import build_all_chunks
    chunks = build_all_chunks()          # data/lg/{aircon,fridge,washer}/*.pdf + data/lg_*_errors.json → 청크(dict) 목록

청크 dict: {"id", "doc_id", "brand", "category", "product_type", "section", "subsection", "body", "tag", "text",
            "page_start", "page_end"}
  - id           : md5 기반, 재실행해도 동일 (Chroma id / RDB chunk_id로도 쓰임)
  - page_start/end : 본문이 실린 PDF 페이지(1부터, PDF 뷰어의 #page=N 과 같은 번호). 출처 표시용. PDF 청크에만 있고
                     에러코드 JSON 청크엔 없다. 파서가 추적하는 게 아니라 assign_pages()가 본문을 페이지 원문에서 사후 탐색
  - doc_id       : PDF 파일명 stem ('AC_FQ18GC1EHN') 또는 공통 문서 'LG-AC-COMMON' 등(에러코드 JSON)
  - category     : data/ 폴더명 (aircon | fridge | washer) — 에러코드 JSON·RDB 조회 범위와 같은 기준
  - product_type : 평가 체계(pipeline.evaluate.MODEL_LIST)의 8분류 (에어컨/냉장고/김치냉장고/세탁기/건조기/세탁건조기/스타일러) — 표시용
  - text         : 임베딩 대상 = "섹션 > 소제목\\n본문"
  - tag          : howto | symptom | safety | diagnosis | generic | error_code

LG 최신 매뉴얼은 프레임메이커로 만들어져 (1) 3단계 목차가 PDF 북마크로 내장되고 (2) 챕터 시작 페이지에서 챕터명이
두 줄 연속 반복되고 (3) 고장신고 절이 [증상 | 원인 및 해결책] 표 객체로 들어 있다. 20개 중 17개가 이 템플릿이고,
구형 3개(RF_CA-H17DC, RF_S825AW35, RF_S835S31)는 목차가 없어 폰트·번호 기반 폴백 파서(parse_pdf_legacy)로 처리한다.
왜 이렇게 자르는지는 experiments/siyeon/docs/overview_lg.md 1절 참고.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[3]          # 프로젝트 루트 (ask-my-appliance/)
LG_DIR = ROOT / "data" / "lg"                        # aircon/ fridge/ washer/
DATA_DIR = LG_DIR / "aircon"                         # (하위 호환) 노트북이 쓰던 이름
ERROR_JSONS = sorted((ROOT / "data").glob("lg_*_errors.json"))
ERROR_JSON = ROOT / "data" / "lg_aircon_errors.json"   # (하위 호환)
COMMON_DOC_ID = "LG-AC-COMMON"                          # (하위 호환) 에어컨 공통 문서 id
CATEGORY_KO = {"aircon": "에어컨", "fridge": "냉장고", "washer": "세탁기"}

# 대챕터. 세탁건조기는 '사용하기 - 세탁기', '관리하기 - 건조기'처럼 접미사가 붙는다 → 접두 매칭(is_chapter_title).
CHAPTER_BASES = [
    "안전을 위해 주의하기", "LG ThinQ 사용하기", "알아보기",
    "리모컨으로 사용하기", "사용하기", "조작부로 사용하기",
    "관리하기", "설치하기", "고장 신고 전 확인하기", "제품 보증서 보기", "부록",
]
TOP_LEVEL_SECTIONS = CHAPTER_BASES                   # (하위 호환)
HOWTO_BASES = {"리모컨으로 사용하기", "사용하기", "조작부로 사용하기"}   # 절차형(1,2,3 단계) 챕터: 4단계 숨은 소제목 감지
HOWTO_SECTIONS = HOWTO_BASES                         # (하위 호환)
SKIP_GROUPS = {"오픈소스 정보", "폐가전제품 처리 절차", "생활 속 전기안전 캠페인"}   # 사용법/고장과 무관 → 색인 제외
MIN_CHAPTERS = 3                                     # 이보다 적게 잡히면 템플릿이 다른 PDF → 건너뜀

MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 100


def chapter_base(title: str) -> str | None:
    """'사용하기 - 세탁기' → '사용하기', '고장 신고 전 확인하기' → 그대로, 그 외 → None"""
    t = title.strip()
    for base in CHAPTER_BASES:
        if t == base or t.startswith(base + " - "):
            return base
    return None


def is_chapter_title(line: str) -> bool:
    return chapter_base(line) is not None


def _product_type_map() -> dict[str, str]:
    """모델명 → 8분류 카테고리 (pipeline.evaluate.MODEL_LIST). 평가 모듈이 없으면 빈 dict."""
    try:
        sys.path.insert(0, str(ROOT))
        from pipeline.evaluate import MODEL_LIST
        return {model: category for brand, category, model in MODEL_LIST if brand == "LG"}
    except Exception:
        return {}


# ── 청크 헬퍼 ────────────────────────────────────────────────────────────────
# 삼성 InDesign PDF에서 아이콘 폰트가 U+0780~U+0DFF(탈라나·은코·데바나가리 …) 구간의 엉뚱한 글자로 추출된다
# ("ऍ࠘߄श", "ߩ") — 8개 문서에 수천 자. 여기에 BOM(U+FEFF) 같은 서식 문자까지 섞이면 리랭커가 그 조각을 통째로
# 무관(거리 1.0)으로 본다 (RT42CG6024S9 '온도 설정 방법' 표 → "여름에 온도 몇 도로 맞춰요?"가 none 처리되던 문제).
_JUNK_RE = re.compile("[\\u0780-\\u0dff\\ufeff\\u200b-\\u200f\\u2028-\\u202e\\ue000-\\uf8ff]")


def clean_text(text: str) -> str:
    """깨진 폰트 글자·서식 문자 제거 + 그 결과 생긴 빈 줄 정리. 청크 본문과 페이지 원문(assign_pages) 양쪽에 같이 쓴다."""
    text = _JUNK_RE.sub("", text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)


def make_chunk(doc_id: str, section: str, subsection: str, body: str, tag: str = "", **meta) -> dict:
    """청크 dict 하나. text = 섹션 경로 + 본문 (임베딩 대상). id = md5(body) 기반이라 재실행해도 같다."""
    body = clean_text(body).strip()
    breadcrumb = f"{section} > {subsection}" if subsection else section
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()[:8]
    chunk_id = re.sub(r"\s+", "_", f"{doc_id}_{section}_{subsection}_{digest}")
    c = {"doc_id": doc_id, "section": section, "subsection": subsection, "body": body, "tag": tag,
         "id": chunk_id, "text": f"{breadcrumb}\n{body}"}
    c.update(meta)
    return c


def split_oversized(chunks: list[dict]) -> list[dict]:
    """MAX_CHUNK_CHARS 초과 청크는 줄 경계에서 자르고 CHUNK_OVERLAP_CHARS만큼 겹친다."""
    result = []
    for c in chunks:
        if len(c["body"]) <= MAX_CHUNK_CHARS:
            result.append(c)
            continue
        meta = {k: v for k, v in c.items() if k not in ("doc_id", "section", "subsection", "body", "tag", "id", "text")}
        piece: list[str] = []
        size = 0
        for line in c["body"].split("\n"):
            if size + len(line) + 1 > MAX_CHUNK_CHARS and piece:
                result.append(make_chunk(c["doc_id"], c["section"], c["subsection"], "\n".join(piece), c["tag"], **meta))
                carry, carry_size = [], 0
                for prev in reversed(piece):
                    if carry_size + len(prev) > CHUNK_OVERLAP_CHARS:
                        break
                    carry.insert(0, prev); carry_size += len(prev) + 1
                piece, size = carry, carry_size
            piece.append(line); size += len(line) + 1
        if piece:
            result.append(make_chunk(c["doc_id"], c["section"], c["subsection"], "\n".join(piece), c["tag"], **meta))
    return result


def to_document(c: dict) -> Document:
    """청크 dict → LangChain Document. page_content(임베딩 대상)는 text, 나머지는 메타데이터(Chroma 필터용)."""
    return Document(
        page_content=c["text"],
        metadata={"id": c["id"], "doc_id": c["doc_id"], "brand": c.get("brand", "lg"),
                  "category": c.get("category", ""), "product_type": c.get("product_type", ""),
                  "section": c["section"], "subsection": c["subsection"], "body": c["body"], "tag": c["tag"]},
    )


# ── 페이지 위치 (출처 표시용) ─────────────────────────────────────────────────
def _squash(s: str) -> str:
    return re.sub(r"\s+", "", clean_text(s))     # 본문(make_chunk)과 같은 정리를 페이지 원문에도 → 비교가 어긋나지 않게


def assign_pages(chunks: list[dict], raw_page_texts: list[str], start: int = 0, end: int | None = None) -> list[dict]:
    """청크가 PDF 몇 페이지에서 왔는지 사후 탐색해 c["page_start"], c["page_end"](1-based)를 채운다.
    파서 4개(목차/고장표/구형/삼성)에 페이지 추적을 각각 넣는 대신, 본문의 앞부분·뒷부분(공백 제거)을 [start, end) 범위의
    페이지 원문에서 찾는다. 공백을 지우고 비교하므로 표 셀 줄바꿈이나 kiwi 띄어쓰기 복원과 무관하다.
    앞부분은 길게(120자) 먼저 시도한다 — '알아두기 • LG전자 제품의 건조 용량은…' 같은 보일러플레이트는 두 페이지에
    다 있어서 짧게 잡으면 앞 페이지에 잘못 붙는다. 페이지 경계에 걸리면 점점 짧게 재시도하고, 못 찾으면 필드를 안 넣는다."""
    pages = [_squash(t) for t in raw_page_texts]
    end = len(pages) if end is None else min(end, len(pages))
    for c in chunks:
        body = _squash(c["body"])
        if body.startswith("증상:"):                      # 고장표 청크는 '증상:' 접두를 뺀 실제 셀 텍스트로
            body = body[3:]
        first = last = None
        probes = [body[:n] for n in (120, 80, 40, 20)] + [body[k:k + 30] for k in (30, 60, 90)]   # 뒤쪽 창은 절차 번호가 빠져
        for head in probes:                                                                   # 순서가 어긋난 조각용
            first = next((i for i in range(start, end) if head and head in pages[i]), None)
            if first is not None:
                break
        if first is None:
            continue
        for n in (120, 80, 40, 20):                    # 뒷부분은 시작 페이지부터 앞으로 — 청크(≤1200자)는 길어야 2쪽이라
            tail = body[-n:]                             # 뒤에서 거꾸로 찾으면 먼 페이지의 같은 상용구에 붙는다
            last = next((i for i in range(first, end) if tail and tail in pages[i]), None)
            if last is not None:
                break
        c["page_start"], c["page_end"] = first + 1, (last if last is not None else first) + 1
    return chunks


# ── 페이지/목차 ───────────────────────────────────────────────────────────────
def extract_page_texts(doc) -> list[str]:
    """페이지별 원문 (러닝헤더·페이지번호 포함 — locate_sections가 그 패턴을 씀. 본문 조립 시 strip_page_furniture)."""
    return [page.get_text() for page in doc]


def strip_page_furniture(text: str) -> str:
    """페이지 앞머리의 페이지 번호·러닝헤더(챕터명) 제거. '38\\n관리하기\\n…' / '62\\n관리하기 - 세탁기\\n관리하기 - 세탁기\\n…'"""
    lines = text.split("\n")
    k = 0
    while k < len(lines) and k < 4:
        s = lines[k].strip()
        if s == "" or s.isdigit() or is_chapter_title(s):
            k += 1
        else:
            break
    return "\n".join(lines[k:])


def locate_sections(page_texts: list[str]) -> dict[str, tuple[int, int]]:
    """대챕터 → (시작 페이지 idx, 끝 idx). 챕터 시작 페이지는 챕터명이 인접 두 줄에 연속 반복된다(러닝헤더 + 제목).
    목차 페이지엔 이 패턴이 없어서 오탐이 없다. 키는 '사용하기 - 세탁기'처럼 전체 제목."""
    starts: dict[str, int] = {}
    for i, text in enumerate(page_texts):
        lines = [ln.strip() for ln in text.strip().split("\n")]
        for j in range(min(6, len(lines) - 1)):
            if lines[j] and lines[j] == lines[j + 1] and is_chapter_title(lines[j]) and lines[j] not in starts:
                starts[lines[j]] = i
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    return {name: (start, ordered[k + 1][1] if k + 1 < len(ordered) else len(page_texts))
            for k, (name, start) in enumerate(ordered)}


def toc_headings(doc) -> list[dict]:
    """PDF 북마크의 2·3단계 제목. LG 최신 매뉴얼 전부에서 본문 줄과 글자 그대로 일치(에어컨 10개 100%, 냉장고·세탁기 17개 98~100%)."""
    return [{"level": level, "title": title.strip(), "page_idx": page_no - 1}
            for level, title, page_no in doc.get_toc() if level in (2, 3)]


# ── 고장 신고 전 확인하기 > 문제 해결하기: 표 기반, 증상 1개 = 청크 1개 ───────────
def _tidy_cell(cell: str | None) -> str:
    if not cell:
        return ""
    flat = re.sub(r"\s*\n\s*", " ", cell.strip())
    return re.sub(r"\s*•\s*", "\n• ", flat).strip()


def _headings_on_page(page, titles: dict[str, int]) -> list[tuple[float, str, int]]:
    """페이지에서 목차 제목과 정확히 같은 텍스트 블록을 (y, 제목, 레벨)로. 카테고리 소제목(운전/소음/에러 메시지/냉장 & 냉동 …) 감지."""
    found = []
    for block in page.get_text("blocks"):
        t = block[4].strip()
        if t in titles:
            found.append((block[1], t, titles[t]))
    return sorted(found)


def parse_troubleshooting(doc, toc: list[dict], doc_id: str, start: int, end: int, meta: dict) -> list[dict]:
    """[증상 | 원인 및 해결책] 표를 읽어 증상 1개 = 청크 1개. subsection = 카테고리(목차 L2/L3에서 감지).
    에어컨: '운전', 냉장고: '냉장 & 냉동', 세탁건조기: '세탁기 > 에러 메시지'"""
    chunks: list[dict] = []
    l2: str | None = None
    l3: str | None = None
    current: dict | None = None

    def label() -> str:
        return " > ".join(x for x in (l2 if l2 not in (None, "문제 해결하기") else None, l3) if x) or "문제 해결하기"

    def flush():
        if current and current["parts"]:
            body = f"증상: {current['symptom']}\n" + "\n".join(current["parts"])
            chunks.append(make_chunk(doc_id, "고장 신고 전 확인하기", current["category"], body, tag="symptom", **meta))

    for i in range(start, end):
        page = doc[i]
        titles = {h["title"]: h["level"] for h in toc if h["page_idx"] in (i - 1, i, i + 1)}
        headings = _headings_on_page(page, titles)
        tables = sorted(page.find_tables().tables, key=lambda t: t.bbox[1])
        hi = 0
        for table in tables:
            rows = table.extract()
            if not rows or rows[0][:2] != ["증상", "원인 및 해결책"]:
                continue
            while hi < len(headings) and headings[hi][0] < table.bbox[1]:   # 표 위의 소제목으로 상태 갱신
                _, t, lvl = headings[hi]
                if lvl == 2:
                    l2, l3 = t, None
                else:
                    l3 = t
                hi += 1
            category = label()
            for row in rows[1:]:
                symptom = re.sub(r"\s+", " ", (row[0] or "")).strip()
                cause_solution = _tidy_cell(row[1])
                if not cause_solution:
                    continue
                if symptom and not (current and current["symptom"] == symptom):   # 페이지 넘김으로 증상이 재인쇄되면 이어붙임
                    flush()
                    current = {"symptom": symptom, "category": category, "parts": []}
                if current is None:
                    current = {"symptom": "(증상 미상)", "category": category, "parts": []}
                current["parts"].append(cause_solution)
        while hi < len(headings):   # 표 아래 남은 소제목 → 다음 페이지 표에 적용
            _, t, lvl = headings[hi]
            if lvl == 2:
                l2, l3 = t, None
            else:
                l3 = t
            hi += 1
    flush()
    return chunks


# ── 목차 기반 소제목 분리 (나머지 챕터 공용) ───────────────────────────────────
def looks_like_hidden_header(stripped: str, ahead: list[str]) -> bool:
    """목차에 없는 4단계 소제목: 짧은 제목 줄 뒤 몇 줄 안에 '1' 단독 줄(절차 시작)."""
    return (0 < len(stripped) <= 20 and not stripped[0].isdigit()
            and not stripped.startswith(("•", "-", "~", "*"))
            and not stripped.endswith(("요.", "다.", "니다", "세요", ":", "?")) and "1" in ahead)


# 목차에 없는 규격 표 라벨. LG 드럼세탁기(F21VDSK 등)는 '최대 세탁 용량 19 kg / 21 kg / 24 kg + W/D/H' 표가 굵은 8.8pt 라벨로만 있어
# 직전 소제목('접지할 때 알아두기') 조각 꼬리에 붙었고, "용량이 얼마예요?"가 이 조각을 못 찾아 LLM이 수치를 지어냈다.
SPEC_HEADERS = ("최대 세탁 용량", "최대 건조 용량", "제품 규격", "제품 규격 정보")


def parse_by_toc(page_texts, toc, doc_id, start, end, section, meta: dict,
                 detect_hidden=False, skip_groups=frozenset(), tag="generic", lookahead=4) -> list[dict]:
    chunks: list[dict] = []
    buf: list[str] = []
    group = heading = hidden = None

    def flush():
        body = "\n".join(buf).strip()
        if len(body) > 30 and group not in skip_groups:
            chunks.append(make_chunk(doc_id, section, " > ".join(x for x in (group, heading, hidden) if x), body, tag, **meta))

    for i in range(start, end):
        titles_here = {h["title"]: h["level"] for h in toc if h["page_idx"] in (i, i - 1)}
        lines = strip_page_furniture(page_texts[i]).split("\n")
        for j, line in enumerate(lines):
            stripped = line.strip()
            level = titles_here.get(stripped)
            if level == 2 and stripped == group:          # 그룹 제목이 본문 안에서 소제목으로 재사용된 경우 → 4단계로
                level = None
                if detect_hidden:
                    flush(); buf = []; hidden = stripped
                    continue
            if level == 2:
                flush(); buf = []; group, heading, hidden = stripped, None, None
                continue
            if level == 3:
                flush(); buf = []; heading, hidden = stripped, None
                continue
            if stripped in SPEC_HEADERS:                    # 규격 표 → '제품 규격' 조각으로 분리 (라벨 줄은 본문에 남긴다)
                flush(); buf = [line]; hidden = "제품 규격"
                continue
            if detect_hidden:
                ahead = [lines[k].strip() for k in range(j + 1, min(j + 1 + lookahead, len(lines)))]
                if looks_like_hidden_header(stripped, ahead):
                    flush(); buf = []; hidden = stripped
                    continue
            buf.append(line)
    flush()
    return split_oversized(chunks)


# ── PDF 1개 → 청크 ────────────────────────────────────────────────────────────
def parse_pdf(pdf_path: Path, category: str = "", product_type: str = "", include_legacy: bool = True) -> list[dict]:
    doc_id = pdf_path.stem
    doc = pymupdf.open(str(pdf_path))
    page_texts = extract_page_texts(doc)
    toc = toc_headings(doc)
    section_ranges = locate_sections(page_texts)
    if len(section_ranges) < MIN_CHAPTERS or not toc:
        doc.close()
        if include_legacy:
            return parse_pdf_legacy(pdf_path, category, product_type)      # 구형 템플릿 → 폰트/번호 기반 폴백
        print(f"[chunking] 건너뜀 {doc_id}: 목차 {len(toc)}개, 챕터 {len(section_ranges)}개 — 다른 템플릿(구형)")
        return []
    meta = {"brand": "lg", "category": category, "product_type": product_type}

    chunks: list[dict] = []
    for section, (start, end) in section_ranges.items():
        base = chapter_base(section)
        if base == "고장 신고 전 확인하기":
            mine = parse_troubleshooting(doc, toc, doc_id, start, end, meta)
            mine += parse_by_toc(page_texts, toc, doc_id, start, end, section, meta,
                                 skip_groups={"문제 해결하기", "세탁기", "건조기", "세탁기/건조기 공통"}, tag="diagnosis")
        elif base in HOWTO_BASES:
            mine = parse_by_toc(page_texts, toc, doc_id, start, end, section, meta, detect_hidden=True, tag="howto")
        elif base == "안전을 위해 주의하기":
            mine = parse_by_toc(page_texts, toc, doc_id, start, end, section, meta, tag="safety")
        else:
            mine = parse_by_toc(page_texts, toc, doc_id, start, end, section, meta, skip_groups=SKIP_GROUPS)
        chunks += assign_pages(mine, page_texts, start, end)     # 챕터 범위 안에서만 찾아 다른 챕터의 같은 문장에 안 붙게
    doc.close()
    return chunks


# ── 구형 템플릿 폴백 파서 (목차 북마크가 없는 PDF: RF_S825AW35, RF_S835S31, RF_CA-H17DC) ──────────
# 목차 기반 파싱이 안 되는 PDF는 (1) 굵고 큰 글씨 또는 '7. 냉장실홈바' 같은 번호 매김 줄을 소제목으로 보고
# (2) 소제목 사이 본문을 한 청크로 묶는다. 세로 탭(한 글자씩 세워 쓴 챕터명)과 페이지 번호는 제거하고,
# 띄어쓰기가 빠진 PDF(S825/S835: '올바른냉장고사용방법')는 kiwi로 띄어쓰기를 복원한다.
# 'N. 제목' 소제목: 번호 뒤가 14자 이내의 명사구. 절차 단계('1. 선반의 앞쪽을 잡고 …', '3. 고정부에 걸려서')는
# 길거나 조사·어미로 끝나므로 제외한다.
_NUMBERED_HEADING_RE = re.compile(r"^\d{1,2}\.\s*(\S.{0,13})$")
_STEP_ENDINGS = ("의", "을", "를", "서", "후", "면", "고", "에", "로", "는", "은", "가", "이", "세요", "니다", "하기", "다.", "요.")
_kiwi_spacer = None


def _restore_spacing(text: str) -> str:
    """띄어쓰기가 거의 없는 텍스트(공백 비율 < 3%)만 kiwi로 복원. 그 외는 그대로."""
    global _kiwi_spacer
    letters = [ch for ch in text if not ch.isspace()]
    if not letters or text.count(" ") / max(len(letters), 1) > 0.03:
        return text
    if _kiwi_spacer is None:
        from kiwipiepy import Kiwi
        _kiwi_spacer = Kiwi()
    return "\n".join(_kiwi_spacer.space(line) if len(line) > 8 else line for line in text.split("\n"))


def _legacy_lines(page, allow_unbold: bool = False) -> list[tuple[str, bool]]:
    """페이지 → [(줄 텍스트, 소제목 여부)]. 소제목 = 본문보다 큰(굵은) 짧은 줄, 또는 'N. 제목' 줄.
    한 글자 줄(세로 탭)·숫자만 있는 줄(페이지 번호)은 버린다."""
    sizes = Counter()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if len(s["text"].strip()) > 5:
                    sizes[round(s["size"], 1)] += len(s["text"])
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if len(text) <= 1 or text.isdigit():
                continue
            size = max(s["size"] for s in spans)
            bold = all((s["flags"] & 16) or "Bold" in s["font"] for s in spans)
            m = _NUMBERED_HEADING_RE.match(text)
            numbered = bool(m) and not m.group(1).strip().endswith(_STEP_ENDINGS)
            # 굵은 글씨 문서(LG 구형): 굵고 +3pt.  굵은 글씨를 안 쓰는 문서(삼성 WF21T 등): +1pt 이상의 짧은 한글 제목 줄
            callout = re.sub(r"\s+", "", text) in ("주의", "참고", "경고", "알아두기", "안내")
            bigger = allow_unbold and size >= body_size + 0.8 and len(text) <= 30 and re.search(r"[가-힣A-Za-z]", text) \
                and not callout and not text.endswith(("요.", "다.", "세요", "니다"))
            is_heading = ((bold and size >= body_size + 3 and len(text) <= 40) or bigger) or numbered
            out.append((text, is_heading))
    return out


def parse_pdf_legacy(pdf_path: Path, category: str = "", product_type: str = "", allow_unbold: bool = False) -> list[dict]:
    """목차 없는 PDF용 폴백. allow_unbold: 굵은 글씨로 소제목을 구분하지 않는 문서(삼성 InDesign)면 True —
    본문보다 1pt 이상 큰 짧은 한글 줄을 소제목으로 본다. LG 구형(굵은 글씨 사용)은 False."""
    doc_id = pdf_path.stem
    doc = pymupdf.open(str(pdf_path))
    meta = {"brand": "lg", "category": category, "product_type": product_type}
    # 챕터: 1단계 목차가 있으면 그 페이지 범위, 없으면 문서 전체를 '본문' 하나로
    toc1 = [(t.strip(), pg - 1) for l, t, pg in doc.get_toc() if l == 1 and t.strip() not in ("앞표지", "차례", "뒷표지")]
    if toc1:
        chapters = [(t, p, toc1[i + 1][1] if i + 1 < len(toc1) else len(doc)) for i, (t, p) in enumerate(toc1)]
    else:
        chapters = [("본문", 1, len(doc))]   # 표지 제외
    raw_page_texts = [pg.get_text() for pg in doc]     # 페이지 위치 사후 탐색용 (doc.close() 전에)

    chunks: list[dict] = []
    for section, start, end in chapters:
        buf: list[str] = []
        heading = ""
        tag = "symptom" if any(k in section for k in ("이상", "증상", "고장")) else "generic"

        def flush():
            body = _restore_spacing("\n".join(buf).strip())
            if len(body) > 30:
                chunks.append(make_chunk(doc_id, section, heading, body, tag, **meta))

        for i in range(start, min(end, len(doc))):
            for text, is_heading in _legacy_lines(doc[i], allow_unbold=allow_unbold):
                if text == section:              # 러닝헤더
                    continue
                if is_heading:
                    flush(); buf = []; heading = text
                else:
                    buf.append(text)
        flush()
    doc.close()
    # 너무 짧은 조각(<120자)은 같은 챕터의 직전 청크에 합친다 (절차 번호 줄 등으로 잘게 쪼개진 것 정리)
    merged: list[dict] = []
    for c in chunks:
        if merged and len(c["body"]) < 120 and merged[-1]["section"] == c["section"]:
            prev = merged[-1]
            merged[-1] = make_chunk(prev["doc_id"], prev["section"], prev["subsection"],
                                    prev["body"] + "\n" + (f"[{c['subsection']}] " if c["subsection"] else "") + c["body"],
                                    prev["tag"], **meta)
        else:
            merged.append(c)
    print(f"[chunking] 구형 템플릿 폴백 파서 {doc_id}: 챕터 {len(chapters)}개 → 청크 {len(merged)}개")
    final = split_oversized(merged)
    for section, start, end in chapters:
        assign_pages([c for c in final if c["section"] == section], raw_page_texts, start, end)
    return final


def load_error_code_chunks(json_paths: list[Path] | Path | None = None) -> list[dict]:
    """LG 고객지원 에러코드 JSON(에어컨/냉장고/세탁기) → 1항목 = 1청크. doc_id는 'LG-AC-COMMON' 등 카테고리 공통."""
    if json_paths is None:
        json_paths = ERROR_JSONS
    elif isinstance(json_paths, Path):
        json_paths = [json_paths]
    chunks = []
    for jp in json_paths:
        if not jp.exists():
            print(f"[chunking] 에러코드 JSON 없음: {jp}")
            continue
        brand, category = jp.stem.replace("_errors", "").split("_", 1)   # 'lg_aircon_errors' → lg, aircon
        for product in json.loads(jp.read_text(encoding="utf-8")):
            common_id = product["product_model"].replace("ERRORCODE", "COMMON")   # LG-AC-COMMON, SAMSUNG-WM-COMMON …
            for sec in product["sections"]:
                chunks.append(make_chunk(common_id, "에러코드", sec["title"].strip(), sec["content"].strip(), tag="error_code",
                                         brand=brand, category=category, product_type=CATEGORY_KO.get(category, "")))
    return split_oversized(chunks)


def common_doc_id_for(category: str, brand: str = "lg") -> str:
    """(브랜드, 폴더 카테고리) → 에러코드 공통 문서 id. 'lg','aircon' → 'LG-AC-COMMON', 'samsung','washer' → 'SAMSUNG-WM-COMMON'"""
    code = {"aircon": "AC", "fridge": "RF", "washer": "WM"}.get(category)
    return f"{brand.upper()}-{code}-COMMON" if code else ""


def build_all_chunks(pdf_dir: Path | None = None, categories: tuple[str, ...] = ("aircon", "fridge", "washer"),
                     include_legacy: bool = True) -> list[dict]:
    """LG 매뉴얼 전체(또는 pdf_dir 한 폴더) + 에러코드 JSON → 청크. 노트북/실험 파일의 진입점.
    include_legacy=False면 목차 없는 구형 PDF 3개를 건너뛴다 (exp03 재현용)."""
    ptype = _product_type_map()
    chunks: list[dict] = []
    if pdf_dir is not None:                       # 하위 호환: 폴더 하나만 (예: data/lg/aircon)
        folders = [pdf_dir]
    else:
        folders = [LG_DIR / c for c in categories]
    for folder in folders:
        category = folder.name
        for pdf_path in sorted(folder.glob("*.pdf")):
            model = pdf_path.stem.split("_", 1)[1] if "_" in pdf_path.stem else pdf_path.stem
            chunks += parse_pdf(pdf_path, category, ptype.get(model, CATEGORY_KO.get(category, "")), include_legacy)
    wanted = {folder.name for folder in folders}
    chunks += load_error_code_chunks([p for p in ERROR_JSONS if p.stem.replace("lg_", "").replace("_errors", "") in wanted])
    return chunks


if __name__ == "__main__":
    cs = build_all_chunks()
    print(f"\n총 {len(cs)}개 청크")
    print("태그별:", Counter(c["tag"] for c in cs))
    for doc_id in sorted({c["doc_id"] for c in cs}):
        mine = [c for c in cs if c["doc_id"] == doc_id]
        sym = Counter(c["subsection"] for c in mine if c["tag"] == "symptom")
        print(f"  {doc_id:18s} {mine[0].get('product_type',''):6s} {len(mine):4d}개  증상카테고리: {dict(sym) if sym else '-'}")
