"""
PDF 사용설명서를 섹션 단위로 청킹해서 {id, text, metadata} 문서로 변환한다.

두 가지 방식을 제공한다:
  1) extract_sections_from_toc()  - PDF에 내장된 목차(북마크)를 사용 (우선, 더 정확함)
  2) extract_sections_by_font()   - 폰트 크기로 "제목처럼 보이는 줄"을 추정 (TOC가 없을 때 폴백)

TOC 방식이 나은 이유: 실제 LG 매뉴얼로 확인해보니, 폰트 방식은 "R 금지 사항" /
"j 준수 사항" 같은 반복되는 하위 라벨까지 전부 새 섹션으로 쪼개서 조각이 너무
잘게 나뉘었다. TOC에는 이 라벨들이 진짜로 레벨4(가장 깊은 계층)에 있다는 게
드러나 있어서, "여러 번 반복되는 제목(=식별력 없는 라벨)은 새 청크를 만들지
않고 상위 섹션에 합친다"는 규칙을 적용할 수 있다.

2026-09-11 개선 (docs/lg-chunking-strategy.md 점검 결과 반영):
  - 문제해결/고장진단 챕터는 max_depth 제한 없이 리프 레벨까지 강제로 세분화한다
    (RAG 목적상 가장 중요한 챕터인데 기존 로직에서 가장 거칠게 뭉쳐졌음).
  - 그 외 챕터도 텍스트가 너무 길면(3,000자 초과) 한 단계 더 깊은 TOC 레벨로
    재귀적으로 대체해서 쪼갠다 (예: "세탁 코스 및 옵션 사용하기" 19,946자 1섹션
    -> 코스별 레벨3/4 항목으로 분할).
  - 같은 제목이 문서 안에서 중복되면(워시타워형 세탁기+건조기 통합 매뉴얼처럼
    챕터 세트가 통째로 반복되는 경우) 가장 가까운 레벨1 조상 제목을 접두어로
    붙여서 구분한다 (예: "[사용하기 - 세탁기] 제품 청소하기" vs
    "[사용하기 - 건조기] 제품 청소하기").
"""
import re
from collections import Counter

import pymupdf

# --- 폰트 기반(폴백) ---
HEADING_SIZE_THRESHOLD = 12.4
MAX_HEADING_CHARS = 40

# --- 공통 ---
MAX_CHUNK_CHARS = 700
CHUNK_OVERLAP_CHARS = 105  # MAX_CHUNK_CHARS의 15% - 청크 경계에서 문장이 끊기는 것을 완화
GENERIC_TITLE_MIN_FREQ = 3  # 같은 제목이 이 횟수 이상 반복되면 "일반 라벨"로 취급

# --- TOC 적응형 분할 (2026-09-11 추가) ---
MAX_SECTION_CHARS_BEFORE_SPLIT = 3000
TROUBLESHOOTING_KEYWORDS = ["문제 해결", "고장 진단", "고장 신고", "에러 메시지", "트러블"]

# --- 증상 단위 재분할 (2026-09-11 추가) ---
# 문제해결 표(증상->원인->해결책)의 리프 TOC 섹션 경계가 실제 증상 행과 안 맞는 경우가
# 실측으로 확인됐다 (예: "냉장고 안에서 냄새가 나요"의 원인 3개 중 2개는 "부품" 섹션에,
# 나머지 1개는 다음 "소음" 섹션에 걸쳐 있었음). 트러블슈팅 서브트리 안의 연속된 리프
# 섹션들을 원문 그대로 이어붙인 뒤, "증상 문장" 패턴으로 다시 쪼갠다.
SYMPTOM_TABLE_JUNK_LINES = {
    "증상", "원인 및 해결책", "고장 신고 전 확인하기", "고장이 아닙니다",
    "간단한 확인조치", "고장신고 전에", "꼭확인하세요",
}
MAX_SYMPTOM_TITLE_CHARS = 45

# --- 그림 연결 안전장치 (2026-09-11 추가, 실측 버그 대응) ---
# TOC가 없는 문서(예: RF_S825AW35.pdf)는 섹션 간 페이지 간격이 커서(최대 10페이지),
# "다음 섹션 시작 전까지의 모든 그림"을 다 끌어오면 그림 12개(4,221자)가 한 청크에
# 붙는 사고가 실제로 발생했다. 임베딩되는 캡션은 "그림 페이지 == 섹션 시작 페이지"로
# 엄격히 제한하고, 화면 표시용 figures 목록도 개수 상한을 둔다.
MAX_CAPTIONS_EMBEDDED_PER_SECTION = 3
MAX_FIGURES_DISPLAYED_PER_SECTION = 5


def ocr_recommended(pdf_path: str, sample_pages: int = 5) -> dict:
    """이 PDF가 스캔본(텍스트 레이어 없음)이라 OCR이 필요한지 진단한다.
    페이지당 평균 추출 텍스트가 매우 적으면서 이미지가 있으면 스캔본일 가능성이 높다."""
    doc = pymupdf.open(pdf_path)
    n = min(sample_pages, len(doc))
    total_text_len = 0
    total_images = 0
    for i in range(n):
        page = doc[i]
        total_text_len += len(page.get_text().strip())
        total_images += len(page.get_images())

    avg_text = total_text_len / n
    avg_images = total_images / n
    likely_scanned = avg_text < 20 and avg_images >= 1
    return {
        "avg_text_chars_per_page": round(avg_text, 1),
        "avg_images_per_page": round(avg_images, 2),
        "ocr_recommended": likely_scanned,
        "reason": (
            "페이지당 추출 텍스트가 거의 없고 이미지는 있음 -> 스캔본으로 추정, OCR 필요"
            if likely_scanned
            else "텍스트 레이어가 정상적으로 존재함 -> OCR 불필요"
        ),
    }


def _is_heading(line_text: str, max_size: float) -> bool:
    if max_size < HEADING_SIZE_THRESHOLD:
        return False
    if len(line_text) == 0 or len(line_text) > MAX_HEADING_CHARS:
        return False
    if line_text.isdigit():
        return False
    # 한글/영문/숫자가 하나도 없으면 깨진 폰트로 간주 (예: '䤏', '!' 단독)
    if not re.search(r"[가-힣a-zA-Z0-9]", line_text):
        return False
    return True


def _group_lines_by_size(lines: list[str], max_chars: int) -> list[str]:
    """줄 리스트를 max_chars를 넘지 않는 선에서 묶는다 - 한 줄(line) 중간에서는
    절대 자르지 않는다 (한 줄 자체가 max_chars보다 길면 그 줄만 통째로 별도 그룹)."""
    groups: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        if buf and buf_len + len(line) + 1 > max_chars:
            groups.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        groups.append("\n".join(buf))
    return groups


def extract_sections_by_font(pdf_path: str, start_page: int = 0, end_page: int | None = None) -> list[dict]:
    """[{heading, text, page}] - 폰트 크기 기반 (TOC가 없는 PDF용 폴백).

    주의: 이 방식은 문서에 따라 품질이 크게 떨어질 수 있다 (docs/lg-chunking-strategy.md
    "문제 1" 참고 - 번호 매긴 주의사항 목록이 헤딩으로 오인식되는 과다분할 문제는 아직
    남아있음). TOC가 없는 문서를 새로 추가할 때는 이 함수의 결과를 반드시 육안으로
    검수할 것. 다만 반대 방향 문제(본문 전체가 한 섹션으로 뭉쳐져서 뒤에서 기계적으로
    잘리는 것)는 _split_oversized_by_paragraph()로 완화한다.
    """
    doc = pymupdf.open(pdf_path)
    end_page = end_page or len(doc)

    sections = []
    current_heading = "개요"
    current_start_page = start_page + 1
    current_lines: list[str] = []

    def flush():
        if not current_lines:
            return
        groups = _group_lines_by_size(current_lines, MAX_SECTION_CHARS_BEFORE_SPLIT)
        for i, text in enumerate(groups):
            text = text.strip()
            if not text:
                continue
            heading = f"{current_heading} ({i + 1})" if len(groups) > 1 else current_heading
            sections.append({"heading": heading, "text": text, "page": current_start_page})

    for page_num in range(start_page, end_page):
        page_dict = doc[page_num].get_text("dict")
        for block in page_dict["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                max_size = max(s["size"] for s in spans)
                if _is_heading(line_text, max_size):
                    flush()
                    current_heading = line_text
                    current_start_page = page_num + 1
                    current_lines = []
                else:
                    current_lines.append(line_text)
    flush()
    return sections


def _page_text(doc, start_page_1based: int, end_page_1based_exclusive: int) -> str:
    start_idx = max(start_page_1based - 1, 0)
    end_idx = min(max(end_page_1based_exclusive - 1, start_idx + 1), len(doc))
    return "\n".join(doc[p].get_text() for p in range(start_idx, end_idx)).strip()


def _build_toc_tree(toc: list[tuple[int, str, int]]) -> list[dict]:
    """평평한 [(level, title, page), ...] 목차를 레벨 기준 중첩 트리로 바꾼다."""
    root: list[dict] = []
    stack: list[tuple[int, list[dict]]] = [(0, root)]
    for lvl, title, page in toc:
        node = {"level": lvl, "title": title, "page": page, "children": []}
        while len(stack) > 1 and stack[-1][0] >= lvl:
            stack.pop()
        stack[-1][1].append(node)
        stack.append((lvl, node["children"]))
    return root


def _needs_force_expand(title: str) -> bool:
    return any(kw in title for kw in TROUBLESHOOTING_KEYWORDS)


def _select_nodes(nodes: list[dict], max_depth: int, ancestors: tuple[str, ...] = (), force: bool = False) -> list[dict]:
    """트리에서 최종 청크 경계로 쓸 노드들을 고른다. 각 노드에 조상 제목 전체 경로를
    '_ancestors' 로 함께 실어보낸다 (나중에 중복 제목 disambiguation에 사용 - 직속
    부모만으로는 부족한 경우가 있어서, 필요한 만큼 상위로 거슬러 올라갈 수 있게 전체를 남긴다).

    - force=True (트러블슈팅 챕터 산하)면 자식이 있는 한 리프까지 무조건 내려간다.
    - 그 외에는 max_depth 레벨에서 멈추되, 자식이 없으면(그 레벨이 이미 리프면) 그대로 쓴다.
    """
    result: list[dict] = []
    for node in nodes:
        should_force = force or _needs_force_expand(node["title"])
        child_ancestors = ancestors + (node["title"],)
        if should_force and node["children"]:
            result.extend(_select_nodes(node["children"], max_depth=10**6, ancestors=child_ancestors, force=True))
        elif node["level"] < max_depth and node["children"]:
            result.extend(_select_nodes(node["children"], max_depth, ancestors=child_ancestors, force=False))
        else:
            node = dict(node, _ancestors=ancestors)
            result.append(node)
    return result


def _split_by_paragraph(text: str, heading: str, page: int, ancestors: tuple[str, ...]) -> list[dict]:
    """TOC 하위 레벨이 없어서 더 쪼갤 수 없는 큰 섹션을 문단(빈 줄) 단위로 재분할한다.
    문단으로 쪼개도 MAX_SECTION_CHARS_BEFORE_SPLIT 이하로 못 줄이면 원래 섹션을 그대로 반환한다."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) <= 1:
        return [{"heading": heading, "text": text, "page": page, "_ancestors": ancestors}]

    groups: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        if buf and buf_len + len(para) + 2 > MAX_SECTION_CHARS_BEFORE_SPLIT:
            groups.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        groups.append("\n\n".join(buf))

    if len(groups) <= 1:
        return [{"heading": heading, "text": text, "page": page, "_ancestors": ancestors}]

    return [
        {
            "heading": f"{heading} ({i + 1})",
            "text": g,
            "page": page,
            "_ancestors": ancestors,
        }
        for i, g in enumerate(groups)
    ]


def _expand_if_too_long(doc, node: dict, next_page: int, ancestors: tuple[str, ...]) -> list[dict]:
    """섹션 텍스트가 너무 길면 자식 노드로 대체해서 재귀적으로 더 쪼갠다.
    자식이 없으면(더 이상 쪼갤 TOC 정보가 없으면) 문단 단위로 폴백 재분할한다."""
    text = _page_text(doc, node["page"], next_page)
    if len(text) <= MAX_SECTION_CHARS_BEFORE_SPLIT:
        return [{"heading": node["title"], "text": text, "page": node["page"], "_ancestors": ancestors}]

    if not node["children"]:
        # TOC 최심 레벨인데 텍스트가 너무 크면 문단 단위로 폴백
        return _split_by_paragraph(text, node["title"], node["page"], ancestors)

    out: list[dict] = []
    children = node["children"]
    child_ancestors = ancestors + (node["title"],)
    for i, child in enumerate(children):
        child_next = children[i + 1]["page"] if i + 1 < len(children) else next_page
        out.extend(_expand_if_too_long(doc, child, child_next, child_ancestors))
    return out


def extract_sections_from_toc(pdf_path: str, max_depth: int = 2) -> list[dict] | None:
    """[{heading, text, page}] - PDF 내장 TOC(북마크) 기반. TOC가 없으면 None 반환.

    기본은 max_depth까지의 레벨만 청크 경계로 삼는다 (그보다 깊은 레벨 - 예: 반복되는
    "R 금지 사항"/"j 준수 사항" 같은 세부 라벨 - 은 가장 가까운 상위 섹션 본문에
    자연스럽게 포함시킨다). 다만 다음 두 경우는 예외적으로 더 깊게 내려간다:

      1) 제목이 TROUBLESHOOTING_KEYWORDS에 해당하면(문제해결/고장진단 등) -
         이 프로젝트의 목적상 가장 중요한 챕터이므로 리프 레벨까지 전부 세분화한다.
      2) 섹션 텍스트가 MAX_SECTION_CHARS_BEFORE_SPLIT을 넘으면 - 한 단계 더 깊은
         TOC 레벨이 있는 한 재귀적으로 대체해서 쪼갠다.

    또한 같은 제목이 문서 안에서 중복되면(예: 세탁기+건조기 통합 매뉴얼처럼 챕터
    세트가 그대로 반복되는 경우) 가장 가까운 레벨1 조상 제목을 접두어로 붙여서 구분한다.
    """
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    if not toc:
        return None

    tree = _build_toc_tree(toc)
    boundaries = _select_nodes(tree, max_depth=max_depth)
    if not boundaries:
        boundaries = [{"level": lvl, "title": t, "page": p, "children": [], "_ancestors": ()} for lvl, t, p in toc]

    sections: list[dict] = []
    for i, node in enumerate(boundaries):
        next_page = boundaries[i + 1]["page"] if i + 1 < len(boundaries) else len(doc) + 1
        if next_page <= node["page"]:
            next_page = node["page"] + 1
        sections.extend(_expand_if_too_long(doc, node, next_page, node.get("_ancestors", ())))

    sections = [s for s in sections if s["text"]]

    # 같은 페이지에 여러 상위 레벨 제목이 몰려있으면(예: L1 제목 바로 다음 줄이 L2
    # 제목) 앞쪽 항목이 뒤 항목 텍스트에 완전히 포함되는 중복이 생긴다 -> 제거.
    deduped = []
    for i, sec in enumerate(sections):
        if i + 1 < len(sections) and sec["text"] in sections[i + 1]["text"]:
            continue
        deduped.append(sec)

    deduped = _resplit_troubleshooting_by_symptom(deduped)

    _disambiguate_titles(deduped)
    for s in deduped:
        s.pop("_ancestors", None)
        s.pop("_base_heading", None)

    return deduped


def _is_troubleshooting_section(sec: dict) -> bool:
    """이 섹션이 트러블슈팅 서브트리(문제해결/고장진단 등) 산하 리프인지 - 자기 자신의
    제목이나 조상 경로 어디든 TROUBLESHOOTING_KEYWORDS가 있으면 해당."""
    ancestors = sec.get("_ancestors", ())
    return _needs_force_expand(sec["heading"]) or any(_needs_force_expand(a) for a in ancestors)


def _merge_wrapped_lines(text: str) -> list[str]:
    """PDF 텍스트 추출은 한 문장이 화면 폭에 맞춰 여러 줄로 끊겨 나온다(단어 중간은 아님).
    문장 종결 부호(.?!또는 닫는 괄호)로 끝나지 않은 줄은 다음 줄과 이어붙여서 "논리적 줄"
    (실제 문장/항목 단위)로 재구성한다. '•'로 시작하는 줄은 항상 새 논리적 줄의 시작."""
    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    raw_lines = [
        ln for ln in raw_lines
        if ln not in SYMPTOM_TABLE_JUNK_LINES and not ln.isdigit() and len(ln) > 3
    ]
    logical: list[str] = []
    buf = ""
    for ln in raw_lines:
        starts_new = (not buf) or buf.endswith((".", "?", "!", ")")) or ln.startswith("•")
        if starts_new and buf:
            logical.append(buf)
            buf = ln
        else:
            buf = f"{buf} {ln}" if buf else ln
    if buf:
        logical.append(buf)
    return logical


def _is_symptom_line(line: str) -> bool:
    """"증상 문장"으로 보이는 논리적 줄인지 판별하는 휴리스틱: 불릿(원인/해결책)이
    아니고, 질문형(물음표로 끝남)도 아니고, 짧고(45자 이하), 평서문 종결 어미로
    끝난다. 완벽하지 않음(가끔 "고장이 아닙니다" 류의 보충 설명 문장을 오인식) -
    실측 근거는 이 함수를 호출하는 _resplit_troubleshooting_by_symptom() 참고."""
    if line.startswith("•") or line.endswith("?") or len(line) > MAX_SYMPTOM_TITLE_CHARS:
        return False
    return bool(re.search(r"(다|요)\.\)?$", line))


def _split_by_symptom(text: str) -> list[tuple[str, str]]:
    """문제해결 표 텍스트를 증상 단위로 재분할한다. 증상 문장 판별에 실패하면(패턴이
    전혀 안 맞는 문서) 빈 리스트를 반환해서 호출부가 원래 섹션 구성을 그대로 쓰게 한다."""
    logical = _merge_wrapped_lines(text)
    symptom_positions = [
        i for i, ln in enumerate(logical)
        if _is_symptom_line(ln) and i + 1 < len(logical) and logical[i + 1].endswith("?")
    ]
    if not symptom_positions:
        return []
    blocks = []
    for k, start in enumerate(symptom_positions):
        end = symptom_positions[k + 1] if k + 1 < len(symptom_positions) else len(logical)
        title = logical[start]
        body = " ".join(logical[start:end])
        blocks.append((title, body))
    return blocks


def _resplit_troubleshooting_by_symptom(sections: list[dict]) -> list[dict]:
    """트러블슈팅 서브트리 산하의 연속된 리프 섹션들을 그룹으로 묶어 원문을 이어붙인
    뒤, 증상 단위로 재분할한다. TOC 리프 경계(예: '부품'/'소음' 카테고리 탭)가 실제
    증상 행과 안 맞아서 하나의 증상이 두 섹션에 걸쳐 잘리는 문제가 실측으로 확인됐다
    (예: "냉장고 안에서 냄새가 나요"의 원인 3개 중 1개가 다음 섹션으로 넘어감).
    같은 트러블슈팅 그룹으로 묶인 연속 섹션들의 텍스트를 합쳐서 재분할하면 이 경계
    문제가 해소된다. 증상 패턴이 아예 안 잡히면(_split_by_symptom이 빈 리스트 반환)
    해당 그룹은 원래 섹션 그대로 둔다 - 무리하게 쪼개다 망가뜨리는 것보다 안전."""
    result: list[dict] = []
    i = 0
    while i < len(sections):
        if not _is_troubleshooting_section(sections[i]):
            result.append(sections[i])
            i += 1
            continue

        group = [sections[i]]
        j = i + 1
        while j < len(sections) and _is_troubleshooting_section(sections[j]):
            group.append(sections[j])
            j += 1

        combined_text = "\n".join(s["text"] for s in group)
        blocks = _split_by_symptom(combined_text)
        if not blocks:
            result.extend(group)
        else:
            group_ancestors = group[0].get("_ancestors", ())
            group_page = group[0]["page"]
            # 같은 증상이 원본 페이지 경계에서 이어지며 제목이 그대로 반복될 수 있다
            # (예: "냉장고 안에서 냄새가 나요."가 원인 목록 중간에 페이지가 넘어가며
            # 한 번 더 인쇄됨) - 조상 경로로는 구분이 안 되므로 번호를 붙여 구분한다.
            title_counts = Counter(title for title, _ in blocks)
            seen: Counter = Counter()
            for title, body in blocks:
                if title_counts[title] > 1:
                    seen[title] += 1
                    display_title = f"{title} ({seen[title]})"
                else:
                    display_title = title
                result.append({
                    "heading": display_title,
                    "text": body,
                    "page": group_page,
                    "_ancestors": group_ancestors,
                })
        i = j
    return result


def _disambiguate_titles(sections: list[dict]) -> None:
    """같은 heading이 2번 이상 나오면, 구분이 될 때까지 조상 경로를 뒤에서부터
    필요한 만큼만 접두어로 붙인다. 예: 직속 부모만으로 안 갈리면 조부모까지 붙인다
    (워시타워형 매뉴얼의 "옵션별 용도 알아보기"처럼 부모까지 같은 경우가 있어서)."""
    depth = 1
    while True:
        counts = Counter(s["heading"] for s in sections)
        if all(c == 1 for c in counts.values()):
            return
        changed = False
        for s in sections:
            if counts[s["heading"]] > 1:
                ancestors: tuple[str, ...] = s.get("_ancestors", ())
                if len(ancestors) < depth:
                    continue  # 더 올라갈 조상이 없음 - 이 이름끼리는 구분 불가, 그냥 둠
                prefix = " > ".join(ancestors[-depth:])
                base = s["_base_heading"] if "_base_heading" in s else s["heading"]
                s["_base_heading"] = base
                s["heading"] = f"[{prefix}] {base}"
                changed = True
        depth += 1
        if not changed or depth > 6:  # 안전장치: 무한루프 방지
            return


def extract_sections(pdf_path: str) -> list[dict]:
    """TOC가 있으면 TOC 기반, 없으면 폰트 기반으로 자동 선택."""
    sections = extract_sections_from_toc(pdf_path)
    if sections is not None:
        return sections
    return extract_sections_by_font(pdf_path)


MAX_CAPTION_CHARS_PER_FIGURE = 300  # OCR 결과가 노이즈가 섞여 길어질 수 있어 상한을 둔다


def _clean_ocr_caption(text: str) -> str:
    """OCR 결과의 과도한 빈 줄/공백을 정리하고 길이를 제한한다."""
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text[:MAX_CAPTION_CHARS_PER_FIGURE]


def sections_to_documents(
    sections: list[dict],
    product_model: str,
    product_name: str,
    category: str,
    figures_by_page: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """섹션 리스트를 {id, text, metadata} 문서로 변환한다.

    figures_by_page가 주어지면({페이지번호: [{"path", "ocr_text", ...}]}, pipeline/figure_extractor.py
    참고), 각 청크의 페이지 범위 안에 있는 그림 정보를 연결한다:
      - metadata["figures"]: 그림 이미지 파일 경로 목록 (답변 생성 시 "이 그림을 참고하세요"처럼 보여주는 용도)
      - 그림 안에서 OCR로 읽은 텍스트는 metadata에만 두지 않고 **임베딩되는 text 자체에도 포함**한다.
        벡터DB는 텍스트를 숫자로 바꿔 저장하는 구조라, 그림 속 글자(예: 리모컨 버튼 라벨)가
        검색에 걸리려면 그 텍스트가 실제로 임베딩 대상 문자열 안에 있어야 하기 때문이다
        (metadata는 필터링/표시용일 뿐 임베딩되지 않는다).
    """
    figures_by_page = figures_by_page or {}
    docs = []
    for i, sec in enumerate(sections):
        text = sec["text"]

        # 화면 표시용(metadata["figures"]): 이 섹션이 커버하는 페이지 범위 안의 그림을
        # 넉넉히 모으되(다음 섹션 시작 전까지), 개수 상한을 둬서 "관련 그림"이 무의미하게
        # 많이 뜨는 것은 막는다.
        sec_page = sec["page"]
        next_page = sections[i + 1]["page"] if i + 1 < len(sections) else sec_page + 1000
        display_figures = []
        for p in range(sec_page, max(next_page, sec_page + 1)):
            display_figures.extend(figures_by_page.get(p, []))
        figure_paths = [f["path"] for f in display_figures[:MAX_FIGURES_DISPLAYED_PER_SECTION]]

        # 임베딩되는 캡션: 그림의 실제 페이지가 이 섹션의 시작 페이지와 정확히 일치할
        # 때만 병합한다. TOC가 없는 문서는 섹션 간 페이지 간격이 커서(최대 10페이지),
        # 느슨한 범위를 그대로 쓰면 무관한 그림 수십 개가 한 청크에 붙어 벡터가
        # 흐려지는 문제가 실측으로 확인됐다(예: 그림 12개, 청크 4,221자).
        exact_page_figures = figures_by_page.get(sec_page, [])
        captions = [_clean_ocr_caption(f["ocr_text"]) for f in exact_page_figures if f.get("ocr_text")]
        captions = [c for c in captions if c][:MAX_CAPTIONS_EMBEDDED_PER_SECTION]
        caption_block = "\n[그림 설명(OCR)] ".join(captions)

        step = MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS
        chunks = [text[j:j + MAX_CHUNK_CHARS] for j in range(0, len(text), step)] or [text]

        for j, chunk in enumerate(chunks):
            metadata = {
                "product_model": product_model,
                "product_name": product_name,
                "category": category,
                "section_type": "사용설명서",
                "section_title": sec["heading"],
                "page": sec["page"],
            }
            if figure_paths:
                metadata["figures"] = figure_paths

            body = f"[{product_name} / 사용설명서 / {sec['heading']}] {chunk}"
            if caption_block:
                body = f"{body}\n[그림 설명(OCR)] {caption_block}"

            docs.append(
                {
                    "id": f"{product_model}_manual_{i}_{j}",
                    "text": body,
                    "metadata": metadata,
                }
            )
    return docs


def build_documents_from_pdf(
    pdf_path: str,
    product_model: str,
    product_name: str,
    category: str,
    extract_figures: bool = False,
    figures_out_dir: str | None = None,
) -> list[dict]:
    """PDF 하나를 {id, text, metadata} 문서 리스트로 변환한다.

    extract_figures=True면 pipeline/figure_extractor.py로 그림을 같이 추출해서
    각 청크의 metadata에 연결한다 (figures_out_dir 필수).
    """
    sections = extract_sections(pdf_path)

    figures_by_page = None
    if extract_figures:
        if not figures_out_dir:
            raise ValueError("extract_figures=True면 figures_out_dir을 지정해야 합니다.")
        from pipeline.figure_extractor import extract_figures as _extract_figures

        figures_by_page = _extract_figures(pdf_path, figures_out_dir)

    return sections_to_documents(sections, product_model, product_name, category, figures_by_page)


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    diag = ocr_recommended(path)
    print(f"OCR 필요 진단: {diag}")
    sections = extract_sections(path)
    print(f"총 {len(sections)}개 섹션 추출")
    for s in sections[:15]:
        print(f"- p{s['page']} [{s['heading']}] ({len(s['text'])}자) {s['text'][:40]!r}")
