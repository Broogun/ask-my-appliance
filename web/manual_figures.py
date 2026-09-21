"""설명서 PDF의 그림 영역 검출 + 답변 줄과 그림 매칭. 결정론(LLM 없음).

- 그림 검출은 래스터 방식: 페이지를 렌더링해 글자를 지우고 남은 덩어리 중 표/구분선/글상자를 뺀 것.
  (드로잉 사각형은 그림 하나에 붙은 흰색 마스크 도형 때문에 이웃 그림과 합쳐져서 못 쓴다)
- 그림마다 이름표 후보를 만든다: ① 2단 페이지(LG)에서 읽기 순서상 바로 앞 소제목 ② 그림 바로 위로 이어지는 글(단계 설명)
  ③ 2단이 아닌 페이지(삼성 표/행 배치)에서 그림 좌우 같은 높이의 글.
- 답변의 한 줄이 이름표 후보와 4-gram으로 충분히 겹칠 때만 그 그림을 붙인다 - 못 찾으면 안 붙인다(틀린 그림보다 없는 게 낫다).
"""
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
from scipy import ndimage

MIN_W, MIN_H = 40.0, 35.0       # 그림으로 볼 최소 크기(pt)
DILATE = 3                      # 같은 그림의 조각을 잇는 팽창(pt)
TEXT_FRAC_MAX = 0.10            # 덩어리 안 글자 면적 비율 상한 - 넘으면 표/글상자
NGRAM = 4
MIN_SHARED = 4                  # 겹치는 4-gram 최소 개수 - 짧은 문구의 우연 일치 차단
MIN_SHORT_LABEL = 5             # 짧은 이름표는 정규화한 글자 수가 이 이상일 때만 통째 포함 검사
MIN_LABEL_COVER = 0.2           # 긴 이름표 4-gram 중 답변 줄과 겹쳐야 하는 최소 비율
MATCH_THRESHOLD = 0.5           # 겹침 / min(이름표 4-gram 수, 줄 4-gram 수)
MAX_PER_LINE, MAX_TOTAL = 6, 24
ABOVE_MAX_LINES, ABOVE_MAX_GAP, ABOVE_MAX_DIST = 6, 40, 130   # 그림 위 글 수집 한도(pt)
BELOW_MAX_LINES, BELOW_MAX_DIST = 14, 220                      # 그림 아래 범례(부품 이름 목록) 수집 한도
NEAR_LINES = 2                                                 # 그림에 가장 가까운 글줄 수(직전 단계 설명)
PARA_GAP_RATIO = 2.4                                           # 이름표 글줄 사이 간격이 줄 높이의 이 배수를 넘으면 문단 경계
LEGEND_MIN_LINES, LEGEND_MAX_AVG_LEN = 3, 18                   # 범례로 볼 조건: 짧은 줄 3개 이상
LINE_KINDS = ("beside", "near", "head", "above", "near_below", "head_far")   # 답변의 한 줄과 비교하는 이름표 종류(구체적인 것부터). 소제목은 그림별 글 다음 - 한 소제목 아래 그림이 많으면 한 줄에 몰린다
SEM_MIN_SIM, SEM_MARGIN, SEM_MIN_CHARS = 0.80, 0.15, 12   # 의미 유사도 보조 단계: 글자 매칭이 실패한 그림만, 아주 엄격하게
EXT_SEM_MIN_SIM, EXT_SEM_MARGIN = 0.50, 0.05               # 제조사 사진(항목에 이미 묶임)의 자리 고르기: 사진 설명('~하는 모습')은 안내문과 문체가 달라 유사도가 낮다
SEM_KINDS = ("beside", "near", "above", "near_below")
CROP_PAD, CROP_ZOOM = 2, 2.0
# 임베딩은 동작의 방향(끄기/켜기, 넣기/꺼내기)을 구분하지 못해서, 반대 동작 쌍이 그림 글과 답변 줄에 엇갈려 나오면 거부한다
_OPPOSITES = [
    (re.compile(r"끄세요|끈 |끕니다|끄고|끄십|꺼 ?주"), re.compile(r"켜|켠|켤")),
    (re.compile(r"꺼내|빼내|분리|제거|떼"), re.compile(r"넣|조립|장착|결합|끼우|끼워|삽입|부착")),
    (re.compile(r"열[어고]|여세요|엽니다|열기"), re.compile(r"닫")),
    (re.compile(r"올[려리]|높[이여]"), re.compile(r"내[려리]|낮[추게]")),
    (re.compile(r"당[겨기]"), re.compile(r"밀[어고]|누르")),
]


def _opposed(label: str, line: str) -> bool:
    for a, b in _OPPOSITES:
        la, lb, na, nb = bool(a.search(label)), bool(b.search(label)), bool(a.search(line)), bool(b.search(line))
        if (la and not lb and nb and not na) or (lb and not la and na and not nb):
            return True
    return False
_CALLOUTS = {"경고", "주의", "알아두기", "알아 두기", "참고", "안전 주의사항", "안전 경고", "중요", "팁", "Tip", "TIP", "NOTE"}


def _figure_boxes(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    mask = a < 235
    textmask = np.zeros_like(mask)
    for w in page.get_text("words"):
        x0, y0, x1, y1 = int(w[0]) - 1, int(w[1]) - 1, int(w[2]) + 2, int(w[3]) + 2
        textmask[max(y0, 0):y1, max(x0, 0):x1] = True
    mask &= ~textmask
    lab, _ = ndimage.label(ndimage.binary_dilation(mask, iterations=DILATE))
    out = []
    for ys, xs in ndimage.find_objects(lab):
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if w < MIN_W or h < MIN_H or min(w, h) < 8:
            continue
        if ys.start < 50 and h < 60:                # 쪽 머리말
            continue
        if textmask[ys, xs].mean() > TEXT_FRAC_MAX:
            continue
        out.append((float(xs.start + DILATE), float(ys.start + DILATE), float(xs.stop - DILATE), float(ys.stop - DILATE)))  # 팽창분을 빼 실제 잉크 범위로
    # 다른 그림 안에 통째로 들어 있는 조각(확대 삽입 그림 등)은 따로 세지 않는다
    return [b for b in out if not any(o is not b and o[0] - 2 <= b[0] and o[1] - 2 <= b[1] and b[2] <= o[2] + 2 and b[3] <= o[3] + 2 for o in out)]


def _lines(page: pymupdf.Page) -> list[dict]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            sp = [s for s in l["spans"] if s["text"].strip()]
            if not sp:
                continue
            out.append({
                "text": "".join(s["text"] for s in l["spans"]).strip(),
                "rect": pymupdf.Rect(l["bbox"]),
                "size": max(s["size"] for s in sp),
                "bold": all((s["flags"] & 16) or "Bold" in s["font"] for s in sp),
            })
    return out


_LONE_TOKEN = re.compile(r"(?<![0-9A-Za-z가-힣])[0-9A-Za-z]{1,2}(?![0-9A-Za-z가-힣])")


def _norm(s: str) -> str:
    # 범례의 ❶❷ 같은 번호 기호가 PDF 폰트 때문에 'a','b' 로 추출돼 이름 사이에 끼므로, 홀로 떨어진 1~2자 영숫자는 뺀다
    return re.sub(r"[^0-9A-Za-z가-힣]", "", _LONE_TOKEN.sub(" ", s))


def _grams(s: str) -> set[str]:
    n = _norm(s)
    return {n[i:i + NGRAM] for i in range(len(n) - NGRAM + 1)}


def _score(label_grams: set[str], line_grams: set[str], min_cover: float = 0.0) -> float:
    shared = len(label_grams & line_grams)
    if shared < MIN_SHARED or shared / len(label_grams) < min_cover:
        return 0.0
    return shared / min(len(label_grams), len(line_grams))


def _text_run(lines: list[dict], fig: tuple, others: list[tuple], up: bool, head_ids: set[int]) -> list[str]:
    """그림 바로 위(up) 또는 아래로 이어지는 글줄들을 읽는 순서(위→아래)로. 같은 열만, 간격이 벌어지거나
    그림 사이에 다른 그림이 끼거나 소제목을 만나면 멈춘다(위쪽은 그 소제목까지 포함, 아래쪽은 제외)."""
    x0, y0, x1, y1 = fig
    col_lines = [l for l in lines if l["rect"].x1 > x0 - 20 and l["rect"].x0 < x1 + 20]
    if up:
        cand = sorted((l for l in col_lines if l["rect"].y1 <= y0 + 3), key=lambda l: -l["rect"].y1)
        edge, max_lines, max_dist = y0, ABOVE_MAX_LINES, ABOVE_MAX_DIST
    else:
        cand = sorted((l for l in col_lines if l["rect"].y0 >= y1 - 3), key=lambda l: l["rect"].y0)
        edge, max_lines, max_dist = y1, BELOW_MAX_LINES, BELOW_MAX_DIST
    picked = []
    for l in cand:
        r = l["rect"]
        gap = (edge - r.y1) if up else (r.y0 - edge)
        dist = (y0 - r.y1) if up else (r.y0 - y1)
        if gap > ABOVE_MAX_GAP or dist > max_dist or len(picked) >= max_lines:
            break
        if picked and gap > PARA_GAP_RATIO * r.height:
            break                                   # 줄 간격이 갑자기 벌어지면 다른 문단/상자 - 앞 블록 글이 이름표에 섞이는 것 방지
        if l["text"] in _CALLOUTS:
            break                                   # '주의/알아두기' 상자 제목은 경계
        lo, hi = (r.y1, edge) if up else (edge, r.y0)
        if any(o[1] >= lo - 2 and o[3] <= hi + 2 and o[2] > r.x0 and o[0] < r.x1 for o in others):
            break                                   # 이 줄과 그림 사이에 다른 그림이 통째로 끼어 있다
        if id(l) in head_ids and not up:
            break
        picked.append(l["text"])
        if id(l) in head_ids:
            break
        edge = r.y0 if up else r.y1
    return picked[::-1] if up else picked


@lru_cache(maxsize=512)
def analyze_page(pdf_path: str, page_no: int) -> tuple[dict, ...]:
    """[{box, labels:[str]}] - 그림 순서는 (열, y)로 고정이라 k 번호가 요청 간에 안정적이다."""
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_no <= len(doc):
            return ()
        page = doc[page_no - 1]
        boxes = _figure_boxes(page)
        if not boxes:
            return ()
        lines = _lines(page)
        mid = page.rect.width / 2
        def in_fig(l):
            r = l["rect"]
            return any(b[0] - 3 <= r.x0 and r.x1 <= b[2] + 3 and b[1] - 3 <= r.y0 and r.y1 <= b[3] + 3 for b in boxes)
        free = [l for l in lines if not in_fig(l)]          # 그림 안의 글자(부품 이름표 등)는 열 판정에서 뺀다
        L = [l for l in free if l["rect"].x1 <= mid + 12]
        R = [l for l in free if l["rect"].x0 >= mid - 12]
        spread = lambda xs: max(l["rect"].y1 for l in xs) - min(l["rect"].y0 for l in xs)
        # 진짜 2단은 양쪽 글이 세로로 넓게 퍼져 있다 - 한쪽이 좁은 구간에 몰려 있으면(그림 옆 캡션 등) 2단이 아니다
        two_col = len(L) >= 6 and len(R) >= 6 and spread(L) >= 250 and spread(R) >= 250
        col = lambda x0, x1: 0 if (x0 + x1) / 2 < mid else 1

        body_size = Counter()
        for l in lines:
            body_size[round(l["size"], 1)] += len(l["text"])
        body = body_size.most_common(1)[0][0] if body_size else 9.0
        heads = [l for l in lines if l["bold"] and l["size"] >= body * 1.12 and len(l["text"]) <= 32
                 and l["rect"].y0 > 50 and l["text"] not in _CALLOUTS and not re.fullmatch(r"[\d\W]+", l["text"])]
        head_ids = {id(h) for h in heads}

        ordered = sorted(boxes, key=lambda b: (col(b[0], b[2]) if two_col else 0, b[1], b[0]))
        figs = []
        for fig in ordered:
            x0, y0, x1, y1 = fig
            labels = []
            owner = None
            if two_col:
                key = (col(x0, x1), y0)
                cand = [h for h in heads if (col(h["rect"].x0, h["rect"].x1), h["rect"].y0) < key]
                owner = max(cand, key=lambda h: (col(h["rect"].x0, h["rect"].x1), h["rect"].y0)) if cand else None
                if owner:
                    labels.append(("head", owner["text"]))
            beside = ""
            if not two_col or owner is None:
                adj = [l for l in lines
                       if min(l["rect"].y1, y1 + 4) - max(l["rect"].y0, y0 - 4) >= 0.4 * l["rect"].height
                       and ((l["rect"].x0 >= x1 - 5 and l["rect"].x0 - x1 < 320) or (l["rect"].x1 <= x0 + 5 and x0 - l["rect"].x1 < 320))
                       and (not two_col or col(l["rect"].x0, l["rect"].x1) == col(x0, x1))]   # 2단이면 다른 열의 글은 옆 글이 아니다
                beside = " ".join(l["text"] for l in sorted(adj, key=lambda l: (l["rect"].y0, l["rect"].x0)))
                if beside:
                    labels.append(("beside", beside))
            others = [o for o in ordered if o is not fig]
            has_above = False
            if not beside:
                above = _text_run(lines, fig, others, True, head_ids)
                has_above = len(_grams(" ".join(above))) >= 8       # 쪽 머리말 정도는 위쪽 설명으로 치지 않는다
                if above:
                    labels.append(("near", " ".join(above[-NEAR_LINES:])))
                    labels.append(("above", " ".join(above)))
            if not two_col:                         # 2단이 아닌 페이지는 그림별 글이 더 구체적이라 소제목은 마지막 후보
                above_heads = [h for h in heads if h["rect"].y1 <= y0 + 3]
                if above_heads:
                    labels.append(("head_far", max(above_heads, key=lambda h: h["rect"].y0)["text"]))
            below = _text_run(lines, fig, others, False, head_ids)
            is_legend = len(below) >= LEGEND_MIN_LINES and sum(map(len, below)) / len(below) <= LEGEND_MAX_AVG_LEN
            if is_legend:
                labels.append(("below", " ".join(below)))          # 부품 이름 범례처럼 짧은 줄이 여러 개 - 답변의 목록 블록과 비교
            elif below and not beside and not has_above:
                labels.append(("near_below", " ".join(below[:NEAR_LINES + 1])))   # 설명이 그림 아래에 오는 배치(위에 설명이 없을 때만)
            figs.append({"box": fig, "labels": tuple(labels)})
        return tuple(figs)


def render_figure(pdf_path: str, page_no: int, k: int, out: Path) -> bool:
    figs = analyze_page(str(pdf_path), page_no)
    if not 0 <= k < len(figs):
        return False
    x0, y0, x1, y1 = figs[k]["box"]
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        clip = pymupdf.Rect(x0 - CROP_PAD, y0 - CROP_PAD, x1 + CROP_PAD, y1 + CROP_PAD) & page.rect
        pix = page.get_pixmap(matrix=pymupdf.Matrix(CROP_ZOOM, CROP_ZOOM), clip=clip)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.{id(pix):x}.tmp")
    pix.save(str(tmp), output="png")
    tmp.replace(out)
    return True


def split_lines(content: str) -> list[str]:
    """프런트(app.js)의 content.split('\\n')와 똑같이 나눠야 줄 번호가 맞는다."""
    return content.split("\n")


def _step_text(line: str) -> str:
    return re.sub(r"^\s*(\d+[.)]|[-•*])\s*", "", line.replace("**", ""))


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", _LONE_TOKEN.sub(" ", _step_text(t))).strip(" •-	﻿")


def _semantic_hit(labels, lines: list[str], embed, min_sim: float = SEM_MIN_SIM, min_margin: float = SEM_MARGIN):
    cand = [(i, _clean(l)) for i, l in enumerate(lines)]
    cand = [(i, t) for i, t in cand if len(t) >= SEM_MIN_CHARS]
    labs = [(kind, _clean(t)) for kind, t in labels if kind in SEM_KINDS and len(_clean(t)) >= SEM_MIN_CHARS]
    if not cand or not labs:
        return None
    vecs = embed([t for _, t in cand] + [t for _, t in labs])
    L, B = vecs[:len(cand)], vecs[len(cand):]
    sims = (L @ B.T).max(axis=1)                            # 줄별로 이름표 후보 중 최대
    order = np.argsort(-sims)
    top = float(sims[order[0]])
    margin = top - (float(sims[order[1]]) if len(order) > 1 else 0.0)
    if top < min_sim or margin < min_margin:
        return None
    bi = int(order[0]); best_lab = int(np.argmax(L[bi] @ B.T))
    if _opposed(labs[best_lab][1], cand[bi][1]):
        return None
    return (cand[bi][0], "", "semantic", labs[best_lab][1])


def _external_figures(sources: list[dict], lines: list[str], line_grams: list[set], embed) -> list[tuple]:
    """소스에 붙은 제조사 사진(에러코드 항목 등) -> [(줄, 그림, kind)]. 대표 사진(thumb)은 답변의 첫 본문 줄 뒤,
    본문 그림(body)은 설명(alt)이 답변 줄과 글자 또는 의미로 맞을 때만 붙인다."""
    out, seen, seen_thumb_alt = [], set(), set()
    first = next((i for i, l in enumerate(lines) if l.strip() and not ("사용 중지" in l and "전원 차단" in l)), None)
    for s in sources:
        for im in s.get("images") or []:
            if im["url"] in seen:
                continue
            seen.add(im["url"])
            alt = im.get("alt", "")
            fig = {"url": im["url"], "caption": alt[:40] or "제조사 사진"}
            if im.get("credit"):
                fig["credit"] = {"text": im["credit"], "url": im.get("page")}
            if im.get("role") == "thumb":
                if first is not None and alt not in seen_thumb_alt:     # 같은 코드 표시부 사진이 기종별로 여러 장이면 하나만
                    seen_thumb_alt.add(alt)
                    out.append((first, fig, "ext-thumb"))
                continue
            lg = _grams(alt)
            best_i, best = max(((i, _score(lg, g, MIN_LABEL_COVER)) for i, g in enumerate(line_grams)), key=lambda x: x[1], default=(-1, 0.0))                 if len(lg) >= MIN_SHARED else (-1, 0.0)
            if best >= MATCH_THRESHOLD:
                out.append((best_i, fig, "ext-body"))
            elif embed is not None:
                h = _semantic_hit([("near", alt)], lines, embed, EXT_SEM_MIN_SIM, EXT_SEM_MARGIN)   # 이미 같은 항목에 묶인 사진이라 자리만 고르면 됨 - 기준 완화
                if h:
                    out.append((h[0], fig, "ext-body-sem"))
    return out


def figures_for_answer(content: str, sources: list[dict], pdf_path_of, debug: bool = False, embed=None) -> list[dict]:
    """[{line, figures:[{url, caption}]}]. 근거로 쓰인 페이지의 그림 중 이름표가 답변의 한 줄과 겹치는 것만."""
    lines = split_lines(content)
    line_grams = [_grams(_step_text(l)) for l in lines]
    line_norm = [_norm(_step_text(l)) for l in lines]
    seen, cands = set(), []
    for s in sources:
        doc_id, ps = s.get("doc_id"), s.get("page_start")
        if not (doc_id and ps and s.get("pdf_url")):
            continue
        path = pdf_path_of(doc_id)
        if path is None:
            continue
        for n in range(ps, min(s.get("page_end") or ps, ps + 2) + 1):
            if (doc_id, n) in seen:
                continue
            seen.add((doc_id, n))
            for k, f in enumerate(analyze_page(str(path), n)):
                cands.append((doc_id, n, k, f))
    blocks, start = [], None                    # 빈 줄로 끊기는 연속 줄 묶음(번호 목록 등) - 범례 비교용
    for i, l in enumerate(lines + [""]):
        if l.strip() and start is None:
            start = i
        elif not l.strip() and start is not None:
            blocks.append((start, i - 1, _grams("".join(_step_text(x) for x in lines[start:i]))))
            start = None
    scored = []
    for doc_id, n, k, f in cands:
        hit = None
        for kind, label in f["labels"]:             # 구체적인 이름표부터, 처음 통과한 것을 쓴다
            if kind not in LINE_KINDS:
                continue
            lg = _grams(label)
            if len(lg) < MIN_SHARED:
                nl = _norm(label)                   # 짧은 이름표('풀 먹인 의류')는 답변 줄에 그 문구가 그대로 있을 때만
                hit_i = next((i for i, ln in enumerate(line_norm) if len(nl) >= MIN_SHORT_LABEL and nl in ln), None)
                if hit_i is not None:
                    hit = (hit_i, label if kind in ("head", "head_far") else "", kind + "/short", label)
                    break
                continue
            # 긴 이름표(문단/범례)는 그 글의 상당 부분이 답변 줄에 실제로 겹쳐야 한다 - 짧은 줄이 긴 글의 일부라서 점수 1.0이 되는 것 차단
            cover = 0.0 if kind in ("head", "head_far") else MIN_LABEL_COVER
            best_i, best = max(((i, _score(lg, g, cover)) for i, g in enumerate(line_grams)), key=lambda x: x[1], default=(-1, 0.0))
            if best >= MATCH_THRESHOLD:
                hit = (best_i, label if kind in ("head", "head_far") else "", kind, label)   # 설명서 문장 조각은 캡션으로 쓰기에 지저분하다
                break
        if hit is None:                             # 범례(부품 이름 목록)는 한 줄이 아니라 목록 블록 전체와 비교, 블록 끝에 붙인다
            for kind, label in f["labels"]:
                lg = _grams(label)
                if kind != "below" or len(lg) < MIN_SHARED:
                    continue
                end, best = max(((e, _score(lg, bg)) for _, e, bg in blocks), key=lambda x: x[1], default=(-1, 0.0))
                if best >= MATCH_THRESHOLD:
                    hit = (end, "", "below-block", label)
                    break
        if hit is None and embed is not None:       # 글자 매칭 실패 - 엄격한 의미 유사도 보조
            hit = _semantic_hit(f["labels"], lines, embed)
        if hit:
            scored.append((hit[0], doc_id, n, k, hit[1], hit[2], hit[3]))
    by_line: dict[int, list[dict]] = {}
    total = 0
    for i, doc_id, n, k, label, kind, full in sorted(scored):
        if total >= MAX_TOTAL or len(by_line.get(i, [])) >= MAX_PER_LINE:
            continue
        fig = {"url": f"/api/manuals/{doc_id}/page/{n}/fig/{k}.png", "caption": label[:40] or "설명서 그림"}
        if debug:
            fig["kind"] = kind
            fig["label_full"] = full
        by_line.setdefault(i, []).append(fig)
        total += 1
    for i, fig, kind in _external_figures(sources, lines, line_grams, embed):
        if total >= MAX_TOTAL or len(by_line.get(i, [])) >= MAX_PER_LINE:
            continue
        if debug:
            fig["kind"] = kind
        by_line.setdefault(i, []).append(fig)
        total += 1
    return [{"line": i, "figures": figs} for i, figs in sorted(by_line.items())]
