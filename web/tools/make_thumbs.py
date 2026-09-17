"""설명서 PDF에서 제품 그림(선화)을 잘라 썸네일을 만든다 — 실제 제품 사진이 없을 때의 대체 그림.

    python web/tools/make_thumbs.py            # 60개 → web/static/img/models/auto/{manual_id}.png + 검수용 sheet.png
    python web/tools/make_thumbs.py AC_FQ18GC1EHN WM_F21VDSK   # 일부만

프론트는 web/static/img/models/{manual_id}.jpg|png (팀이 넣는 실제 사진) → auto/{manual_id}.png (이 스크립트) → 제품군 아이콘 순으로 보여준다.
자동 크롭은 완벽하지 않다. 만든 뒤 web/static/img/models/auto/sheet.png 을 열어 이상한 것은 지우면 아이콘으로 폴백된다.

방법: 목차에서 '각 부분의 이름 / 모습 / 명칭 / 살펴보기' 페이지를 찾고, 그 페이지의 벡터 드로잉·이미지 중 가장 큰 덩어리(인접한 것 병합)를
잘라낸다. 페이지의 85% 이상을 차지하는 덩어리(= 페이지 전체)는 버리고 다음 후보를 본다.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "static" / "img" / "models" / "auto"
DB = ROOT / "data" / "appliance.sqlite"
SIZE = 240

# 우선순위 순 — 앞 패턴이 있는 목차 항목을 먼저 쓴다
PATTERNS = [re.compile(r"각 부분의 (이름|명칭)|외부 명칭|제품 앞면|실내기 살펴보기|세탁기의 모습|냉장고의 모습|에어컨의 모습|건조기의 모습|스타일러의 모습|모습 살펴보기"),
            re.compile(r"모습|명칭"), re.compile(r"살펴보기")]


def manuals() -> list[tuple[str, Path]]:
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT manual_id, file_path FROM manuals").fetchall()
    con.close()
    out = []
    for mid, rel in rows:
        p = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        if not p.exists():   # DB에 경로가 없거나 옮겨졌으면 data/ 아래에서 파일명으로 찾는다
            hits = list((ROOT / "data").rglob(f"{mid}.pdf"))
            p = hits[0] if hits else None
        if p:
            out.append((mid, p))
    return out


def find_page(doc) -> int:
    toc = [(t.strip(), pg - 1) for l, t, pg in doc.get_toc() if pg > 0]
    for pat in PATTERNS:
        for t, pg in toc:
            if pat.search(t):
                return pg
    return min(3, len(doc) - 1)   # 목차 없으면 4페이지


def crop_rect(page) -> pymupdf.Rect | None:
    rects = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    rects += [d["rect"] for d in page.get_drawings() if d["rect"].width > 30 and d["rect"].height > 30]
    rects = [r for r in rects if r.is_valid and not r.is_empty]
    if not rects:
        return None
    page_area = page.rect.width * page.rect.height
    # 큰 것부터 시작해 인접(30pt) 드로잉을 합친 덩어리. 페이지 85%를 넘으면 전체 그림이라 버림
    for seed in sorted(rects, key=lambda r: -(r.width * r.height)):
        union = pymupdf.Rect(seed)
        changed = True
        while changed:
            changed = False
            for r in rects:
                if not union.contains(r) and r.intersects(union + (-30, -30, 30, 30)):
                    union |= r; changed = True
        union &= page.rect
        if union.width * union.height < 0.85 * page_area and union.width > 60 and union.height > 60:
            return union
    return None


def make(mid: str, pdf: Path) -> str:
    doc = pymupdf.open(pdf)
    pg = find_page(doc)
    page = doc[pg]
    clip = crop_rect(page)
    if clip is None:
        return f"{mid:22s} p{pg+1:3d} 그림 못 찾음"
    pix = page.get_pixmap(dpi=110, clip=clip + (-8, -8, 8, 8) & page.rect)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail((SIZE, SIZE))
    canvas = Image.new("RGB", (SIZE, SIZE), "white")
    canvas.paste(img, ((SIZE - img.width) // 2, (SIZE - img.height) // 2))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / f"{mid}.png", optimize=True)
    return f"{mid:22s} p{pg+1:3d} {clip.width:.0f}x{clip.height:.0f}pt"


def sheet():
    files = sorted(OUT.glob("*.png"))
    files = [f for f in files if f.name != "sheet.png"]
    cols = 10; rows = (len(files) + cols - 1) // cols
    W = cols * (SIZE + 10) + 10; H = rows * (SIZE + 30) + 10
    s = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(s)
    for i, f in enumerate(files):
        x, y = 10 + (i % cols) * (SIZE + 10), 10 + (i // cols) * (SIZE + 30)
        s.paste(Image.open(f), (x, y)); d.text((x, y + SIZE + 4), f.stem, fill="black")
    s.save(OUT / "sheet.png")
    return OUT / "sheet.png"


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for mid, pdf in manuals():
        if want and mid not in want:
            continue
        print(make(mid, pdf))
    print("검수용:", sheet())
