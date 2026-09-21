"""제조사 고객지원 에러코드 페이지의 사진을 받아 우리 에러코드 항목(error_entries)에 연결한다.

- 이미지는 data/error_images/ (gitignore)에 로컬 캐시만 한다 - 저장소에 올리지 않는다. 각자 이 스크립트로 받는다.
- 이미지 저작권은 제조사에 있다: 개인 포트폴리오/학습용, 화면에는 항상 출처 링크를 표시한다(web/manual_figures 의 credit).
- 요청은 페이지 몇 건 + 이미지, 요청 사이 0.4초. robots.txt 확인 결과 /support 는 차단 규칙 없음(2026-09-22).

LG 고객지원 페이지는 세 가지 구조다:
  tabs    코드별 탭 패널(#cs-tabpanel-NN)과 옵션 목록이 1:1 (세탁기 드럼/통돌이) - 패널 안의 그림이 그 코드의 조치 그림
  aircon  페이지 위쪽 대표 사진(alt 가 코드 표기) + 본문 그림(앞 문단의 CH 코드로 연결)
  thumbs  위쪽 코드 표시부 사진만(alt 가 코드 그대로, 냉장고) - 본문 그림은 alt 만으로 코드를 알 수 없어 제외

사용법: python scripts/crawl_error_images.py            (PAGES 전부)
        python scripts/crawl_error_images.py --dry-run  (내려받지 않고 항목 연결만 출력)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "siyeon" / "rdb"))
from build_db import code_variants  # noqa: E402

DB_PATH = ROOT / "data" / "appliance.sqlite"
OUT_DIR = ROOT / "data" / "error_images"
INDEX_PATH = OUT_DIR / "index.json"
UA = "Mozilla/5.0 (compatible; appliance-rag-portfolio/0.1; personal non-commercial study)"
DELAY = 0.4
LG = "LG전자 고객지원"
_SUP = "https://www.lge.co.kr/support/solutions-"

PAGES = [
    {"brand": "lg", "category": "aircon", "variant": "aircon", "mode": "aircon", "credit": LG, "url": _SUP + "20154389991658"},
    {"brand": "lg", "category": "washer", "variant": "drum", "mode": "tabs", "credit": LG, "url": _SUP + "1001261"},
    {"brand": "lg", "category": "washer", "variant": "top", "mode": "tabs", "credit": LG, "url": _SUP + "20152965504689"},
    {"brand": "lg", "category": "fridge", "variant": "fridge", "mode": "thumbs", "credit": LG, "url": _SUP + "20153810464793"},
    {"brand": "lg", "category": "fridge", "variant": "kimchi", "mode": "thumbs", "credit": LG, "url": _SUP + "20153812732020"},
]


def _entry_for_tokens(db: sqlite3.Connection, brand: str, category: str, tokens: list[str], aircon: bool = False) -> str | None:
    """코드 토큰들 -> 항목의 chunk_id (첫 토큰부터, 여럿이면 entry_id 가장 작은 것). 에어컨 숫자는 CH 접두."""
    for t in tokens:
        variants = {v for v in code_variants(t)} | {t.upper()}
        if aircon and t.isdigit():
            variants |= {f"CH{int(t)}", f"CH{t}"}
        row = db.execute(
            f"SELECT e.chunk_id FROM error_codes c JOIN error_entries e USING (entry_id) "
            f"WHERE c.brand=? AND c.category=? AND c.code_norm IN ({','.join('?' * len(variants))}) ORDER BY e.entry_id LIMIT 1",
            (brand, category, *variants)).fetchone()
        if row:
            return row[0]
    return None


def _tokens(label: str) -> list[str]:
    """'dE,dE1,dE2 [도어 안 닫힘 문제]' -> ['dE','dE1','dE2'], 'tCL/tC [..]' -> ['tCL','tC']"""
    head = label.split("[")[0]
    return re.findall(r"[A-Za-z0-9]+", head)


def _ext(content: bytes) -> str:
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if content[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if content[:4] == b"GIF8":
        return ".gif"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    return ""


def _classify(page: dict, soup: BeautifulSoup, img, alt: str, opts: list[str]) -> tuple[str, list[str]] | None:
    """이미지 하나의 (role, 코드 토큰들). 버릴 것은 None."""
    mode = page["mode"]
    if mode == "tabs":
        panel = img.find_parent(id=re.compile(r"^cs-tabpanel-\d+$"))
        if panel is None:                           # 페이지 위쪽 대표 사진: alt 가 옵션과 같은 코드 표기
            return "thumb", _tokens(alt)
        if re.match(r"^(?:표시창에 )?\S+(?: 표시)? (?:에러|코드)", alt) or alt.endswith("에러 표시"):
            return None                             # 코드가 표시된 모습(대표 사진과 중복)
        idx = int(panel["id"].rsplit("-", 1)[1]) - 1
        return ("body", _tokens(opts[idx])) if 0 <= idx < len(opts) else None
    if mode == "aircon":
        m = re.match(r"\s*(\d{1,3})\b", alt)
        if m:
            return "thumb", [m.group(1)]
        prev = img.find_previous(string=re.compile(r"CH\s?\d+"))
        mm = re.search(r"CH\s?(\d+)", prev) if prev else None
        return ("body", [mm.group(1)]) if mm else None
    if mode == "thumbs":                            # alt 가 코드 그대로(IF, FF, rF, CO ...)인 대표 사진만
        return ("thumb", [alt]) if re.fullmatch(r"[A-Za-z]{1,3}\d?", alt) else None
    return None


def crawl(page: dict, dry: bool) -> dict[str, list[dict]]:
    brand, category, variant = page["brand"], page["category"], page["variant"]
    html = requests.get(page["url"], headers={"User-Agent": UA}, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    opts = [re.sub(r"\s+", " ", o.get_text(" ", strip=True)) for o in soup.find_all("option")]
    opts = [o for o in opts if o]
    if page["mode"] == "tabs" and len(soup.select("div[id^=cs-tabpanel-]")) != len(opts):
        print(f"  ! 패널 수와 옵션 수가 다르다 - 페이지 구조가 바뀐 듯. 건너뜀")
        return {}
    db = sqlite3.connect(str(DB_PATH))
    folder = OUT_DIR / f"{brand}_{category}"
    result: dict[str, list[dict]] = {}
    seen: set[str] = set()
    stats = {"linked": 0, "no_entry": 0, "skipped": 0}
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if "downloadFile" not in src:
            continue
        src = urljoin(page["url"], src.replace(".com:/", ".com/"))
        if src in seen:
            continue
        seen.add(src)
        alt = re.sub(r"\s+", " ", img.get("alt") or "").strip()
        cls = _classify(page, soup, img, alt, opts)
        if cls is None:
            stats["skipped"] += 1
            continue
        role, tokens = cls
        chunk_id = _entry_for_tokens(db, brand, category, tokens, aircon=page["mode"] == "aircon")
        if not chunk_id:
            stats["no_entry"] += 1
            continue
        stats["linked"] += 1
        if role == "thumb" or stats["linked"] <= 3:
            print(f"  [{role:5s}] {tokens} -> {chunk_id[:44]} | alt={alt[:36]!r}")
        name = re.sub(r"[^A-Za-z0-9_\-]", "", re.search(r"fileId=([^&]+)", src).group(1))
        entry = {"alt": alt, "role": role, "variant": variant, "page": page["url"], "credit": page["credit"], "src": src}
        if not dry:
            existing = next(folder.glob(f"{name}.*"), None) if folder.exists() else None
            if existing is None:
                time.sleep(DELAY)
                r = requests.get(src, headers={"User-Agent": UA, "Referer": page["url"]}, timeout=30)
                ext = _ext(r.content)
                if r.status_code != 200 or not ext:
                    print(f"    ! 이미지 받기 실패 status={r.status_code} ext={ext!r}")
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                existing = folder / f"{name}{ext}"
                existing.write_bytes(r.content)
            entry["file"] = f"{brand}_{category}/{existing.name}"
        else:
            entry["file"] = f"{brand}_{category}/{name}.?"
        result.setdefault(chunk_id, []).append(entry)
    print(f"  -> 연결 {stats['linked']}장 / 항목 없는 코드 {stats['no_entry']}장 / 중복·표시부 {stats['skipped']}장")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    index: dict[str, list[dict]] = {}
    for page in PAGES:                              # 페이지마다 새로 모아 전체를 다시 쓴다(옛 항목 잔재 방지)
        print(f"{page['brand']}/{page['category']}/{page['variant']}: {page['url']}")
        for cid, items in crawl(page, args.dry_run).items():
            index.setdefault(cid, []).extend(items)
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"항목 {len(index)}개, 이미지 {sum(len(v) for v in index.values())}장" + (" (dry-run)" if args.dry_run else f" -> {INDEX_PATH}"))


if __name__ == "__main__":
    sys.exit(main())
