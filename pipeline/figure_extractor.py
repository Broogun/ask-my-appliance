"""
PDF 사용설명서 안의 그림을 추출해서, 나중에 사용자가 사용법을 물어봤을 때 텍스트
답변과 함께 관련 그림도 보여줄 수 있게 한다.

전역 중복 제거 (2026-09-11 추가): 모델 여러 개가 사실상 같은 매뉴얼을 쓰는 경우(색상/용량
변형 시리즈)가 실제로 확인됐다 (삼성 매뉴얼 수집 때 여러 모델이 완전히 같은 CttFileID를
공유하는 걸 봤음). 그림 하나하나를 SHA256으로 해시해서 `data/manual_figures/_global_hash_cache.json`
에 전역으로 캐싱한다 - 다른 모델 문서에서 바이트가 완전히 같은 그림이 또 나오면 OCR을
다시 돌리지 않고 캐시된 결과를 재사용한다. 지금은 로컬이라 절약되는 게 시간뿐이지만,
나중에 OpenAI Vision 같은 유료 API로 바꾸면 이게 곧 비용 절감이 된다.

중요한 발견 (2026-09-11): LG 매뉴얼의 다이어그램(리모컨 그림, 버튼 설명, 설치 방법
그림 등)은 거의 전부 임베디드 래스터 이미지가 아니라 **벡터 드로잉**으로 그려져 있다.
실제로 확인해보니 83페이지 매뉴얼 한 개에서 임베디드 이미지가 있는 페이지는 2개뿐이고,
벡터 드로잉이 있는 페이지는 76개였다. 그래서 `page.get_images()`로 이미지만 뽑는
방식으로는 실제 다이어그램을 거의 못 잡는다.

대신 페이지별 "드로잉 잉크 비율"(벡터 도형들의 bbox 면적 합 / 페이지 면적)을 계산해서,
이 비율이 높은 페이지(=실제 삽화가 있는 페이지)만 페이지 전체를 이미지로 렌더링한다.
같은 문서에서 실측한 값: 리모컨 그림 페이지는 0.31~1.11, 법률/경고 텍스트 위주 페이지는
0.00~0.09로 뚜렷하게 갈렸다. 임계값 0.15로 이 둘을 구분한다.

동작 요약:
  1) 페이지에 임베디드 래스터 이미지가 있으면(사진 등) 그대로 추출한다.
  2) 벡터 드로잉 잉크 비율이 임계값을 넘으면 페이지 전체를 이미지로 렌더링한다
     (어느 부분이 그림인지 정교하게 잘라내는 건 하지 않음 - 페이지 단위가 실용적인
     선에서 충분하다).
  3) Tesseract-OCR(+pytesseract)이 설치돼 있으면 저장된 이미지에서 텍스트 라벨까지
     뽑아서 벡터 검색에 걸리게 한다 (없으면 이미지 추출만 하고 넘어간다 - best-effort).

pipeline/pdf_chunker.py의 build_documents_from_pdf(extract_figures=True)에서 호출된다.
"""
import hashlib
import json
import os
from pathlib import Path

import pymupdf

MIN_IMAGE_WIDTH = 120
MIN_IMAGE_HEIGHT = 120
INK_RATIO_THRESHOLD = 0.15  # 이 이상이면 "실제 삽화가 있는 페이지"로 판단
RENDER_ZOOM = 2.0  # 페이지 렌더링 배율 (2.0 = 대략 144dpi, 화면 표시용으로 충분)

GLOBAL_CACHE_PATH = "data/manual_figures/_global_hash_cache.json"

_ocr_available: bool | None = None
_global_cache: dict[str, dict] | None = None


def _load_global_cache() -> dict[str, dict]:
    global _global_cache
    if _global_cache is not None:
        return _global_cache
    if os.path.exists(GLOBAL_CACHE_PATH):
        with open(GLOBAL_CACHE_PATH, encoding="utf-8") as f:
            _global_cache = json.load(f)
    else:
        _global_cache = {}
    return _global_cache


def _save_global_cache() -> None:
    if _global_cache is None:
        return
    Path(GLOBAL_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_global_cache, f, ensure_ascii=False)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_ocr_available() -> bool:
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        _ocr_available = True
    except Exception:
        _ocr_available = False
        print(
            "[figure_extractor] Tesseract OCR을 찾을 수 없어 그림 안의 텍스트는 추출하지 "
            "않습니다 (그림 추출 자체는 계속 진행). pip install pytesseract 와 "
            "https://github.com/UB-Mannheim/tesseract/wiki 의 Tesseract 바이너리(한국어 "
            "언어팩 포함)를 설치하면 활성화됩니다."
        )
    return _ocr_available


def _ocr_bytes(image_bytes: bytes, label: str) -> str:
    if not _check_ocr_available():
        return ""
    import io

    import pytesseract
    from PIL import Image

    try:
        text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)), lang="kor+eng")
        return text.strip()
    except Exception as e:
        print(f"[figure_extractor] OCR 실패 ({label}): {e}")
        return ""


def _ocr_with_cache(image_bytes: bytes, label: str, run_ocr: bool) -> tuple[str, bool]:
    """이미지 바이트를 해시로 전역 캐시에서 조회한다. 캐시에 있으면(다른 모델 문서에서
    이미 같은 그림을 처리한 적 있으면) OCR을 다시 돌리지 않고 재사용한다.
    반환값: (ocr_text, cache_hit 여부)."""
    cache = _load_global_cache()
    h = _hash_bytes(image_bytes)
    if h in cache:
        return cache[h]["ocr_text"], True
    ocr_text = _ocr_bytes(image_bytes, label) if run_ocr else ""
    cache[h] = {"ocr_text": ocr_text}
    return ocr_text, False


def _page_ink_ratio(page) -> float:
    """벡터 드로잉들의 bbox 면적 합 / 페이지 면적. 실제 삽화가 있는 페이지일수록
    크고 채워진 도형이 많아서 값이 크고, 텍스트 위주 페이지는 얇은 선(표 경계선,
    장식용 아이콘)뿐이라 값이 작다 - 실측 근거는 모듈 docstring 참고."""
    page_area = page.rect.width * page.rect.height
    if not page_area:
        return 0.0
    total_ink = sum(
        d["rect"].width * d["rect"].height for d in page.get_drawings() if d.get("rect")
    )
    return total_ink / page_area


MANIFEST_FILENAME = "_manifest.json"


def extract_figures(
    pdf_path: str,
    out_dir: str,
    min_width: int = MIN_IMAGE_WIDTH,
    min_height: int = MIN_IMAGE_HEIGHT,
    ink_ratio_threshold: float = INK_RATIO_THRESHOLD,
    run_ocr: bool = True,
    use_cache: bool = True,
) -> dict[int, list[dict]]:
    """PDF에서 그림(임베디드 이미지 + 벡터 다이어그램 페이지 렌더링)을 추출해 out_dir에 저장한다.

    반환값: {페이지번호(1-based): [{"path", "width", "height", "ocr_text", "kind"}, ...]}
    "kind"는 "embedded_image"(원본 임베디드 이미지) 또는 "rendered_page"(벡터 다이어그램이라
    페이지 전체를 렌더링한 것)이다. pdf_chunker.sections_to_documents()가 이 결과를 받아서
    각 청크가 커버하는 페이지 범위 안의 그림을 metadata에 연결한다.

    use_cache=True(기본)면 out_dir에 이전 실행 결과(_manifest.json)가 있을 때 그걸 그대로
    읽어서 반환한다 - 페이지 렌더링+OCR은 문서당 수십 초가 걸려서, 청킹 로직을 바꿔 재적재할
    때마다 매번 다시 돌리면 낭비다. 그림 추출 로직 자체를 바꿨다면 out_dir을 지우거나
    use_cache=False로 강제 재실행할 것.
    """
    import json

    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    if use_cache and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            cached = json.load(f)
        return {int(k): v for k, v in cached.items()}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    result: dict[int, list[dict]] = {}
    n_embedded = 0
    n_rendered = 0
    n_cache_hits = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_no = page_num + 1
        kept = []

        # 1) 임베디드 래스터 이미지 (사진 등)
        seen_xrefs = set()
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            width, height = base.get("width", 0), base.get("height", 0)
            if width < min_width or height < min_height:
                continue
            image_bytes = base["image"]
            ext = base.get("ext", "png")
            fname = f"p{page_no:03d}_img{img_idx}.{ext}"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "wb") as f:
                f.write(image_bytes)
            ocr_text, cache_hit = _ocr_with_cache(image_bytes, fpath, run_ocr)
            n_cache_hits += int(cache_hit)
            kept.append({
                "path": fpath, "width": width, "height": height,
                "ocr_text": ocr_text, "kind": "embedded_image",
            })
            n_embedded += 1

        # 2) 벡터 다이어그램 -> 페이지 전체 렌더링
        if _page_ink_ratio(page) >= ink_ratio_threshold:
            fname = f"p{page_no:03d}_page.png"
            fpath = os.path.join(out_dir, fname)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM))
            pix.save(fpath)
            image_bytes = pix.tobytes("png")
            ocr_text, cache_hit = _ocr_with_cache(image_bytes, fpath, run_ocr)
            n_cache_hits += int(cache_hit)
            kept.append({
                "path": fpath, "width": pix.width, "height": pix.height,
                "ocr_text": ocr_text, "kind": "rendered_page",
            })
            n_rendered += 1

        if kept:
            result[page_no] = kept

    print(
        f"[figure_extractor] {pdf_path}: 임베디드 이미지 {n_embedded}개, "
        f"렌더링한 다이어그램 페이지 {n_rendered}개, 전역 캐시 히트 {n_cache_hits}개 "
        f"(out_dir={out_dir})"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    _save_global_cache()
    return result


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "figures_out"
    figs = extract_figures(path, out)
    for page, items in sorted(figs.items()):
        for it in items:
            print(f"p{page} [{it['kind']}] {it['path']} ({it['width']}x{it['height']}) ocr={it['ocr_text'][:40]!r}")
