# 제품 썸네일

| 위치 | 무엇 | 누가 |
|---|---|---|
| `{manual_id}.png` 또는 `.jpg` (이 폴더) | **실제 제품 사진** — 있으면 최우선으로 표시. **LG 30개는 들어 있음**(원본 `experiments/siyeon/LG_images/*.avif` → 400px PNG 변환) | 삼성은 발표 시연 범위 밖이라 비워둠 → 아이콘 |
| `auto/{manual_id}.png` | 설명서 PDF의 제품 선화를 자동으로 잘라낸 것 | `python web/tools/make_thumbs.py` |
| (둘 다 없음) | 제품군 아이콘(❄️ 🧊 🫧 …) | 자동 폴백 |

`auto/sheet.png`는 검수용 모음 — 이상하게 잘린 선화는 그 파일만 지우면 아이콘으로 폴백됩니다.
manual_id는 `data/appliance.sqlite`의 `manuals.manual_id` (= PDF 파일명, 예: `WM_F21VDSK`).
사진은 정방형에 가깝고 배경이 흰 것이 카드에 잘 맞습니다 (권장 400×400 이하).
