"""청크별 합성 질문(FAQ) 생성 + 별도 Chroma 컬렉션 적재.

**왜 하는가**: 어제 감사에서 확인된 핵심 실패 원인이 "질문과 정답 청크의 표현이
다르다"는 것이었다 - "온도 설정 방법 알려줘"라는 질문이, 실제 정답인 "제어창:
버튼을 눌러 온도를 조절할 수 있습니다"보다 오답인 트러블슈팅 문서("설정 온도를
확인했나요? 낮게 설정하세요")에 더 가깝게 임베딩되는 문제. 청크마다 "이 내용이
답할 수 있는 질문"을 미리 생성해서 그 질문 임베딩을 인덱싱해두면, 사용자 질문이
청크 원문이 아니라 같은 '질문' 모드끼리 비교되므로 이 갭이 줄어든다.

**중요**: 생성된 FAQ 질문은 검색 매칭용 앵커일 뿐이다. 실제 답변 근거로 LLM에
넘기는 건 항상 원본 청크 텍스트(source_chunk_id로 되돌려서 조회)다 - 생성된
문장을 근거로 쓰면 환각이 근거로 둔갑한다.

사용법:
  python experiments/faq_generate.py --models GC-B40BSCQ DF90H24R5C   # 프로토타입
  python experiments/faq_generate.py --all                            # 전체
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
import torch
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faq_filter import build_faq_targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CHROMA_PATH = ROOT / "chroma_db"
SOURCE_COLLECTION = "appliance_manuals"
FAQ_COLLECTION = "appliance_faq"

OAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
GEN_MODEL = "gpt-4o-mini"
FAQ_PER_CHUNK = 5  # 3->5 (2026-09-16, #17 대책 - 아래 v2 프롬프트 참고)
MAX_WORKERS = 5  # 어제 429 겪고 5로 고정 (15은 rate limit 터짐)

# v2 (2026-09-16) - #17 대책: section_title에 갇혀서 "음성으로 냉장고 온도
# 조절은?"처럼 좁은 질문만 나오던 문제(M402ND_6_0 사례로 확인)를 겨냥해
# (1) 제목이 아니라 본문 전체를 반영하라고 명시, (2) 3->5개로 늘려 다양성
# 확보, (3) 질문마다 짧은 답도 같이 쓰게 해서(Q+A) 그 질문이 진짜 본문으로
# 답이 되는지 모델 스스로 확인하게 함(chain-of-thought 효과) - 단, 답변은
# 생성 품질 검증용일 뿐 검색 인덱싱에는 안 쓴다(질문만 임베딩 - 여전히
# 유지: 원본 청크가 항상 실제 근거, 요약/생성 답변을 근거로 쓰면 환각 위험).
FAQ_SYSTEM_PROMPT = (
    "너는 가전제품 사용설명서를 보고, 실제 사용자가 고객센터에 물어볼 법한 질문을 "
    "만들어내는 어시스턴트야. 주어진 설명서 일부를 읽고, 그 내용으로 답할 수 있는 "
    "질문과 간단한 답을 정확히 {n}쌍 만들어. 규칙: "
    "(1) 소제목([설명서 섹션])에 얽매이지 말고, 본문 전체가 다루는 내용을 골고루 "
    "반영해서 서로 다른 주제의 질문을 만들 것 - 소제목이 가리키는 좁은 주제 하나만 "
    "반복해서 묻지 마. "
    "(2) 반드시 본문만으로 답할 수 있는 질문일 것 - 내용에 없는 걸 묻지 마. "
    "(3) 실제 사용자의 말투로 쓸 것(예: '온도는 어떻게 바꿔요?', '필터 언제 갈아요?'). "
    "(4) 답은 그 질문이 본문으로 실제 답이 되는지 확인하는 용도라 한두 문장으로 짧게. "
    "(5) 모델명은 넣지 마. "
    "(6) JSON 배열만 출력. 다른 설명 없이 "
    "[{{\"q\": \"질문1\", \"a\": \"답1\"}}, ...] 형식."
)


def generate_faqs(target: dict, n: int = FAQ_PER_CHUNK) -> tuple[list[str], dict]:
    """청크 하나에서 FAQ 질문 n개 생성. (질문목록, usage) 반환.

    생성된 답변(a)은 품질 확인용으로만 쓰고 버린다 - 반환하는 건 질문 리스트뿐."""
    user_msg = (
        f"[제품 종류] {target['category']}\n"
        f"[설명서 섹션] {target['section_title']}\n"
        f"[내용]\n{target['text']}"
    )
    for attempt in range(5):
        try:
            resp = OAI.chat.completions.create(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": FAQ_SYSTEM_PROMPT.format(n=n)},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
            )
            break
        except RateLimitError:
            time.sleep(2 ** attempt)
    else:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}

    raw = resp.choices[0].message.content
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [], usage
    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], usage
    questions = [
        p["q"].strip() for p in pairs
        if isinstance(p, dict) and isinstance(p.get("q"), str) and p["q"].strip()
    ]
    return questions, usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="이 모델들만 생성 (프로토타입용)")
    parser.add_argument("--all", action="store_true", help="전체 컬렉션 대상")
    parser.add_argument("--dry-run", action="store_true", help="비용만 추산하고 종료")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    source = client.get_collection(SOURCE_COLLECTION)
    all_docs = source.get(include=["documents", "metadatas"])

    targets, blocked, merged = build_faq_targets(
        all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
    )
    if args.models:
        targets = [t for t in targets if t["product_model"] in args.models]
    elif not args.all:
        parser.error("--models 또는 --all 중 하나는 지정해야 함")

    print(f"FAQ 생성 대상: {len(targets)}개 청크 (블록리스트 제외 {blocked}, 파편 병합 {merged})")

    avg_chars = sum(len(t["text"]) for t in targets) / max(len(targets), 1)
    est_in = len(targets) * (avg_chars * 1.2 + 200)   # 한글 대략 1.2토큰/자 + 지시문
    est_out = len(targets) * 90
    est_cost = est_in / 1e6 * 0.15 + est_out / 1e6 * 0.60
    print(f"예상 비용: 약 ${est_cost:.3f} (입력 ~{est_in/1000:.0f}K, 출력 ~{est_out/1000:.0f}K 토큰)")
    if args.dry_run:
        return

    t0 = time.time()
    rows, usage_total = [], {"prompt_tokens": 0, "completion_tokens": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_faqs, t): t for t in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            target = futures[fut]
            questions, usage = fut.result()
            usage_total["prompt_tokens"] += usage["prompt_tokens"]
            usage_total["completion_tokens"] += usage["completion_tokens"]
            for j, q in enumerate(questions):
                rows.append({
                    "id": f"faq_{target['source_chunk_id']}_{j}",
                    "question": q,
                    "metadata": {
                        "source_chunk_id": target["source_chunk_id"],
                        "product_model": target["product_model"],
                        "category": target["category"],
                        "brand": target["brand"],
                        "section_title": target["section_title"],
                    },
                })
            if i % 50 == 0:
                print(f"  {i}/{len(targets)} 완료 ({time.time()-t0:.0f}초)")

    cost = (usage_total["prompt_tokens"] / 1e6 * 0.15
            + usage_total["completion_tokens"] / 1e6 * 0.60)
    print(f"\n생성 완료: FAQ {len(rows)}개 / 청크 {len(targets)}개, "
          f"{time.time()-t0:.0f}초, 실제 비용 ${cost:.4f}")

    print("임베딩 + 적재 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("dragonkue/BGE-m3-ko", device=device)
    embeddings = embed_model.encode(
        [r["question"] for r in rows], normalize_embeddings=True, batch_size=64,
        show_progress_bar=True,
    ).tolist()

    try:
        faq_col = client.get_collection(FAQ_COLLECTION)
    except Exception:
        faq_col = client.create_collection(FAQ_COLLECTION)

    faq_col.upsert(
        ids=[r["id"] for r in rows],
        embeddings=embeddings,
        documents=[r["question"] for r in rows],
        metadatas=[r["metadata"] for r in rows],
    )
    print(f"'{FAQ_COLLECTION}' 컬렉션 적재 완료 (현재 총 {faq_col.count()}개)")


if __name__ == "__main__":
    main()
