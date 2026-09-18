"""전체 6,105개 청크에 "의미 유형"(content_type) 메타데이터를 소급 태깅한다.

**배경**: round1은 #18(트러블슈팅)/#19(증상보고+패널)/#21(스펙표) 구조적 매칭으로
크게 개선됐는데(93.7% 실질성공률), round2/3은 새로운 질문 유형("OO 기능이 뭐야"
류 기능설명형, round2 6모델 전부 FALSE_DECLINE)에 그대로 노출돼서 72~76%에
그쳤다 - 타이틀을 일일이 손으로 찾는 방식(#18/#19/#21)은 "실패 사례를 보고
반응적으로 규칙을 만드는" 구조라 새 질문 유형마다 사각지대가 계속 생긴다.

팀원이 공유한 청킹 전략(PROCEDURE/TROUBLESHOOTING/GENERAL 의미 유형 태깅)에서
착안해, 청크마다 미리 의미 유형을 분류해두면 질문 의도와 매칭하는 일반화된
필터로 쓸 수 있다. 원안 3종에 오늘 실측된 두 가지 큰 실패군(기능설명형/스펙형)을
분리해 5종으로 확장:
  PROCEDURE          - 사용/설정/조작/청소 등 절차 안내
  TROUBLESHOOTING     - 증상/원인/해결책, 에러/고장 대응
  FEATURE_EXPLANATION - "이 기능이 뭐야/뭘 하는 기능이야" 식 기능 설명
  SPEC                - 용량/전력/무게/치수 등 규격·수치 정보
  GENERAL             - 설치, 안전 경고, 기타(위 4종에 안 맞는 전부)

**중요**: 이 태그는 검색을 "대체"하지 않고 "추가"하는 용도로만 쓴다(FAQ와 같은
패턴) - 분류가 가끔 틀려도(예: 애매한 GENERAL/PROCEDURE 경계) 기존 검색 경로가
그대로 살아있어서 회귀 위험이 없다.

사용법: python experiments/tag_content_type.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
CHROMA_PATH = ROOT / "chroma_db"

# 첫 시도(2026-09-16)에서 300개까지는 43초 만에 끝냈는데 그 이후 10분+ 진행이
# 멈췄다 - 타임아웃을 안 걸어둬서 네트워크가 걸리면 무한정 대기했을 가능성이
# 높고, 중간 저장도 없어서 강제 종료하면 전부 날아가는 구조였다. (1) 호출에
# 명시적 타임아웃(20초)을 걸고 (2) 체크포인트 파일에 주기적으로 저장해서
# 죽여도 이어서 할 수 있게 고침.
OAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=20.0)
GEN_MODEL = "gpt-4o-mini"
MAX_WORKERS = 5
CHECKPOINT_PATH = ROOT / "experiments" / "results" / "content_type_checkpoint.json"

TYPES = ("PROCEDURE", "TROUBLESHOOTING", "FEATURE_EXPLANATION", "SPEC", "GENERAL")

TAG_SYSTEM_PROMPT = (
    "너는 가전제품 사용설명서 청크를 의미 유형으로 분류하는 어시스턴트야. "
    "다음 5종 중 정확히 하나만 골라: "
    "PROCEDURE(사용/설정/조작/청소 등 절차 안내), "
    "TROUBLESHOOTING(증상-원인-해결책 형식의 고장/에러 대응), "
    "FEATURE_EXPLANATION(특정 기능이 무엇인지 설명하는 내용), "
    "SPEC(용량/전력/무게/치수 등 규격·수치 표), "
    "GENERAL(설치, 안전 경고, 위 4종에 안 맞는 기타 전부). "
    "다른 설명 없이 저 5개 단어 중 하나만 정확히 출력해."
)


def classify(chunk_id: str, title: str, text: str) -> tuple[str, str]:
    user_msg = f"[섹션 제목] {title}\n[내용]\n{text[:300]}"
    for attempt in range(5):
        try:
            resp = OAI.chat.completions.create(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": TAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            break
        except (RateLimitError, APITimeoutError):
            time.sleep(2 ** attempt)
        except Exception:
            # 예상 못한 예외(네트워크 등)도 재시도하되, 스레드가 영원히
            # 걸려있지 않도록 동일한 백오프 한도 안에서만 시도한다.
            time.sleep(2 ** attempt)
    else:
        return chunk_id, "GENERAL"
    raw = resp.choices[0].message.content.strip().upper()
    label = next((t for t in TYPES if t in raw), "GENERAL")
    return chunk_id, label


def load_checkpoint() -> dict[str, str]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(labels: dict[str, str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection("appliance_manuals")
    all_docs = col.get(include=["documents", "metadatas"])
    ids, docs, metas = all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
    print(f"전체 청크: {len(ids)}개")

    labels: dict[str, str] = load_checkpoint()
    if labels:
        print(f"체크포인트에서 {len(labels)}개 재개")
    meta_by_id = {cid: meta for cid, meta in zip(ids, metas)}
    todo = [
        (cid, meta_by_id[cid].get("section_title", ""), doc)
        for cid, doc in zip(ids, docs)
        if cid not in labels
    ]

    t0 = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(classify, cid, title, doc): cid
                for cid, title, doc in todo
            }
            n_done = 0
            for fut in as_completed(futures):
                cid, label = fut.result()
                labels[cid] = label
                n_done += 1
                if n_done % 100 == 0:
                    save_checkpoint(labels)
                if n_done % 300 == 0:
                    print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)")
        save_checkpoint(labels)

    from collections import Counter
    print(f"\n분류 완료: {time.time()-t0:.0f}초 (신규 {len(todo)}개)")
    print(Counter(labels.values()))

    print("\nChroma 메타데이터 업데이트 중...")
    idx_map = {cid: idx for idx, cid in enumerate(ids)}
    batch = 500
    id_list = list(labels.keys())
    for i in range(0, len(id_list), batch):
        chunk_ids = id_list[i:i + batch]
        # 원본 metadata를 유지한 채 content_type만 추가(덮어쓰기 아님)
        new_metas = []
        for cid in chunk_ids:
            m = dict(metas[idx_map[cid]])
            m["content_type"] = labels[cid]
            new_metas.append(m)
        col.update(ids=chunk_ids, metadatas=new_metas)
    print("업데이트 완료")
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()
