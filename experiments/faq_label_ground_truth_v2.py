"""faq_label_ground_truth.py의 순환성·생존 편향을 근본적으로 고친 재라벨링.

기존 문제: label_question()이 "벡터 검색 top-20 안에서" 정답을 고르라고 시켰다 -
이러면 ① 정답 판정 자체가 검색 엔진(임베딩 모델)이 이미 거른 결과에 의존해서
순환적이고, ② top-20 밖에 정답이 있는 질문은 "no_answer_in_doc"으로 버려져서
(150문항 중 53개) 생존 편향이 생긴다(검색이 어려워하는 질문은 애초에 벤치마크에
못 들어옴). 2026-09-22 사용자 지적으로 발견, docs/05-evaluation.md 5.5/5.10 참고.

이 스크립트는 top-20 대신 **그 product_model에 속한 청크 전체**를 gpt-4o-mini에게
보여주고 정답을 고르게 한다 - 정답 판정이 벡터 검색 결과와 완전히 독립적이게 되고,
진짜로 매뉴얼에 없을 때만 "없다"고 판정된다. 모델당 청크 수가 26~128개(토큰
5천~2.5만)라 gpt-4o-mini 컨텍스트로 충분히 가능하다.

같은 10개 모델 x 15문항(150개, MODEL_QUESTIONS round1~3)을 그대로 재사용해서
기존 결과와 직접 비교 가능하게 했다.

2026-09-22 1차 실행(worker=4)이 20/150에서 OpenAI RateLimitError로 죽었다 -
프롬프트가 커서(최대 2.5만 토큰) 4개 병렬이 분당 토큰 한도(TPM)를 넘은 것으로
추정. worker=2로 낮추고, 실패한 문항은 재시도(최대 3회, 대기 늘림)하며, 10개마다
중간 저장(--resume으로 이어하기)하도록 고침.

⚠ OpenAI 비용 발생(150문항 x gpt-4o-mini 판정 1회, 전체 청크를 매번 보여줘서
질문당 프롬프트가 기존보다 큼 - 예상 총 비용 $0.3~0.5, 사전 승인 후 실행,
2026-09-22).

사용법: python experiments/faq_label_ground_truth_v2.py [--resume]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import exp02_retrieval as exp02  # noqa: E402
from experiments.harness.evaluate import MODEL_QUESTIONS  # noqa: E402

JUDGE_PROMPT = (
    "너는 가전제품 사용설명서 검색 결과를 채점하는 심사위원이야. [질문]에 실제로 "
    "답할 수 있는 내용이 [후보] 중에 있는지 판단해. 아래 후보는 이 모델 설명서의 "
    "청크 전체다(검색으로 거르지 않음). 번호가 붙은 후보들을 읽고, 질문에 명확히 "
    "답이 되는 후보가 있으면 그 번호만, 없으면(전부 무관하거나 일부만 겹치고 핵심은 "
    "안 담고 있으면) 'NONE'만 출력해. 다른 설명 없이 숫자 하나 또는 NONE만 출력해."
)

MODELS = ["FQ18GU1BHN", "AR-EH05", "M402ND", "RS63R557EB4", "RP20C3111S9",
          "WF17M9100KG", "DV90TA040TE", "FC2521KX6C", "SC3GBE50", "DJ30T9500FE"]

OUT_PATH = Path(__file__).resolve().parent / "results" / "faq_ground_truth_labels_v2_full.json"


def load_full_chunks(state: dict) -> dict[str, list[dict]]:
    """모델별 전체 청크(청크 id, 제목, 본문)를 한 번만 가져와 캐싱 - 같은 모델의
    15문항이 전부 같은 후보 목록을 재사용한다(벡터 검색 없이 컬렉션 직접 조회)."""
    col = state["collection"]
    out = {}
    for m in MODELS:
        res = col.get(where={"product_model": m}, include=["documents", "metadatas"])
        chunks = [
            {"id": cid, "text": doc, "section_title": meta.get("section_title", "")}
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        out[m] = chunks
        print(f"  {m}: {len(chunks)}개 청크 로딩", flush=True)
    return out


def label_question_full(question: str, model: str, chunks: list[dict]) -> dict:
    listing = "\n\n".join(
        f"[{i}] ({c['section_title']}) {c['text']}" for i, c in enumerate(chunks)
    )
    # generate_openai 자체가 이미 재시도(5회, 최대 31초 백오프 + 매 시도 30초
    # 타임아웃)를 한다 - 여기서 또 감쌌더니 최악의 경우 10분 넘게 한 문항을
    # 물고 늘어지는 이중 재시도가 돼서(2026-09-22 실측, 두 번 연속 발생) 제거함.
    # 한 번 실패하면 바로 error로 넘기고 스크립트 재실행(--resume)으로 재시도한다.
    try:
        verdict = exp02.generate_openai(
            JUDGE_PROMPT, f"[질문]\n{question}\n\n[후보]\n{listing}", temperature=0.0,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return {"question": question, "model": model, "source_chunk_id": None,
                "note": "error", "error": str(e)}

    match = re.search(r"\d+", verdict)
    if "NONE" in verdict.upper() or not match:
        return {"question": question, "model": model, "source_chunk_id": None, "note": "no_answer_in_doc"}
    idx = int(match.group(0))
    if idx >= len(chunks):
        return {"question": question, "model": model, "source_chunk_id": None, "note": "bad_index"}
    return {
        "question": question, "model": model,
        "source_chunk_id": chunks[idx]["id"],
        "source_section_title": chunks[idx]["section_title"],
        "note": "labeled",
    }


def main(resume: bool) -> None:
    state = exp02._load_state()
    print("모델별 전체 청크 로딩 중...", flush=True)
    chunks_by_model = load_full_chunks(state)

    targets = [q for q in MODEL_QUESTIONS if q[2] in MODELS]
    print(f"\n라벨링 대상 질문: {len(targets)}개 (전체 청크 기준)", flush=True)

    done_map: dict[tuple[str, str], dict] = {}
    if resume and OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        done_map = {(d["question"], d["model"]): d for d in existing if d.get("note") != "error"}
        print(f"이어하기: 기존 {len(done_map)}개 유지(에러난 건 재시도 대상)", flush=True)

    labels: list[dict | None] = [None] * len(targets)
    todo = []
    for i, (question, keyword, model, description, round_name) in enumerate(targets):
        key = (question, model)
        if key in done_map:
            labels[i] = done_map[key]
        else:
            todo.append(i)
    print(f"이번에 처리할 문항: {len(todo)}개", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:  # 4->2: TPM 초과 방지
        futures = {
            pool.submit(label_question_full, targets[i][0], targets[i][2], chunks_by_model[targets[i][2]]): i
            for i in todo
        }
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            question, keyword, model, description, round_name = targets[i]
            r = fut.result()
            r["round"] = round_name
            r["keyword"] = keyword
            labels[i] = r
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} 완료 ({time.time()-t0:.0f}초)", flush=True)
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUT_PATH.write_text(
                    json.dumps([l for l in labels if l is not None], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )  # 중간 저장(크래시 대비)

    n_labeled = sum(1 for r in labels if r and r["note"] == "labeled")
    n_no_answer = sum(1 for r in labels if r and r["note"] == "no_answer_in_doc")
    n_error = sum(1 for r in labels if r and r["note"] == "error")
    print(f"\n라벨링 완료: {n_labeled}개 정답 확보, {n_no_answer}개는 문서 전체에도 답 없음, {n_error}개 에러", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps([l for l in labels if l is not None], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"저장: {OUT_PATH}")

    # 기존(top-20 기준) 라벨과 비교
    old_path = Path(__file__).resolve().parent / "results" / "faq_ground_truth_labels.json"
    if old_path.exists():
        old = {(d["question"], d["model"]): d for d in json.loads(old_path.read_text(encoding="utf-8"))}
        flipped_to_labeled = 0
        flipped_answer = 0
        same_answer = 0
        for r in labels:
            if r is None:
                continue
            o = old.get((r["question"], r["model"]))
            if o is None:
                continue
            if o["note"] == "no_answer_in_doc" and r["note"] == "labeled":
                flipped_to_labeled += 1
            elif o["note"] == "labeled" and r["note"] == "labeled":
                if o["source_chunk_id"] != r["source_chunk_id"]:
                    flipped_answer += 1
                else:
                    same_answer += 1
        print(f"\n=== 기존(top-20) 대비 변화 ===")
        print(f"no_answer_in_doc -> labeled로 바뀜(생존 편향으로 놓쳤던 것): {flipped_to_labeled}건")
        print(f"labeled -> labeled인데 정답 청크가 바뀜(top-20 밖에 더 나은 답이 있었던 것): {flipped_answer}건")
        print(f"동일한 정답 유지: {same_answer}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
