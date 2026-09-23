"""faq_label_ground_truth_v2.py의 심사 프롬프트 보강 - 스키마는 그대로(정답 청크 1개).

v2(청크 전체 기준, 순환성 없음)로 만든 99건 중 리랭킹 "미스" 34건을 눈으로 직접
대조해보니(rerank_miss_qualitative_check.py), 최소 21/28건이 리랭커가 실제로는
맞는 답을 골랐는데 라벨과 ID만 다른 경우였다. 원인은 두 가지 오라벨 패턴이었다:
  - 목차(TOC) 청크가 우연히 키워드를 포함해서 "정답"으로 잘못 뽑힘(5건)
  - 포괄적인 챕터(예: "설치 관련" 안전수칙)가 라벨인데, 실제로는 더 구체적인
    전용 챕터(예: "안심하세요! 고장이 아닙니다" FAQ)가 더 나은 답인 경우(11건)
둘 다 "정답이 여러 개"라서 생기는 문제가 아니라 "심사 프롬프트가 목차/키워드만
스치는 후보를 걸러내지 못해서" 생기는 문제라, v2와 동일하게 정답 청크 1개만
고르되 이 두 가지를 명시적으로 제외하도록 프롬프트만 보강했다.

(나머지 패턴 - 같은 챕터가 여러 청크로 쪼개진 경우(5건)는 정답 리스트화도
검토했으나, "버튼과 표시부"처럼 큰 제목 하나에 서로 다른 세부 주제가 섞인
사례가 확인돼(RP20C3111S9 "온도 설정" 질문에 "잠금/센서" 청크까지 같은 제목으로
묶여 있음, 2026-09-23) 챕터 단위 자동 판정은 위험하다고 판단, 스키마 변경은
보류. 애매한 나머지 7건도 우선 이 프롬프트 보강만으로 재확인.)

같은 10개 모델 x 15문항(150개, MODEL_QUESTIONS round1~3) 재사용, v2와 직접 비교.

⚠ OpenAI 비용 발생(150문항 x gpt-4o-mini 판정 1회, 예상 총 비용 $0.3~0.5,
사전 승인 후 실행, 2026-09-23).

사용법: python experiments/faq_label_ground_truth_v3.py [--resume]
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
    "청크 전체다(검색으로 거르지 않음).\n"
    "다음 후보는 정답에서 제외해:\n"
    "- 목차/색인처럼 챕터 제목과 페이지 번호만 나열돼 있고 실제 설명이 없는 후보\n"
    "- 질문의 키워드가 스치듯 한 번 언급될 뿐, 실제 절차·원인·해결책 같은 "
    "실질적인 내용은 없는 후보(예: 안전수칙 문단에 '소음'이라는 단어만 들어 있고 "
    "소음의 원인이나 해결책은 없는 경우)\n"
    "완결된 답을 주는 후보가 여러 개면, 질문에 가장 구체적이고 직접적으로 "
    "답하는 후보 하나만 골라(포괄적인 안내보다 전용 설명/FAQ 챕터를 우선). "
    "번호가 붙은 후보들을 읽고, 위 조건을 만족하는 후보가 있으면 그 번호만, "
    "없으면 'NONE'만 출력해. 다른 설명 없이 숫자 하나 또는 NONE만 출력해."
)

MODELS = ["FQ18GU1BHN", "AR-EH05", "M402ND", "RS63R557EB4", "RP20C3111S9",
          "WF17M9100KG", "DV90TA040TE", "FC2521KX6C", "SC3GBE50", "DJ30T9500FE"]

OUT_PATH = Path(__file__).resolve().parent / "results" / "faq_ground_truth_labels_v3_full.json"
OLD_V2_PATH = Path(__file__).resolve().parent / "results" / "faq_ground_truth_labels_v2_full.json"


def load_full_chunks(state: dict) -> dict[str, list[dict]]:
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
    print(f"\n라벨링 대상 질문: {len(targets)}개 (프롬프트 보강판)", flush=True)

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
    with ThreadPoolExecutor(max_workers=2) as pool:
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
                )

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

    if OLD_V2_PATH.exists():
        old = {(d["question"], d["model"]): d for d in json.loads(OLD_V2_PATH.read_text(encoding="utf-8"))}
        flipped_to_labeled, flipped_answer, same_answer = 0, 0, 0
        for r in labels:
            if r is None or r["note"] != "labeled":
                continue
            o = old.get((r["question"], r["model"]))
            if o is None:
                continue
            if o["note"] == "no_answer_in_doc":
                flipped_to_labeled += 1
            elif o["note"] == "labeled":
                if o["source_chunk_id"] == r["source_chunk_id"]:
                    same_answer += 1
                else:
                    flipped_answer += 1
        print(f"\n=== v2 대비 변화 ===")
        print(f"no_answer_in_doc -> labeled: {flipped_to_labeled}건")
        print(f"정답 청크가 바뀜(v2가 목차/포괄챕터를 잘못 골랐던 것으로 추정): {flipped_answer}건")
        print(f"동일한 정답 유지: {same_answer}건")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
