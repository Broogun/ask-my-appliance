"""통합 웹 파이프라인(rag_myretriever + WEB_SYSTEM_PROMPT) baseline 측정.

목적: "Hierarchical RAG 일반화" A/B를 하려면 먼저 지금 방식의 baseline이
있어야 한다. 팀 공용 MODEL_QUESTIONS(round1, 60모델 x 5문항 = 300개)를
재사용해서 web/rag_service.py의 실제 검색+생성 경로(exp02 리트리버+시연님
RDB+WEB_SYSTEM_PROMPT)를 그대로 돌리고, pipeline.evaluate.judge_results로
LLM 4분류 판정까지 매긴다 - my_answer()(구 SYSTEM_PROMPT, 텍스트에서 모델명
파싱)가 아니라 실제 웹 경로(manual_id 기반)를 측정하는 게 핵심이다.

⚠ OpenAI 비용 발생(질문당 리랭킹 1회 + 생성 1회 + 판정 1회 ≈ 900회 호출,
gpt-4o-mini라 소액이지만 실행 전 팀 규칙대로 승인 필요).

사용법: python experiments/web_baseline_eval.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
import openai
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from experiments.harness.evaluate import MODEL_QUESTIONS, judge_results  # noqa: E402
from web import rag_service  # noqa: E402

DB_PATH = ROOT / "data" / "appliance.sqlite"

_BRAND_KO_TO_CODE = {"LG": "lg", "삼성": "samsung"}


def _resolve_manual_id(db: sqlite3.Connection, brand_ko: str, model: str) -> str | None:
    """MODEL_LIST의 (브랜드, 모델) -> manual_id. manual_models 표를 모델패턴
    GLOB으로 조회(정확 일치 우선) - build_db.py가 채운 실제 매핑을 그대로 쓴다."""
    brand = _BRAND_KO_TO_CODE.get(brand_ko)
    row = db.execute(
        "SELECT mm.manual_id FROM manual_models mm JOIN manuals m USING (manual_id) "
        "WHERE m.brand = ? AND (mm.model_pattern = ? OR ? GLOB replace(mm.model_pattern, '*', '?')) "
        "ORDER BY (mm.model_pattern = ?) DESC, mm.source = 'filename' DESC LIMIT 1",
        (brand, model, model, model),
    ).fetchone()
    return row[0] if row else None


def main(n: int | None, resume: bool = False, round_name: str = "round1") -> None:
    db = sqlite3.connect(str(DB_PATH))
    out_path = ROOT / "experiments" / "results" / f"web_baseline_{round_name}.json"

    rows = [r for r in MODEL_QUESTIONS if r[4] == round_name]
    if n:
        rows = rows[:n]
    print(f"대상: {len(rows)}개 ({round_name})", flush=True)

    # brand_ko는 MODEL_LIST에만 있고 MODEL_QUESTIONS엔 model만 있어 재구성
    from experiments.harness.evaluate import MODEL_LIST
    brand_by_model = {model: brand for brand, _category, model in MODEL_LIST}

    results = []
    if resume and out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
        print(f"이어하기: 기존 {len(results)}개 유지", flush=True)
    done = {(r["question"], r["model"]) for r in results}
    t0 = time.time()
    skipped_no_manual = 0
    for i, (question, keyword, model, description, _round) in enumerate(rows, 1):
        if (question, model) in done:
            continue
        manual_id = _resolve_manual_id(db, brand_by_model.get(model, ""), model)
        if manual_id is None:
            skipped_no_manual += 1
            continue
        for attempt in range(6):
            try:
                res = rag_service.retrieve(question, manual_id=manual_id, top_k=5)
                answer = "".join(rag_service.stream_answer(question, res, history=[]))
                break
            except openai.RateLimitError:
                wait = 15 * (attempt + 1)
                print(f"  429 rate limit - {wait}초 대기 후 재시도({attempt + 1}/6)", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"재시도 6회 실패: {question}")
        results.append({
            "question": question, "keyword": keyword, "model": model,
            "description": description, "round": round_name, "route": res.route,
            "answer": answer,
            "candidates": [{"id": c["id"], "text": c["text"], "distance": round(float(d), 4)} for c, d in res.used],
        })
        if len(results) % 20 == 0:
            print(f"  {i}/{len(rows)} 완료 ({time.time()-t0:.0f}초)", flush=True)
            out_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")  # 중간 저장(크래시 대비)

    print(f"\n검색 완료: {len(results)}개 (manual_id 못 찾아 스킵: {skipped_no_manual}개), {time.time()-t0:.0f}초", flush=True)

    out_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out_path}", flush=True)

    print("\nLLM 4분류 판정 시작...", flush=True)
    label_counts = judge_results(results)
    print("판정 분포:", dict(label_counts), flush=True)

    out_path.write_text(json.dumps({"results": results, "label_counts": dict(label_counts)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"최종 저장: {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="테스트용 일부만(파일럿)")
    parser.add_argument("--resume", action="store_true", help="기존 저장분 유지하고 남은 문항만")
    parser.add_argument("--round", type=str, default="round1", choices=["round1", "round2", "round3"])
    args = parser.parse_args()
    main(args.n, args.resume, args.round)
