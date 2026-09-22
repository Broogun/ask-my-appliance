"""답변 품질 채점 (LLM-as-judge) — 검색은 같은데 답이 구린 경우를 잰다. 자체 테스트셋에서 문항을 뽑아
(a) 팀 채점용 프롬프트(SYSTEM_PROMPT + build_context)와 (b) 웹 전용 프롬프트(WEB_SYSTEM_PROMPT + build_web_context)로 답을 만들고,
gpt-4o-mini 심판이 근거 조각을 보며 세 가지를 0/1로 채점한다.

    python experiments/siyeon/eval_answers.py --n 30 --tag round1
    → experiments/siyeon/results/siyeon_answers_<tag>.json, 콘솔에 (a)/(b) 비교표

채점 항목 (실사용 로그에서 나온 결함 3종에 대응)
  faithful  : 답의 모든 사실·수치·절차가 근거에 있다 (설명서 밖 일반 조언·추정 수치 없음)
  matched   : 근거 여러 개 중 질문에 맞는 항목을 골라 답했다 (맞는 근거가 있는데 "없다"고 하거나, 다른 증상의 조치를 섞지 않음)
  direction : 높이다/낮추다·켜다/끄다 등 방향과 버튼·수치 표현이 근거와 같다
비용: 문항 30개 × (답변 2 + 심판 2) ≈ 120 호출, $0.05 안팎. 검색 결과는 (a)(b) 공통이라 검색기 1회.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402
from experiments.harness.evaluate import SYSTEM_PROMPT  # noqa: E402
from siyeon.rag.chunking_lg import build_all_chunks  # noqa: E402
from siyeon.rag.chunking_samsung import build_samsung_chunks  # noqa: E402
from siyeon.rag.retrieval import LGAirconRetriever, build_context  # noqa: E402
from siyeon.rag.understand import understand  # noqa: E402
from web.rag_service import WEB_SYSTEM_PROMPT, build_web_context  # noqa: E402

TESTSET = Path(__file__).with_name("testset.json")
OUT_DIR = Path(__file__).with_name("results")   # 자체 채점 결과 — 팀 보고서(experiments/results, compare_results.py)와 형식이 달라 따로 둔다
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

JUDGE = """당신은 가전 설명서 챗봇의 답변을 채점하는 심판입니다. [근거]는 챗봇이 받은 설명서 조각 전부이고, [답변]은 그 근거로 만든 답입니다.
다음 세 항목을 각각 1(충족) 또는 0(위반)으로 채점하고 JSON만 출력하세요.

faithful : 답변의 모든 사실·수치·절차·원인이 [근거]에 실제로 있다. 근거에 없는 일반 상식·추정 수치·"보통은 ~" 식 조언이 하나라도 있으면 0.
           근거가 비어 있거나 무관한데 "설명서에 없다"고 답했으면 1.
matched  : [근거]에 질문과 맞는 항목이 있으면 그것으로 답했다. 맞는 근거가 있는데 "설명서에 없다"고 했거나, 질문과 다른 증상·항목의 조치를 가져다 답했으면 0.
           맞는 근거가 없을 때 없다고 한 것은 1.
direction: 높이다/낮추다, 켜다/끄다, 열다/닫다 같은 방향, 버튼 이름, 수치·시간·온도가 [근거] 표현과 일치한다. 뒤집히거나 바뀐 게 있으면 0. 해당 표현이 없으면 1.

출력: {"faithful": 0|1, "matched": 0|1, "direction": 0|1, "note": "위반이 있으면 한 줄로 어디가 문제인지"}"""


def answer(system: str, user: str) -> str:
    r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content


def judge(question: str, evidence: str, ans: str) -> dict:
    r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"},
                                       messages=[{"role": "system", "content": JUDGE},
                                                 {"role": "user", "content": f"[질문]\n{question}\n\n[근거]\n{evidence or '(없음)'}\n\n[답변]\n{ans}"}])
    try:
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {"faithful": 0, "matched": 0, "direction": 0, "note": "judge parse error"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--tag", default="latest"); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cases = [c for c in json.loads(TESTSET.read_text(encoding="utf-8"))["cases"] if c.get("doc_id") and not c["expect"].get("none")]
    random.Random(a.seed).shuffle(cases)
    # 증상·관리·사용법·안전이 골고루 들어가게 카테고리별로 뽑는다
    picked, per_cat = [], {}
    for c in cases:
        k = c["category"]
        if per_cat.get(k, 0) < max(2, a.n // 5):
            picked.append(c); per_cat[k] = per_cat.get(k, 0) + 1
        if len(picked) >= a.n:
            break
    r = LGAirconRetriever(build_all_chunks() + build_samsung_chunks(), verbose=False)
    r.reranker.predict([("워밍업", "리랭커")])
    rows = []
    t0 = time.time()
    for i, c in enumerate(picked, 1):
        appl = r.appliance_for(c["doc_id"])
        und = understand(c["query"], appl)
        res = r.retrieve(c["query"], doc_id=c["doc_id"], top_k=3, feature=und.feature)
        evidence = "\n\n".join(f"[{ch['section']} > {ch['subsection']}]\n{ch['body']}" for ch, _ in res.used)
        # (a) 팀 채점용 프롬프트  (b) 웹 전용 프롬프트
        ans_a = answer(SYSTEM_PROMPT, f"[검색된 문서]\n{build_context(res.used, res.appliance)}\n\n[질문]\n{c['query']}")
        ans_b = answer(WEB_SYSTEM_PROMPT, f"{build_web_context(res)}\n\n[질문]\n{c['query']}")
        ja, jb = judge(c["query"], evidence, ans_a), judge(c["query"], evidence, ans_b)
        rows.append({"id": c["id"], "category": c["category"], "doc_id": c["doc_id"], "query": c["query"], "route": res.route,
                     "n_used": len(res.used), "answer_a": ans_a, "answer_b": ans_b, "judge_a": ja, "judge_b": jb})
        print(f"  [{i:2d}/{len(picked)}] {c['id']:12s} a={ja['faithful']}{ja['matched']}{ja['direction']} b={jb['faithful']}{jb['matched']}{jb['direction']}  {c['query'][:30]}")

    def score(key, side):
        return sum(int(r_[f"judge_{side}"].get(key, 0)) for r_ in rows)
    n = len(rows)
    print(f"\n=== 답변 품질 {n}문항 · {time.time()-t0:.0f}s ===")
    print(f"{'항목':10s} {'(a) 팀 프롬프트':>14s} {'(b) 웹 프롬프트':>14s}")
    for k, name in (("faithful", "근거 충실"), ("matched", "맞는 근거 선택"), ("direction", "방향·수치 보존")):
        print(f"{name:10s} {score(k,'a'):7d}/{n:<6d} {score(k,'b'):7d}/{n}")
    all_a = sum(1 for r_ in rows if all(int(r_["judge_a"].get(k, 0)) for k in ("faithful", "matched", "direction")))
    all_b = sum(1 for r_ in rows if all(int(r_["judge_b"].get(k, 0)) for k in ("faithful", "matched", "direction")))
    print(f"{'3항목 모두':10s} {all_a:7d}/{n:<6d} {all_b:7d}/{n}")
    bad_b = [r_ for r_ in rows if not all(int(r_["judge_b"].get(k, 0)) for k in ("faithful", "matched", "direction"))]
    if bad_b:
        print(f"\n=== (b) 위반 {len(bad_b)}건 ===")
        for r_ in bad_b:
            print(f"  [{r_['id']}] {r_['query']}  → {r_['judge_b'].get('note','')}")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"siyeon_answers_{a.tag}.json"
    out.write_text(json.dumps({"n": n, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[저장] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
