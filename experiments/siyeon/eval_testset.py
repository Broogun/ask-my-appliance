"""자체 테스트셋(testset.json)으로 검색 정확도 측정 — LLM 없이 검색만.

    python experiments/siyeon/eval_testset.py --tag exp05           # 결과 → experiments/siyeon/results/siyeon_testset_exp05.json
    python experiments/siyeon/eval_testset.py --tag base --mode base   # 절제 실험: base(벡터만) / dense(벡터×2+리랭커) / v3(기본: 이해+라우터)

지표
  top-1  : 1등이 정답인 비율          ← 검색기 실력
  hit@k  : top_k 안에 정답이 있는 비율 ← LLM에 정답이 전달되는 비율
  MRR    : 정답 순위의 역수 평균 (1등 1.0, 2등 0.5, 3등 0.33)
  none   : 그 모델에 없는 내용 질문 → top-1 거리가 임계값 이상이면 통과
정답 판정: expect.subsection / expect.body 문자열 중 하나가 청크 소제목/본문에 포함되면 정답.
저장되는 케이스별 필드: rank, d1(1등 거리), d2(2등 거리; 1·2등 차 분석용), top3 라벨.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from siyeon.rag.chunking_lg import build_all_chunks  # noqa: E402
from siyeon.rag.chunking_samsung import build_samsung_chunks  # noqa: E402
from siyeon.rag.retrieval import LGAirconRetriever  # noqa: E402
from siyeon.rag.understand import understand  # noqa: E402

TESTSET = Path(__file__).with_name("testset.json")
OUT_DIR = Path(__file__).with_name("results")   # 자체 채점 결과 — 팀 보고서(experiments/results, compare_results.py)와 형식이 달라 따로 둔다
BRAND_KO = {"lg": "LG", "samsung": "삼성"}


def matches(chunk: dict, expect: dict) -> bool:
    return (any(s in chunk["subsection"] for s in expect.get("subsection", []))
            or any(b in chunk["body"] for b in expect.get("body", [])))


def run(retriever, cases, mode="v3", top_k=3, none_threshold=0.85, use_understanding=True):
    rows = []
    for case in cases:
        doc_id = case.get("doc_id")
        route = mode
        if mode == "v3":
            # 웹과 같은 경로: LLM 질문 이해(기능명 정규화·재작성) → 라우터. 단일 턴이라 재작성은 거의 없고 feature 판정이 핵심
            appl = retriever.appliance_for(doc_id) if doc_id else retriever.appliance_from_query(case["query"])
            und = understand(case["query"], appl) if (use_understanding and appl) else None
            res = retriever.retrieve(case["query"], doc_id=doc_id, top_k=top_k, appliance=appl, feature=und.feature if und else None)
            hits, route = res.hits, res.route
        elif mode == "dense":
            hits = retriever.search_dense(case["query"], doc_id, top_k=top_k)
        else:
            hits = retriever.search_baseline(case["query"], doc_id=doc_id, top_k=top_k)
        e = case["expect"]
        d1 = hits[0][1] if hits else 9.9
        d2 = hits[1][1] if len(hits) > 1 else 9.9
        if e.get("none"):
            # v3: 자기 매뉴얼로 근거를 대지 않았으면 통과 (route none/sibling/feature_absent). 절제 모드는 거리 임계값으로
            rank = None
            ok = route in ("none", "sibling", "feature_absent") if mode == "v3" else d1 >= none_threshold
        else:
            rank = next((i for i, (c, _) in enumerate(hits, 1) if matches(c, e)), None)
            ok = rank == 1
        brand = retriever.doc_brand.get(doc_id) if doc_id else (retriever.doc_brand.get(hits[0][0]["doc_id"]) if hits else None)
        rows.append({**case, "brand": brand, "route": route, "rank": rank, "ok": ok, "d1": round(d1, 3), "d2": round(d2, 3),
                     "top3": [f'{c["doc_id"]} > {c["subsection"][:50]} ({d:.3f})' for c, d in hits]})
    return rows


def report(rows, top_k):
    normal = [r for r in rows if not r["expect"].get("none")]
    nones = [r for r in rows if r["expect"].get("none")]

    def line(name, sub):
        if not sub: return
        t1 = sum(r["rank"] == 1 for r in sub) / len(sub); hk = sum(r["rank"] is not None for r in sub) / len(sub)
        mrr = sum(1 / r["rank"] for r in sub if r["rank"]) / len(sub)
        print(f"  {name:26s} {len(sub):4d} {t1:6.0%} {hk:6.0%} {mrr:6.2f}")

    print(f"\n  {'구분':26s} {'n':>4s} {'top-1':>6s} {'hit@'+str(top_k):>6s} {'MRR':>6s}")
    print("  " + "-" * 52)
    grp = defaultdict(list)
    for r in normal: grp[(BRAND_KO.get(r["brand"], "?"), r["category"])].append(r)
    for k in sorted(grp): line(f"{k[0]}/{k[1]}", grp[k])
    print("  " + "-" * 52)
    for b in ("lg", "samsung"): line(BRAND_KO[b], [r for r in normal if r["brand"] == b])
    for note in ("hard_pair", "vocab_gap"): line(f"[{note}]", [r for r in normal if r.get("note") == note])
    print("  " + "-" * 52)
    line("전체", normal)
    if nones:
        print(f"  기능 없음 판정: {sum(r['ok'] for r in nones)}/{len(nones)} 통과  "
              f"(실패: {[r['id'] for r in nones if not r['ok']]})")
    from collections import Counter
    print("  경로 분포:", dict(Counter(r["route"] for r in rows)))
    misses = [r for r in normal if r["rank"] != 1]
    print(f"\n=== 1등 실패 {len(misses)}건 (정답 2~3등: {sum(1 for r in misses if r['rank'])}, top-{top_k} 밖: {sum(1 for r in misses if not r['rank'])}) ===")
    for r in misses:
        print(f"  [{r['id']}] {r['query']}  → 정답 {r['rank'] or '밖'}  d1={r['d1']} d2={r['d2']}  top1: {r['top3'][0] if r['top3'] else '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest"); ap.add_argument("--mode", default="v3", choices=["v3", "dense", "base"])
    ap.add_argument("--top-k", type=int, default=3); ap.add_argument("--none-threshold", type=float, default=0.85)
    ap.add_argument("--no-understand", action="store_true", help="LLM 질문 이해 단계 끄기 (기능명 정규화 없음)")
    ap.add_argument("--ids", default="", help="쉼표로 구분한 케이스 id만 실행 (틀린 것만 빨리 재확인할 때). 결과 파일은 저장하지 않는다")
    a = ap.parse_args()
    cases = json.loads(TESTSET.read_text(encoding="utf-8"))["cases"]
    if a.ids:
        want = set(a.ids.split(","))
        cases = [c for c in cases if c["id"] in want]
    r = LGAirconRetriever(build_all_chunks() + build_samsung_chunks(), verbose=False)
    r.reranker.predict([("워밍업", "리랭커")])
    t0 = time.time()
    rows = run(r, cases, a.mode, a.top_k, a.none_threshold, use_understanding=not a.no_understand)
    print(f"\n=== 자체 테스트셋 {len(rows)}문항 · mode={a.mode}{' -understand' if a.no_understand else ''} · {time.time()-t0:.0f}s ===")
    report(rows, a.top_k)
    if a.ids:
        for r in rows:
            mark = "OK " if r["ok"] else "NG "
            print(f"  {mark}[{r['id']}] {r['query']}  route={r['route']} rank={r['rank']} d1={r['d1']}")
            print(f"      top3: {r['top3']}")
        return
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"siyeon_testset_{a.tag}.json"
    out.write_text(json.dumps({"mode": a.mode, "top_k": a.top_k, "n": len(rows), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[저장] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
