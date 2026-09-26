"""재라벨링 v4 기준 pool_size·top_k 스윕 집계 — 비용 0, 저장된 결과·판정만 읽는다.

입력: results/relabel_v4_judgments_20260926.json        (v4 본판정 692쌍)
      results/relabel_v4_sweep_judgments_20260926.json  (스윕에서 새로 나온 청크 판정 133쌍)
      results/relabel_v4_sweep_candidates_20260926.json (gpt-4o-mini 리랭크 결과: pool 15/20/40 top-5, pool 35 top-10 순서)
pool 35 의 top-5 는 v4 본판정의 in_mini5 를 쓴다(같은 실행). top_k 스윕은 top-10 요청 순서의 앞 k개다(기존 topk_sweep_v3 방식).
"""
import collections
import json
import math
import pathlib

R = pathlib.Path(__file__).parent / "results"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - a) / d * 100, 1), round((c + a) / d * 100, 1)


def sign_p(win, lose):
    n = win + lose
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(win, lose) + 1)) / 2 ** n)


def main():
    v4 = json.loads((R / "relabel_v4_judgments_20260926.json").read_text(encoding="utf-8"))["items"]
    sw = json.loads((R / "relabel_v4_sweep_judgments_20260926.json").read_text(encoding="utf-8"))["items"]
    rows = json.loads((R / "relabel_v4_sweep_candidates_20260926.json").read_text(encoding="utf-8"))["rows"]
    code = {(i["qi"], i["chunk_id"]): i["code"] for i in v4 + sw}
    mini35 = collections.defaultdict(list)
    for i in v4:
        if i["in_mini5"]:
            mini35[i["qi"]].append(i["chunk_id"])
    n = len(rows)
    for qi, r in enumerate(rows):  # 모든 후보가 판정됐는지 확인
        for k in ("pool15_top5", "pool20_top5", "pool40_top5", "pool35_top10_order"):
            for c in r[k]:
                assert (qi, c) in code, (qi, c)

    def hit(qi, ids, lv):
        return any(code[(qi, c)] in lv for c in ids)

    for name, lv in [("엄격 Y", {"Y"}), ("관대 Y+P", {"Y", "P"})]:
        print(f"===== {name} (n={n}) =====")
        pools = {15: [r["pool15_top5"] for r in rows], 20: [r["pool20_top5"] for r in rows],
                 35: [mini35[qi] for qi in range(n)], 40: [r["pool40_top5"] for r in rows]}
        h = {p: [hit(qi, pools[p][qi], lv) for qi in range(n)] for p in pools}
        print("[pool] " + " | ".join(f"{p}: {sum(h[p])} {wilson(sum(h[p]), n)}" for p in pools))
        for p in (15, 20, 40):
            win = sum(h[35][q] and not h[p][q] for q in range(n))
            lose = sum(h[p][q] and not h[35][q] for q in range(n))
            print(f"   35 대 {p}: 35만 {win} / {p}만 {lose} (부호검정 p={sign_p(win, lose):.3f})")
        ks = {1: 1, 3: 3, 5: 5, 10: 10}
        ht = {k: [hit(qi, r["pool35_top10_order"][:k], lv) for qi, r in enumerate(rows)] for k in ks}
        hv = {k: [hit(qi, r["vec_order"][:k], lv) for qi, r in enumerate(rows)] for k in (1, 3, 5)}
        print("[top_k 리랭크] " + " | ".join(f"{k}: {sum(ht[k])}" for k in ks))
        print("[top_k vector-only] " + " | ".join(f"{k}: {sum(hv[k])}" for k in hv))
        prev = None
        for k in ks:
            if prev:
                win = sum(ht[k][q] and not ht[prev][q] for q in range(n))
                lose = sum(ht[prev][q] and not ht[k][q] for q in range(n))
                print(f"   {prev}→{k}: 좋아진 {win} / 나빠진 {lose} (p={sign_p(win, lose):.3f})")
            prev = k


if __name__ == "__main__":
    main()
