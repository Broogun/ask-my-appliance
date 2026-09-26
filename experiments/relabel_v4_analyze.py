"""재라벨링(v4) 판정 집계 — 비용 0, 저장된 판정 파일만 읽는다.

입력: experiments/results/relabel_v4_judgments_20260926.json
  질문마다 (v3 라벨 + vector top-5 + gpt-4o-mini 리랭크 top-5)의 합집합을
  출처를 가린 채 Y/P/N으로 판정한 코드. 판정자는 Claude 1명(사람 교차검증 없음).
출력: 엄격(Y)·관대(Y+P) 기준 Hit@5·MRR@5(vector-only 대 리랭크), 옛 라벨 품질, 부호검정.
"""
import collections
import json
import math
import pathlib

SRC = pathlib.Path(__file__).parent / "results" / "relabel_v4_judgments_20260926.json"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - a) / d * 100, 1), round((c + a) / d * 100, 1)


def sign_p(win, lose):
    n = win + lose
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(win, lose) + 1)) / 2 ** n)


def main():
    items = json.loads(SRC.read_text(encoding="utf-8"))["items"]
    by_q = collections.defaultdict(list)
    for it in items:
        by_q[it["qi"]].append(it)
    print("판정 분포", dict(collections.Counter(i["code"] for i in items)), "/", len(items))

    def hit(qi, src, levels):
        return any(i["code"] in levels for i in by_q[qi] if i[src])

    n = len(by_q)
    for name, lv in [("엄격 Y", {"Y"}), ("관대 Y+P", {"Y", "P"})]:
        v = {q: hit(q, "in_vec5", lv) for q in by_q}
        m = {q: hit(q, "in_mini5", lv) for q in by_q}
        win = sum(m[q] and not v[q] for q in by_q)
        lose = sum(v[q] and not m[q] for q in by_q)
        print(f"[{name}] n={n} vector-only {sum(v.values())} {wilson(sum(v.values()), n)} | "
              f"리랭크 {sum(m.values())} {wilson(sum(m.values()), n)} | "
              f"리랭크만 {win} / vector만 {lose} (부호검정 p={sign_p(win, lose):.4f})")

    old_v = sum(any(i["is_v3_label"] and i["in_vec5"] for i in by_q[q]) for q in by_q)
    old_m = sum(any(i["is_v3_label"] and i["in_mini5"] for i in by_q[q]) for q in by_q)
    print(f"[옛 라벨] vector-only {old_v} | 리랭크 {old_m}")
    lab = collections.Counter(next(i["code"] for i in by_q[q] if i["is_v3_label"]) for q in by_q)
    print("옛 라벨 자체의 품질(Y/P/N)", dict(lab))
    no_y = [q for q in by_q if not any(i["code"] == "Y" for i in by_q[q])]
    print("후보 중 Y가 하나도 없는 질문", len(no_y), [by_q[q][0]["question"] for q in no_y])


if __name__ == "__main__":
    main()
