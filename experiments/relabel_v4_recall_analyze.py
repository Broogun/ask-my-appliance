"""재라벨링 v4 기준 질문 단위 Recall@K — 비용 0, 저장된 판정만 읽는다.

Recall@K = 벡터 검색(원문 질문, top-40) 상위 K개 안에 '질문에 답하는 청크'(엄격 Y / 관대 Y+P)가 하나라도 있는 문항 비율.
첫 정답 순위를 정확히 알려면 그 앞 순위 청크가 모두 판정돼 있어야 한다. 아래 세 파일이 그 조건을 채운다.
  relabel_v4_judgments_20260926.json        (본판정 692쌍: 옛 라벨 + vector top-5 + 리랭크 top-5)
  relabel_v4_sweep_judgments_20260926.json  (스윕에서 나온 133쌍)
  relabel_v4_recall_judgments_20260926.json (Recall 확정용 264쌍: 정답이 안 보이던 8문항의 top-40 전부 + 첫 정답 앞 순위)
판정되지 않은 청크를 만나면 그 문항은 '확정 불가'로 세고 출력에 표시한다(현재 0건).
"""
import json
import pathlib

R = pathlib.Path(__file__).parent / "results"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))["items"]


def main():
    items = load("relabel_v4_judgments_20260926.json") + load("relabel_v4_sweep_judgments_20260926.json") \
        + load("relabel_v4_recall_judgments_20260926.json")
    rows = json.loads((R / "relabel_v4_sweep_candidates_20260926.json").read_text(encoding="utf-8"))["rows"]
    code = {(i["qi"], i["chunk_id"]): i["code"] for i in items}
    n = len(rows)
    for name, lv in [("엄격 Y", {"Y"}), ("관대 Y+P", {"Y", "P"})]:
        first, unknown = {}, 0
        for q, r in enumerate(rows):
            first[q] = None
            for k, c in enumerate(r["vec_order"][:40], 1):
                cd = code.get((q, c))
                if cd is None:
                    unknown += 1
                    break
                if cd in lv:
                    first[q] = k
                    break
        ranks = [f for f in first.values() if f]
        print(f"=== {name} (n={n}) | 확정 불가 {unknown} | top-40 안에 정답 없음 {sum(f is None for f in first.values())} ===")
        for K in (1, 3, 5, 10, 20, 35, 40):
            print(f"  Recall@{K}: {sum(f <= K for f in ranks)}/{n} = {sum(f <= K for f in ranks) / n:.1%}")
        print(f"  MRR(vector-only, 40위까지): {sum(1 / f for f in ranks) / n:.3f} | P@1={sum(f == 1 for f in ranks) / n:.3f}")
        if lv == {"Y"}:
            print("  정답이 top-40 안에 없는 문항:", [rows[q]["question"] for q, f in first.items() if f is None])


if __name__ == "__main__":
    main()
