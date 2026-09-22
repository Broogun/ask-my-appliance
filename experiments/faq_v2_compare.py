"""FAQ v2(5문항, 제목 비의존, Q+A 생성) 재검증 - 기존 97개 라벨링된 질문으로
켬/끔 비교. chunk_id 정확매칭 + 섹션단위 관대매칭 둘 다 리포트."""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import importlib.util

spec = importlib.util.spec_from_file_location("exp02", Path("experiments/exp02_retrieval.py").resolve())
exp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp02)
state = exp02._load_state()
real_faq_col = state["faq_collection"]
print("FAQ 컬렉션 총 개수:", real_faq_col.count() if real_faq_col else None)

labels = json.load(open("experiments/results/faq_ground_truth_labels.json", encoding="utf-8"))
labeled = [r for r in labels if r["note"] == "labeled"]
print(f"평가 대상: {len(labeled)}개")


def get_section_title(chunk_id):
    res = state["collection"].get(ids=[chunk_id], include=["metadatas"])
    return res["metadatas"][0]["section_title"] if res["metadatas"] else None


def run(use_faq):
    state["faq_collection"] = real_faq_col if use_faq else None
    exact, section, miss = 0, 0, 0
    detail = []
    t0 = time.time()
    for r in labeled:
        where = {"product_model": r["model"]}
        cands = exp02.search_v2(r["question"], state, top_k=3, where=where)
        ids = [c["id"] for c in cands]
        titles = [c["metadata"].get("section_title") for c in cands]
        if r["source_chunk_id"] in ids:
            exact += 1
            level = "exact"
        elif r["source_section_title"] in titles:
            section += 1
            level = "section"
        else:
            miss += 1
            level = "miss"
        detail.append({"question": r["question"], "model": r["model"], "level": level})
    elapsed = time.time() - t0
    return exact, section, miss, elapsed, detail


n = len(labeled)
off_exact, off_section, off_miss, off_t, off_detail = run(False)
on_exact, on_section, on_miss, on_t, on_detail = run(True)

print(f"\n[FAQ 끔]     정확:{off_exact}/{n}={off_exact/n:.1%}  섹션포함:{(off_exact+off_section)}/{n}={(off_exact+off_section)/n:.1%}  완전미스:{off_miss} ({off_t:.0f}초)")
print(f"[FAQ v2 켬] 정확:{on_exact}/{n}={on_exact/n:.1%}  섹션포함:{(on_exact+on_section)}/{n}={(on_exact+on_section)/n:.1%}  완전미스:{on_miss} ({on_t:.0f}초)")

print("\n=== 레벨이 달라진 질문 (개선/후퇴) ===")
n_changed = 0
for o, on in zip(off_detail, on_detail):
    if o["level"] != on["level"]:
        rank = {"miss": 0, "section": 1, "exact": 2}
        change = "개선" if rank[on["level"]] > rank[o["level"]] else "후퇴"
        print(f"[{change}] {o['model']} | {o['question']} : {o['level']} -> {on['level']}")
        n_changed += 1
if n_changed == 0:
    print("  (변화 없음)")
