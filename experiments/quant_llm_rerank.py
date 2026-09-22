"""LLM listwise 리랭킹 적용 후 97문항 정량 검증. chunk_id 정확매칭 + 섹션단위
관대매칭. 이전 cross-encoder 베이스라인(64/97=66.0%, 섹션포함 71/97=73.2%)과 비교."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import importlib.util
spec = importlib.util.spec_from_file_location("exp02", Path("experiments/exp02_retrieval.py").resolve())
exp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp02)
state = exp02._load_state()

labels = json.load(open("experiments/results/faq_ground_truth_labels.json", encoding="utf-8"))
labeled = [r for r in labels if r["note"] == "labeled"]
n = len(labeled)
print(f"평가 대상: {n}개")

exact, section, miss = 0, 0, 0
detail = []
t0 = time.time()
for i, r in enumerate(labeled):
    # gather_candidates()로 my_answer()와 동일한 로직(RDB+구조적매칭+search_v2)
    # 사용 - search_v2()만 쓰면 #18/#19/#21 구조적 매칭 효과가 측정에서 빠진다.
    # my_answer()는 이 전체 목록을 그대로 답변 근거로 쓰므로(3개로 자르지 않음)
    # 여기서도 전체 목록에 정답이 있는지로 채점한다.
    cands = exp02.gather_candidates(r["question"], state)
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
    detail.append({"question": r["question"], "model": r["model"], "level": level,
                    "source_chunk_id": r["source_chunk_id"], "top3": ids})
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{n} 완료 ({time.time()-t0:.0f}초)")

elapsed = time.time() - t0
print(f"\n[LLM listwise 리랭킹] 정확:{exact}/{n}={exact/n:.1%}  섹션포함:{(exact+section)}/{n}={(exact+section)/n:.1%}  완전미스:{miss} ({elapsed:.0f}초)")
print(f"[기존 cross-encoder 베이스라인] 정확:64/97=66.0%  섹션포함:71/97=73.2%  완전미스:26")

with open("experiments/results/quant_llm_rerank_97.json", "w", encoding="utf-8") as f:
    json.dump({"exact": exact, "section": section, "miss": miss, "n": n, "elapsed": elapsed, "detail": detail}, f, ensure_ascii=False, indent=2)

print("\n=== 완전 미스 목록 ===")
for d in detail:
    if d["level"] == "miss":
        print(f"  [{d['model']}] {d['question']} (정답={d['source_chunk_id']})")
