"""chroma_db(기존, 700자/15%오버랩) vs chroma_db_team_v2(신규, 무오버랩+문단경계)
같은 97문항 라벨셋으로 MRR/Recall@K 비교 - API 호출 없음(로컬 임베딩만).

사용법: python experiments/compare_chunking_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp02_retrieval as exp02  # noqa: E402
from retrieval_metrics_offline import LABELS_PATH, RECALL_KS, compute_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(chroma_path: Path, label: str) -> dict:
    exp02.CHROMA_PATH = chroma_path
    exp02._state.clear()
    state = exp02._load_state()
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labeled = [x for x in labels if x["note"] == "labeled"]
    m = compute_metrics(labeled, state, use_content_type=True)
    print(f"\n=== {label} ({chroma_path.name}) ===")
    print(f"MRR: {m['mrr']:.4f}")
    for k in RECALL_KS:
        print(f"  Recall@{k:>3}: {m['recall_at_k'][k]*100:.1f}%")
    print(f"Precision@1: {m['precision_at_1']*100:.1f}%")
    print(f"완전 미스: {len(m['misses'])}개")
    for r in m["misses"]:
        print(f"  [{r['model']}] {r['question']}")
    return m


def main() -> None:
    before = run(ROOT / "chroma_db", "기존(700자/15%오버랩)")
    after_v3 = run(ROOT / "chroma_db_team_v3", "v3(700자/25%오버랩)")

    print("\n=== rank 변화 (기존 -> v3) ===")
    for b, a in zip(before["rows"], after_v3["rows"]):
        if b["rank"] != a["rank"]:
            print(f"  [{b['model']}] {b['question']} :: {b['rank']} -> {a['rank']}")


if __name__ == "__main__":
    main()
