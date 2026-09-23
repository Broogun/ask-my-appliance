# experiments/ — 실험 기록

제품 코드가 아니라 **선택의 근거를 남기는 기록**이다. 청킹·검색·리랭킹 전략을 비교한 실험 파일과
결과 JSON, 팀 공용 평가 하니스가 들어 있다.

여기서 검증된 조합이 `web/`의 서비스 파이프라인이 됐다. 제품 쪽 구조는
[`../docs/03-architecture.md`](../docs/03-architecture.md), 선택 이유는
[`../docs/04-rag-pipeline.md`](../docs/04-rag-pipeline.md)에 정리돼 있다.

## 구조

```
harness/evaluate.py     공용 평가 하니스 — MODEL_QUESTIONS(900문항) · SYSTEM_PROMPT · 채점·저장
template_exp.py         실험 시작 템플릿
planning_template.md    사전 기획서 템플릿
RULES.md                실험 규칙 (통일/자유 항목, 브랜치 전략, 작성 규칙)
compare_results.py      팀 전체 결과 비교 보고서 생성
rag_tutorial.ipynb      청킹·임베딩 방식 비교 학습 노트북
web_baseline_eval.py    실제 서비스 경로(web/rag_service) 평가
results/                실험 결과 JSON (커밋 대상)
reports/latest.md       자동 생성 비교 보고서
siyeon/                 팀원 개인 작업 폴더 (검색기·RDB·자체 테스트셋)
exp01_toc_chunking.py, _exp02.py   팀원 개인 실험 (exp02가 통합 베이스라인)
```

## 통일 기준 (수정 금지)

비교가 성립하려면 채점 기준이 같아야 한다. 아래는 `harness/evaluate.py`에 있고 **고치지 않는다.**

| 항목 | 값 |
|---|---|
| 평가 질문 | `MODEL_QUESTIONS` — 모델 60개 × 3라운드 × 5문항 = 900문항 |
| 답변 프롬프트 | `SYSTEM_PROMPT` |
| LLM | gpt-4o-mini |
| 반환 형식 | `{"answer": str, "candidates": [{"id", "text", "distance"}]}` |

**자유:** 청킹 방식 / 임베딩 모델 / 벡터 DB / 검색 전략. 이게 비교 대상이다.

## 실험 시작하기

```bash
git checkout -b exp/{이름}
cp experiments/template_exp.py experiments/{이름}_exp01.py
```

`my_answer()` 하나만 구현한다. 나머지(질문 순회·채점·저장)는 하니스가 맡는다.

```python
from experiments.harness.evaluate import SYSTEM_PROMPT, run_and_save

STRATEGY = {                      # 보고서에 자동 포함되니 반드시 채운다
    "chunking": "...", "chunking_reason": "...",
    "embedding": "...", "embedding_reason": "...",
    "db": "...", "db_reason": "...",
    "retrieval": "...", "retrieval_reason": "...",
}

def my_answer(query: str) -> dict:
    ...
    return {"answer": ..., "candidates": [...]}
```

```bash
python experiments/{이름}_exp01.py          # → results/{이름}_exp01.json
git add experiments/{이름}_exp01.py experiments/results/{이름}_exp01.json
git commit -m "[{이름}] exp01: 한 줄 설명"
git push origin exp/{이름}
```

`exp/*` 브랜치에 결과 JSON을 push하면 GitHub Actions가 팀 전체 비교 보고서를 PR 댓글로 단다
(`.github/workflows/eval_report.yml` → `reports/latest.md`).

자세한 규칙은 [`RULES.md`](RULES.md)를 본다.

## 평가 실행

```bash
# 실제 서비스 경로 평가 (round1 300문항)
python experiments/web_baseline_eval.py --n 5      # 파일럿
python experiments/web_baseline_eval.py            # 전체
python experiments/web_baseline_eval.py --resume   # 중단 지점부터

# 검색 품질 재현: 정답 라벨링(청크 전체 기준, 순환성 없음) -> 지표 -> pool_size/top_k 선정
python experiments/faq_label_ground_truth_v3.py [--resume]
python experiments/retrieval_metrics_v3.py
python experiments/pool_size_rerank_sweep_v3.py [--resume]
python experiments/topk_sweep_v3.py --pool 35 [--resume]
python experiments/rerank_shuffle_test_v2.py [--resume]
python experiments/rerank_ensemble_check_v3.py [--resume]

# RAGAS
python experiments/ragas_eval.py
```

> **OpenAI 비용이 발생한다.** 300문항 전체 실행은 질문당 리랭킹·생성·판정으로 약 900회를 호출한다.
> 팀 규칙상 전체 실행 전에 공유한다.

지표의 의미와 해석은 [`../docs/05-evaluation.md`](../docs/05-evaluation.md)에 있다.

## 주요 실험 기록

| 실험 | 확인한 것 |
|---|---|
| `exp01_toc_chunking` → `exp02` | TOC 청킹 + 삼성 PDF 손상 방어. exp02가 통합 베이스라인 |
| `faq_label_ground_truth_v3.py` | 청크 전체 기준 정답 라벨링(순환성 없음, 목차/포괄챕터 제외 가드레일) — 104/150건 확보 |
| `retrieval_metrics_v3.py` | MRR·Recall@k(청크 전체 라벨 기준) |
| `pool_size_rerank_sweep_v3.py` | pool_size(15/20/35/40) 리랭킹까지 포함해 실측 비교 — 35 유지(40도 동일), 리랭킹 정확도가 진짜 병목 |
| `topk_sweep_v3.py` | 최종 근거 개수(top_k=1/3/5/10) 히트율 비교 — 5 유지(10은 수확체감) |
| `rerank_shuffle_test_v2.py` | 리랭킹 위치 편향("lost in the middle") 검증 — 순서 셔플은 전체 히트율을 오히려 낮춤 |
| `rerank_cot_test_v2.py` | 후보별 개별 판단(CoT) 프롬프트 시도 — 전체 히트율이 더 크게 떨어져 기각 |
| `rerank_ensemble_check_v3.py` | 정렬+셔플 결과 합집합(앙상블) 검증 — 실제 존재하는 개선 여지지만 비용 2배라 미도입 |
| `faq_*` | 예상질문(FAQ) 생성·필터·효과 측정 |
| `tagging_vs_structural_coverage.py` | 의미 유형 태깅 vs 구조적 챕터 매칭 |
| `siyeon/` | CrossEncoder+RRF 파이프라인, 에러코드 RDB 설계 |
| `web_baseline_eval.py` | 통합 후 실제 서비스 경로 baseline |

## 왜 코드를 합치지 않았나

4명이 각자 다른 방식으로 구현한 뒤 **같은 질문셋으로 비교**하는 것이 목적이었다. 공통 하니스만
공유하고 파이프라인은 분리해, 누구의 어떤 선택이 성능 차이를 만들었는지 추적할 수 있게 했다.
통합 후에도 실험 파일을 남겨둔 이유는 선택의 근거가 여기 있기 때문이다.
