"""이미 만든 round1 공식 결과(박형건_round1_official_20260916.json)를 keyword_hit
대신 LLM 판정으로 재채점한다. 판정자는 질문+최종답변+실제 근거후보를 같이 보고
"진짜 문서 내용으로 정확히 답했는지"를 본다 - 이러면 거짓 겸손(답할 수 있는데
못 찾았다고 함)과 거짓 확신(무관한 근거로 자신 있게 답함) 둘 다 잡아낼 수 있다."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location("exp02", Path(__file__).resolve().parent / "박형건_exp02.py")
exp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp02)

JUDGE_PROMPT = (
    "너는 가전제품 RAG 챗봇의 답변을 채점하는 심사위원이야. [질문], [답변], "
    "[근거후보](실제 검색된 문서 조각 전문)를 보고 판정해.\n"
    "\n"
    "**가장 중요한 규칙: [근거후보]에 실제로 적힌 문장만 근거로 삼아라. 네가 "
    "일반적으로 알고 있는 지식(예: '이 브랜드/제품군은 보통 이런 기능이 있다')으로 "
    "근거후보의 내용을 추측하거나 보충하지 마라 - 반드시 [근거후보] 원문에서 "
    "그 근거가 되는 문구를 직접 인용할 수 있어야 한다.**\n"
    "\n"
    "다음 중 하나로 분류해:\n"
    "SUCCESS - 답변이 근거후보의 실제 내용을 정확히 반영해서 질문에 구체적으로 답함\n"
    "FALSE_DECLINE - 근거후보 안에 명백히 답이 되는 내용이 있는데도 답변이 "
    "\"문서에 없다\"거나 얼버무리며 일반 상식으로 대체함 (반드시 근거후보에서 "
    "그 답이 되는 문구를 직접 인용할 수 있어야만 이 라벨을 쓸 것 - 인용할 문구가 "
    "없으면 GENUINE_NO_ANSWER로 분류해)\n"
    "FALSE_CONFIDENCE - 근거후보가 질문과 무관한데도 답변이 자신 있게(일반 "
    "상식/추측으로) 답변함 - 명시적으로 '없다'고 하지 않고 그럴듯하게 답변한 경우\n"
    "GENUINE_NO_ANSWER - 근거후보에 진짜 관련 내용이 없고 답변도 정직하게 "
    "모른다고 함(이건 실패가 아니라 올바른 동작)\n"
    "형식: 한 줄에 \"라벨|인용문구(또는 없음)|한줄이유\" 만 출력. 다른 설명 없이.\n"
    "FALSE_DECLINE으로 판정할 땐 인용문구 자리에 근거후보 원문에서 그대로 "
    "가져온 문구를 반드시 넣어라."
)


def judge_one(r: dict) -> dict:
    # 200자로 자르면 뒷부분에 있는 실제 근거를 판정자가 못 보고 GENUINE_NO_ANSWER/
    # FALSE_CONFIDENCE로 오판할 위험이 있어서 전체 텍스트를 그대로 보여준다
    # (청크당 최대 700자+15%오버랩이라 비용 영향은 미미함).
    cands_text = "\n\n".join(
        f"({c.get('id')}) {c.get('text', '')}" for c in r["candidates"][:3]
    )
    user_msg = f"[질문]\n{r['question']}\n\n[답변]\n{r['answer']}\n\n[근거후보]\n{cands_text}"
    raw = exp02.generate_openai(JUDGE_PROMPT, user_msg, temperature=0.0).strip()
    parts = raw.split("|", 2)
    label = parts[0].strip()
    quote = parts[1].strip() if len(parts) > 1 else ""
    reason = parts[2].strip() if len(parts) > 2 else ""
    if label not in {"SUCCESS", "FALSE_DECLINE", "FALSE_CONFIDENCE", "GENUINE_NO_ANSWER"}:
        label = "PARSE_ERROR"
    # 인용 검증: FALSE_DECLINE인데 인용문구가 실제 근거후보 원문에 없으면
    # (모델이 인용을 지어낸 것) GENUINE_NO_ANSWER로 강등 - "무풍" 사례처럼
    # 근거에 없는 내용을 "있다"고 우기는 환각을 코드로 한 번 더 걸러낸다.
    #
    # ⚠ 첫 구현(2026-09-17 오전)은 단순 `in` 문자열 검사라 PDF 원문의 줄바꿈과
    # 판정자가 인용하며 붙인 따옴표(" ") 때문에 "실제로 맞게 인용했는데도
    # 불일치"로 오판하는 버그가 있었음(FQ18GC1EHN "제습 모드" 사례로 확인 -
    # 원문엔 "쾌적제습으로 설정되어\n있습니다", 인용은 " ...설정되어 있습니다"
    # 로 줄바꿈만 다름). 공백/줄바꿈 정규화 + 따옴표 제거 후 비교하도록 수정.
    def _normalize(s: str) -> str:
        s = s.strip().strip("\"'“”‘’")
        return re.sub(r"\s+", " ", s)

    if label == "FALSE_DECLINE":
        norm_quote = _normalize(quote)
        norm_cands = re.sub(r"\s+", " ", cands_text)
        if not norm_quote or norm_quote in ("없음", "None", "N/A") or norm_quote not in norm_cands:
            label = "GENUINE_NO_ANSWER"
            reason = f"[자동강등: 인용문구 원문 불일치 - 원래판정 FALSE_DECLINE, 인용='{quote}'] {reason}"
    return {"label": label, "reason": reason, "quote": quote}


def _checkpoint_path(round_name: str, date: str = "20260916") -> Path:
    return Path(f"experiments/results/judge_checkpoint_{round_name}_{date}.json")


def _judge_one_safe(r: dict) -> dict:
    """RateLimitError가 재시도 5회를 넘겨도 스레드가 예외로 전체 스크립트를
    죽이지 않도록 감싼다 - 어제 태깅 작업/오늘 이 스크립트 둘 다 이 이유로
    300개 중 240개까지 갔다가 진행분을 통째로 날린 적이 있어서(체크포인트도
    없었음) 반영. 실패하면 ERROR 라벨로 표시하고 나중에 순차 재시도한다."""
    try:
        return judge_one(r)
    except Exception as e:
        return {"label": "ERROR", "reason": f"{type(e).__name__}: {e}", "quote": ""}


if __name__ == "__main__":
    import argparse
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser()
    parser.add_argument("round_name", nargs="?", default="round1", choices=["round1", "round2", "round3"])
    parser.add_argument("--date", default="20260916", help="재생성 날짜(입력 official 파일 선택용, 예: 20260917)")
    args = parser.parse_args()

    in_path = Path(f"experiments/results/박형건_{args.round_name}_official_{args.date}.json")
    data = json.load(open(in_path, encoding="utf-8"))
    results = data["results"]
    print(f"판정 대상: {len(results)}개")

    ckpt_path = _checkpoint_path(args.round_name, args.date)
    checkpoint: dict[str, dict] = {}
    if ckpt_path.exists():
        checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
        print(f"체크포인트에서 {len(checkpoint)}개 재개")

    def _key(i: int, r: dict) -> str:
        return f"{i}:{r['model']}:{r['question']}"

    def _save_checkpoint() -> None:
        ckpt_path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    judged: list[dict | None] = [None] * len(results)
    todo = []
    for i, r in enumerate(results):
        k = _key(i, r)
        if k in checkpoint:
            judged[i] = checkpoint[k]
        else:
            todo.append((i, r))

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_judge_one_safe, r): (i, r) for i, r in todo}
        n_done = 0
        for fut in as_completed(futures):
            i, r = futures[fut]
            res = fut.result()
            judged[i] = res
            checkpoint[_key(i, r)] = res
            n_done += 1
            if n_done % 10 == 0:
                _save_checkpoint()
            if n_done % 20 == 0:
                print(f"  {n_done}/{len(todo)} 신규 완료 ({time.time()-t0:.0f}초)", flush=True)
    _save_checkpoint()

    # ERROR로 남은 것들만 순차로 한 번 더 재시도(병렬 대신 순차라 rate limit 회복 여유를 줌)
    error_idxs = [i for i, j in enumerate(judged) if j and j["label"] == "ERROR"]
    if error_idxs:
        print(f"\nERROR {len(error_idxs)}건 순차 재시도")
        for i in error_idxs:
            time.sleep(3)
            r = results[i]
            res = _judge_one_safe(r)
            judged[i] = res
            checkpoint[_key(i, r)] = res
            print(f"  재시도 [{i}] -> {res['label']}", flush=True)
        _save_checkpoint()

    from collections import Counter
    label_counts = Counter(j["label"] for j in judged)
    print(f"\n완료: {time.time()-t0:.0f}초")
    for label, cnt in label_counts.most_common():
        print(f"  {label}: {cnt} ({cnt/len(judged):.1%})")

    out = []
    for r, j in zip(results, judged):
        out.append({
            **r,
            "llm_judge_label": j["label"],
            "llm_judge_reason": j["reason"],
            "llm_judge_quote": j.get("quote", ""),
        })

    # v2: 200자 절단 제거 + 인용 강제 + 인용-원문 불일치 자동강등 반영판
    # (2026-09-17) - 기존 v1 파일(_llm_judged_20260916.json)은 비교용으로 보존
    out_path = Path(f"experiments/results/박형건_{args.round_name}_llm_judged_v2_src{args.date}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"label_counts": dict(label_counts), "results": out}, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
    if ckpt_path.exists():
        ckpt_path.unlink()

    print("\n=== FALSE_DECLINE 샘플 (최대 10개) ===")
    shown = 0
    for r, j in zip(results, judged):
        if j["label"] == "FALSE_DECLINE" and shown < 10:
            print(f"- [{r['model']}] {r['question']} :: {j['reason']}")
            shown += 1

    print("\n=== FALSE_CONFIDENCE 샘플 (최대 10개) ===")
    shown = 0
    for r, j in zip(results, judged):
        if j["label"] == "FALSE_CONFIDENCE" and shown < 10:
            print(f"- [{r['model']}] {r['question']} :: {j['reason']}")
            shown += 1
