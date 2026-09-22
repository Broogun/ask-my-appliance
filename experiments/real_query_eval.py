"""실제 사용자 질문(네이버 지식in/카페) 기반 평가.

출처: https://docs.google.com/spreadsheets/d/1fXAnYovErk1i15vYgbmWcTvnUImmakZ8LXpjPD31Jo4
900문항(MODEL_QUESTIONS)은 모델명을 명시한 정형 질문이라, 실제 서비스 사용 패턴(등록된 가전에
증상만 묻고 모델명은 안 씀)과 문체가 다르다. 이 데이터셋은 실제 지식in/카페 질문 원문 + 사람이
PDF를 직접 읽고 확인한 정답 페이지 + must_not_claim(이렇게 말하면 안 됨, 환각 함정)을 갖고 있어
900문항으로는 못 재는 두 가지를 잰다:
  1. 실제 콜로퀄 질문에서 검색이 사람이 확인한 정답 페이지를 찾는가
  2. 매뉴얼이 진단하지 않는 원인(컴프레서 고장 등)을 답변이 지어내지 않는가

⚠ OpenAI 비용 발생(질문당 리랭킹+생성 최대 2회, must_not_claim 판정 1회 - 팀 규칙대로 실행 전 승인받음, 2026-09-22).

사용법: python experiments/real_query_eval.py
결과: experiments/results/real_query_eval.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402
from web import rag_service as s  # noqa: E402

OUT_PATH = ROOT / "experiments" / "results" / "real_query_eval.json"
client = OpenAI()

# (raw_question_id, 원문 질문) — Sheet1(참고_id)에서 그대로 옮김. 정형 질문이 아니라 실제 질문 원문을 쓴다.
RAW_QUESTIONS = {
    1: "위아래 문열리는 2도어 일반형 냉장고입니다. 삼성 디지털인버터 냉장고라적혀있고  현재 냉동실 냉장고 모두 작동을 안하는 것 같아요.. 조명도 전원도 모두 들어오고 설정온도를 아무리 낮게 눌러도 냉동실 15도 냉장고9도 이렇게 올라갑니다.. 이게 삼일째에요 점점 냉장고 내부온도는 올라가는.것 같더라구요",
    2: "10년이상 사용한 lg 냉장고 소음때문에 잠을 못 자고 있습니다",
    3: "삼성 페스티벌때 키친핏 4도어 냉장고를 구매, 다들 냉장고 소음 어떻게 해결하셨나요?",
    4: "김치냉장고에 김치가 얼었습니다....",
    5: "냉장고 문이 안 닫혀서 계속 어저밤주터 삐삐 소리가 나는데 출장수리도 내일 불러야 될 거 같습니다.",
    7: "세탁기 작동이 중간에 멈췄습니다. 어떻게 해결하나요??",
    8: "WW90T3100KW 이게 모델번호 같습니다  세탁기 돌리고 거의 6분? 남앗을때 탈수진행하려고 하는 것 같은데 모터?돌아가는 소리가 엄청 크게나고 세탁기가 안 돌아갑니다.",
    9: "삼성세탁 시작 누르면 저렇게 시작은 안되고 계속 무한로딩만 되는데 왜이러나요?",
    10: "스타일러에서 소음이 심해졌는데 당장 고쳐야하는 고장일까요?",
    11: "갑자기 배수가 안되어서요 OE 에러도 떠요",
    12: "잘 돌아가던 에어컨 갑자기 고장.. 켜지지가 않아요",
    13: "갑자기 에어컨이 고장났는데... LG 휘슬 벽걸이 에어컨입니다 증상은 전원 버튼이 2번 깜박이고 밑에 실외기와 예약 사이에 있는 부분?의 빛이 3번 깜박입니다 껐다켜도 똑같아요 버튼색은 둘다 하얀색으로 깜박이고 숫자로 표시되는 에어컨이 아니라 숫자코드는 모르겠습니다...",
    14: "에어컨인데 닫힌상태에서 바람이 나와 시원하지않고 물이 맺힙니다.",
    15: "시원한 바람도 살짝만 나오고 그리고 켜졌다가 한 5분 있다가 꺼지네요",
    16: "에어컨이 안돌아요 아니 실외기가 안돌아요",
    17: "에어컨에 ch54라는 오류코드가 뜨는데 뭔가요ㅠㅠ 고치는 방법 있을까요?",
    18: "삼성에어컨 c101에러코드 자꾸 뜨는데",
}

# Sheet2(수작업 정답 라벨셋)를 그대로 옮김. gt_page: 사람이 PDF를 직접 읽고 확인한 정답 페이지.
_SAME_AS_ROW2 = "컴프레서 고장이다 / 냉매가 샜다"
CASES = [
    # (raw_question_id, manual_id, gt_page, must_not_claim, answerable)
    (1, "RF_RT42CG6024S9", 24, None, "yes"),
    (1, "RF_RT53DG7A1CWW", 23, None, "yes"),
    (2, "RF_W826AAA492", 46, _SAME_AS_ROW2, "yes"),
    (2, "RF_S835S31", 43, _SAME_AS_ROW2, "yes"),
    (2, "RF_M402ND", 36, _SAME_AS_ROW2, "yes"),
    (3, "RF_RF85B90P1AP", 71, "압축기가 고장났다", "yes"),
    (4, "RF_K135LW123", 20, "이건 센서 고장입니다", "yes"),
    (5, "RF_B242S32", 19, "경보음 자체가 고장의 증거다(삐삐 소리는 문이 안 닫힌 결과일 뿐)", "yes"),
    (7, "WM_WF21T6500KW", 71, "모터가 고장났다(매뉴얼은 부품 진단 안 함)", "yes"),
    (8, "WM_WA19CG6745BV", 49, None, "yes"),
    (9, "WM_WF21T6500KW", 71, "소프트웨어 오류다 / 메인보드 고장이다", "partial"),
    (10, "ST_SC3GBE50", 33, "컴프레서가 고장났다", "yes"),
    (10, "WM_WF17M9100KG", 40, None, "yes"),
    (10, "WM_FC4KC", 39, None, "yes"),
    (11, "WM_WA80F19E8W", 50, None, "yes"),
    (12, "AC_FQ17FN5BDN", 77, None, "yes"),
    (13, "AC_SQ07GA3WBN", 46, None, "partial"),
    (14, "AC_AR11B9150HZT", 38, None, "partial"),
    (15, "AC_FQ16FV6EDN", 74, None, "yes"),
    (16, "AC_FQ18GN7BKN", 77, None, "partial"),
]
# 에러코드 경로 케이스: PDF 정답 페이지 대신 코드 매칭을 확인. manual_id는 해당 브랜드/카테고리의
# 아무 실제 모델(에러코드는 브랜드+카테고리 공용 문서라 어느 모델로 등록됐든 같은 코드를 조회한다).
ERROR_CASES = [
    (11, "WM_FC4KC", "OE"),
    (17, "AC_FQ16FV6EDN", "CH54"),
    (18, "AC_AR11B9150HZT", "C101"),
]

MUST_NOT_CLAIM_JUDGE = (
    "너는 가전 설명서 챗봇의 답변을 검수하는 심사위원이다. [답변]이 [금지된 주장]을 사실인 것처럼 "
    "단정하거나 강하게 암시하면 'VIOLATION', 그렇지 않으면(그런 가능성도 있을 수 있다는 식의 유보적 "
    "언급 정도는 위반 아님) 'OK'만 출력해라. 다른 설명 없이 한 단어만.\n\n"
    "[금지된 주장]\n{claim}\n\n[답변]\n{answer}"
)


def judge_must_not_claim(answer: str, claim: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": MUST_NOT_CLAIM_JUDGE.format(claim=claim, answer=answer)}],
    )
    return r.choices[0].message.content.strip()


def page_hit(used, gt_page: int, tolerance: int = 1) -> bool:
    for c, _ in used:
        ps, pe = c.get("page_start"), c.get("page_end")
        if ps is None:
            continue
        if ps - tolerance <= gt_page <= (pe or ps) + tolerance:
            return True
    return False


def main() -> None:
    results = []
    for rid, manual_id, gt_page, must_not_claim, answerable in CASES:
        query = RAW_QUESTIONS[rid]
        res = s.retrieve(query, manual_id=manual_id, top_k=5)
        answer = "".join(s.stream_answer(query, res, history=[]))
        hit = page_hit(res.used, gt_page)
        row = {
            "raw_question_id": rid, "query": query, "manual_id": manual_id, "gt_page": gt_page,
            "answerable": answerable, "route": res.route, "page_hit": hit,
            "retrieved_pages": [[c.get("page_start"), c.get("page_end")] for c, _ in res.used],
            "answer": answer, "must_not_claim": must_not_claim,
        }
        if must_not_claim:
            row["claim_verdict"] = judge_must_not_claim(answer, must_not_claim)
        results.append(row)
        print(f"  #{rid} {manual_id}: route={res.route} page_hit={hit}" + (f" claim={row.get('claim_verdict')}" if must_not_claim else ""), flush=True)

    for rid, manual_id, code in ERROR_CASES:
        query = RAW_QUESTIONS[rid]
        res = s.retrieve(query, manual_id=manual_id, top_k=5)
        answer = "".join(s.stream_answer(query, res, history=[]))
        code_hit = res.route == "error_code" and any(code.upper() in (c.get("section", "") + c.get("body", "")).upper() for c, _ in res.used)
        results.append({
            "raw_question_id": rid, "query": query, "manual_id": manual_id, "expected_code": code,
            "route": res.route, "code_hit": code_hit, "answer": answer,
        })
        print(f"  #{rid} {manual_id} (에러코드 {code}): route={res.route} code_hit={code_hit}", flush=True)

    n_page_cases = [r for r in results if "page_hit" in r]
    n_hit = sum(r["page_hit"] for r in n_page_cases)
    claim_cases = [r for r in results if "claim_verdict" in r]
    n_violation = sum(r["claim_verdict"] == "VIOLATION" for r in claim_cases)
    n_error_cases = [r for r in results if "code_hit" in r]
    n_code_hit = sum(r["code_hit"] for r in n_error_cases)

    summary = {
        "page_hit_rate": f"{n_hit}/{len(n_page_cases)}",
        "must_not_claim_violations": f"{n_violation}/{len(claim_cases)}",
        "error_code_hit_rate": f"{n_code_hit}/{len(n_error_cases)}",
    }
    print("\n요약:", summary)
    OUT_PATH.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
