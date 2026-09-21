"""AnswerRelevancy가 낮게 나온 게 "페이지 인용/문의 꼬리말 같은 부가 텍스트"
때문인지, 아니면 지표 자체의 구조적 상한 때문인지 분리 검증.

방법: 실제 round3 답변 7개를 골라, 내용은 그대로 두고 페이지 인용/인트로·
아웃트로 상투구만 손으로 제거한 "핵심만" 버전을 만들어 같은 질문으로 원본 vs
축약본 AnswerRelevancy를 비교한다. 답변 rewrite는 API 호출 없이 직접 작성,
채점만 gpt-4o-mini+text-embedding-3-small 사용(약 14회 호출 - 원본 7 + 축약 7).

사용법: python experiments/test_terse_answer_relevancy.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    {
        "question": "J825MEE042 냉장고 설치할 때 주의사항 알려줘",
        "original": (
            "J825MEE042 냉장고를 설치할 때 주의사항은 다음과 같습니다:\n\n"
            "1. **간격 유지**: 냉장고와 벽면 사이에는 최소 10 cm의 간격을 유지해야 합니다. "
            "너무 가까우면 냉각 성능이 떨어지고, 냉장고 겉에 이슬이 맺힐 수 있으며 전기요금이 증가할 수 있습니다.\n\n"
            "2. **문 위 간격**: 냉장고 제품 문 위쪽과 설치장 사이에는 최소 2.5 cm의 간격을 유지해야 합니다. "
            "이는 조도 센서가 올바르게 작동하기 위해 필요합니다.\n\n"
            "3. **적정 온도**: 냉장고를 설치할 장소의 주위 온도는 5℃ 이상 43℃ 이하이어야 합니다.\n\n"
            "냉장고를 제대로 설치하고, 주의사항을 지키는 것이 최상의 성능을 발휘하는 데 도움이 됩니다.\n\n"
            "(참고: 30페이지)"
        ),
        "terse": (
            "1. 벽면과 최소 10cm 간격 - 가까우면 냉각 성능 저하, 이슬 맺힘, 전기요금 증가.\n"
            "2. 문 위쪽과 설치장 사이 최소 2.5cm 간격 - 조도 센서 작동에 필요.\n"
            "3. 설치 장소 주위 온도 5℃~43℃."
        ),
    },
    {
        "question": "S835S31 냉장고 설치할 때 주의사항 알려줘",
        "original": (
            "S835S31 냉장고 설치 시 주의해야 할 사항은 다음과 같습니다:\n\n"
            "1. **안정적인 바닥**: 냉장고를 설치할 때 바닥이 튼튼하고 수평인 곳에 설치해야 합니다. "
            "불안정한 곳에 두면 냉장고가 넘어질 수 있어 사람에게 위험할 수 있습니다.\n\n"
            "2. **공간 확보**: 냉장고의 뒷면이 벽에 닿지 않도록 하고, 주위에 적당한 간격을 두어야 합니다. "
            "또한 냉장고 위에 무거운 물건을 올리지 않도록 주의하세요.\n\n"
            "3. **냉각구역 방해 금지**: 냉기가 나오는 부분이나 밑부분, 뒷면 방열 그릴(배기구) 등에 손이나 "
            "금속재의 막대를 넣지 않도록 주의하세요. 이는 감전이나 상해의 우려가 있습니다.\n\n"
            "4. **선반 및 바스켓 주의**: 냉장고 선반을 뒤집어 넣지 않도록 하며, 도어 바스켓에는 물통이나 "
            "반찬 용기를 과도하게 보관하지 않는 것이 좋습니다.\n\n"
            "이러한 사항들을 준수하여 안전한 설치를 하시기 바랍니다. 추가 질문이 있으시면 언제든지 물어보세요!\n\n"
            "(참고: 10페이지, 37페이지, 46페이지)"
        ),
        "terse": (
            "1. 바닥이 튼튼하고 수평인 곳에 설치 - 불안정하면 넘어질 위험.\n"
            "2. 뒷면이 벽에 닿지 않게, 주위 간격 확보, 위에 무거운 물건 금지.\n"
            "3. 냉기 배출구·방열 그릴에 손이나 금속 막대 넣지 않기 - 감전·상해 위험.\n"
            "4. 선반을 뒤집어 넣지 않기, 도어 바스켓에 물통·반찬통 과적재 금지."
        ),
    },
    {
        "question": "S825AW35 냉장고 문 열림 알람 기능 알려줘",
        "original": (
            "S825AW35 냉장고에는 문 열림 경보 기능이 있습니다. 냉장고 문이나 홈바 문이 열려 있거나 "
            "완전히 닫히지 않은 채로 1분이 지나면 30초 간격으로 3회의 경보음이 울립니다. 만약 문을 닫은 "
            "후에도 경보음이 계속 울린다면, 이때는 서비스센터에 연락하시는 것이 좋습니다. 추가적인 점검이 "
            "필요할 수 있습니다.\n\n(참고: 20페이지, 36페이지, 47페이지)"
        ),
        "terse": (
            "문/홈바 문이 1분 이상 열려 있으면 30초 간격으로 3회 경보음이 울립니다. "
            "문을 닫아도 계속 울리면 서비스센터 점검이 필요합니다."
        ),
    },
    {
        "question": "SC3GBE50 스타일러 스마트폰 앱으로 제어하는 방법 알려줘",
        "original": (
            "SC3GBE50 스타일러를 스마트폰 앱으로 제어하는 방법은 다음과 같습니다:\n\n"
            "1. **전원 켜기**: 제품의 전원 버튼을 눌러 전원을 켭니다.\n"
            "2. **LG ThinQ 앱 실행**: 스마트폰에서 LG ThinQ 앱을 실행합니다.\n"
            "3. **제품 선택**: 앱 화면에서 '스타일러'를 선택합니다.\n"
            "4. **코스 선택**: 원하는 코스를 선택한 후 '다운로드' 버튼을 클릭합니다.\n"
            "5. **옷 넣기**: 스타일러에 스타일링할 옷을 넣습니다.\n"
            "6. **코스 시작하기**: 제품에서 '다운로드' 버튼을 눌러 선택한 코스를 확인한 후, "
            "'시작/일시정지' 버튼을 눌러 작동을 시작합니다.\n\n"
            "이 과정에서 앱을 통해 원격으로 제어할 수 있으며, 다운로드한 코스를 활용하여 다양한 스타일링 "
            "옵션을 사용할 수 있습니다. 만약 연결에 문제가 발생하면, 무선 공유기나 비밀번호를 확인하고 "
            "등록된 제품을 다시 등록해 주시기 바랍니다.\n\n(참고: 11페이지, 16페이지)"
        ),
        "terse": (
            "1. 전원 버튼으로 전원 켜기\n"
            "2. LG ThinQ 앱 실행\n"
            "3. 앱에서 '스타일러' 선택\n"
            "4. 원하는 코스 선택 후 '다운로드'\n"
            "5. 옷 넣기\n"
            "6. 제품에서 '다운로드' 확인 후 '시작/일시정지'로 작동 시작"
        ),
    },
    {
        "question": "AR06A9170HNQ 에어컨 스마트폰 앱으로 제어하는 방법 알려줘",
        "original": (
            "AR06A9170HNQ 에어컨을 스마트폰 앱으로 제어하는 방법은 다음과 같습니다:\n\n"
            "1. **SmartThings 앱 다운로드**: 먼저, 스마트폰에 SmartThings 앱을 다운로드하고 설치합니다.\n\n"
            "2. **Wi-Fi 연결 설정**: 에어컨이 Wi-Fi에 연결되어 있어야 AI(인공지능)쾌적 모드 등의 기능을 "
            "사용 가능합니다. 에어컨의 Wi-Fi 설정을 확인하고 연결합니다.\n\n"
            "3. **에어컨 추가하기**: SmartThings 앱을 열고, '기기 추가' 옵션을 선택하여 에어컨을 찾아 "
            "추가합니다. 이 과정에서 앱의 지시에 따라 진행하시면 됩니다.\n\n"
            "4. **모드 선택**: 앱에서 에어컨을 선택하면 AI(인공지능) 쾌적, 냉방, 제습, 청정 모드 중 원하는 "
            "모드를 선택할 수 있습니다.\n\n"
            "5. **온도 및 풍량 조절**: 각 모드에 따라 희망 온도와 풍량을 조절할 수 있습니다. 예를 들어, "
            "냉방 모드에서는 온도를 16~30℃ 범위 내에서 조절할 수 있습니다.\n\n"
            "6. **절전 기능 사용**: 절전 기능을 선택하면 소모 전력을 줄이며, 냉방 운전이 24℃로 조정될 수 있습니다.\n\n"
            "이렇게 해서 스마트폰을 통해 AR06A9170HNQ 에어컨을 편리하게 제어하실 수 있습니다. 추가적인 도움이 "
            "필요하시거나 문제가 발생하면 언제든지 문의해주세요!\n\n(참고: 7페이지, 11페이지, 18페이지)"
        ),
        "terse": (
            "1. SmartThings 앱 다운로드 및 설치\n"
            "2. 에어컨 Wi-Fi 연결 설정\n"
            "3. 앱에서 '기기 추가'로 에어컨 등록\n"
            "4. AI쾌적/냉방/제습/청정 모드 중 선택\n"
            "5. 온도(냉방 16~30℃)·풍량 조절\n"
            "6. 절전 기능 선택 시 냉방 24℃로 자동 조정"
        ),
    },
    {
        "question": "FQ25GN9BKN 에어컨 예약 기능 사용법 알려줘",
        "original": (
            "FQ25GN9BKN 에어컨의 예약 기능 사용법은 다음과 같습니다.\n\n"
            "**켜짐 예약하기:**\n1. 리모컨의 **설정** 버튼을 누릅니다.\n"
            "2. 리모컨의 **L M** 버튼을 눌러 **켜짐**을 선택한 후, 리모컨의 **확인** 버튼을 누릅니다.\n"
            "3. **L M** 버튼을 눌러 원하는 시간을 설정한 다음, **확인** 버튼을 누릅니다.\n"
            "   - 1시간부터 24시간까지 1시간 단위로 설정할 수 있습니다.\n\n"
            "**꺼짐 예약하기:**\n1. 리모컨의 **설정** 버튼을 누릅니다.\n"
            "2. 리모컨의 **L M** 버튼을 눌러 **꺼짐**을 선택한 후, 리모컨의 **확인** 버튼을 누릅니다.\n"
            "3. **L M** 버튼을 눌러 원하는 시간을 설정한 다음, **확인** 버튼을 누릅니다.\n"
            "   - 1시간부터 24시간까지 1시간 단위로 설정할 수 있습니다.\n\n"
            "**예약 취소하기:**\n1. 리모컨의 **설정** 버튼을 누릅니다.\n"
            "2. 리모컨의 **L M** 버튼을 눌러 **예약취소**를 선택한 후, 리모컨의 **확인** 버튼을 누릅니다.\n"
            "   - 설정된 모든 예약이 해제됩니다.\n\n"
            "예약된 기능은 제품이 작동 중일 때도 설정할 수 있으며, 예약 시간을 0으로 설정하면 예약이 "
            "해제됩니다. 추가적으로 궁금한 점이 있으면 언제든지 질문해 주세요!\n\n(참고: 37페이지, 57페이지)"
        ),
        "terse": (
            "켜짐 예약: 설정 → L M으로 '켜짐' 선택 → 확인 → L M으로 시간(1~24시간, 1시간 단위) → 확인.\n"
            "꺼짐 예약: 설정 → L M으로 '꺼짐' 선택 → 확인 → L M으로 시간(1~24시간) → 확인.\n"
            "예약 취소: 설정 → L M으로 '예약취소' → 확인, 모든 예약 해제."
        ),
    },
    {
        "question": "S825AW35 냉장고 급속 냉동 기능이 뭐야",
        "original": (
            "S825AW35 냉장고의 급속 냉동 기능은 식품을 최적의 조건으로 보관하기 위해 빠른 속도로 냉동할 수 "
            "있도록 도와주는 기능입니다. 이 기능은 두 가지 선택이 가능합니다.\n\n"
            "1. **특급냉동**: 냉동실 전체를 빠르게 냉동시키는 기능입니다. 많은 양의 냉동 식품을 한꺼번에 "
            "투입할 때 사용하면 좋습니다. 이 기능은 켜진 상태에서 일정 시간이 지나면 자동으로 꺼집니다.\n\n"
            "2. **-35℃ 특냉실**: 냉동실의 특냉 코너만 빠르게 냉동시키는 기능입니다. 적은 양의 육류나 생선을 "
            "빠르게 냉동하여 신선하게 보관할 때 유용합니다.\n\n"
            "이 기능은 버튼을 누를 때마다 \"특급냉동\", \"-35℃특냉실\", \"꺼짐\"이 반복되며, 필요에 따라 "
            "선택하여 사용할 수 있습니다.\n\n(참고: 20페이지, 35페이지, 51페이지)"
        ),
        "terse": (
            "식품을 빠르게 냉동하는 기능, 두 가지 옵션.\n"
            "1. 특급냉동: 냉동실 전체를 빠르게 냉동, 많은 양 투입 시 사용, 일정 시간 후 자동 해제.\n"
            "2. -35℃ 특냉실: 특냉 코너만 빠르게 냉동, 소량의 육류·생선용.\n"
            "버튼을 누를 때마다 특급냉동 → -35℃특냉실 → 꺼짐 순으로 전환됩니다."
        ),
    },
]


async def main() -> None:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import AnswerRelevancy

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI()
    evaluator_llm = llm_factory("gpt-4o-mini", client=client, temperature=0, max_tokens=2000)
    evaluator_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1)

    # group3: 우리 RAG 파이프라인(검색/SYSTEM_PROMPT) 전혀 안 씀 - GPT한테 일반
    # 지식으로 친절한 챗봇 스타일로 답하게만 시킨 순수 답변. "우리 시스템이라서
    # 불리한 것 아니냐"는 반박을 막기 위한 대조군.
    GENERIC_SYSTEM_PROMPT = (
        "너는 친절한 AI 어시스턴트야. 사용자 질문에 자연스럽고 도움이 되는 방식으로 답해줘. "
        "일반적으로 알려진 지식을 바탕으로 답하면 되고, 특정 제품 매뉴얼을 인용할 필요는 없어."
    )

    orig_scores, terse_scores, gpt_scores = [], [], []
    for p in PAIRS:
        gpt_raw = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
                {"role": "user", "content": p["question"]},
            ],
            temperature=0.7,
        )
        gpt_answer = gpt_raw.choices[0].message.content

        r_orig = await answer_relevancy.ascore(user_input=p["question"], response=p["original"])
        r_terse = await answer_relevancy.ascore(user_input=p["question"], response=p["terse"])
        r_gpt = await answer_relevancy.ascore(user_input=p["question"], response=gpt_answer)
        orig_scores.append(r_orig.value)
        terse_scores.append(r_terse.value)
        gpt_scores.append(r_gpt.value)
        print(f"{p['question']}")
        print(f"  원본(RAG)      AR={r_orig.value:.4f}")
        print(f"  축약본         AR={r_terse.value:.4f}  (delta vs 원본={r_terse.value - r_orig.value:+.4f})")
        print(f"  GPT순수답변    AR={r_gpt.value:.4f}")
        print(f"  GPT답변 미리보기: {gpt_answer[:150]}")
        print()

    def avg(x): return sum(x) / len(x)
    print("=== 평균 ===")
    print(f"원본(RAG): {avg(orig_scores):.4f}")
    print(f"축약본: {avg(terse_scores):.4f}  (delta={avg(terse_scores) - avg(orig_scores):+.4f})")
    print(f"GPT순수답변: {avg(gpt_scores):.4f}  (delta vs 원본={avg(gpt_scores) - avg(orig_scores):+.4f})")


if __name__ == "__main__":
    asyncio.run(main())
