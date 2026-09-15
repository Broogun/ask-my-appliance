"""RAG 실험 공통 평가 하니스.

팀원 실험 파일(experiments/{이름}_expNN.py)에서 이 모듈을 import해서 쓴다.
MODEL_QUESTIONS/SYSTEM_PROMPT(통일 기준)와 채점/보고서 저장 로직(RULES.md 기준
"수정 불필요" 부분)을 여기 한 곳에 모아뒀다 - 예전에는 실험 파일마다 이 코드를
그대로 복붙해서 4명이 똑같은 코드를 반복해서 들고 있었는데, 정작 이 부분은
"비교 대상"이 아니라 다 같아야 하는 채점 기준이라 복붙할 이유가 없었다. 이렇게
공용 모듈로 빼면 실험 파일에는 본인이 실제로 다르게 구현하는 my_answer()만
남는다.

**2026-09 변경 (관리자 승인)**: 기존 `COMMON_QUESTIONS`(22개, 브랜드/카테고리
단위로 골고루 섞은 고정 질문)를 `MODEL_QUESTIONS`(제품 모델 60개 × 5문항 = 300개)
로 교체했다. 계기: 22개짜리 질문만으로 팀원 실험을 돌려보니 60개 제품 모델 중
실제로 검색에 한 번이라도 걸린 건 39개뿐이었고 21개는 한 번도 검증되지 않았다
(실제로 확인됨) - 청킹 버그가 있어도 하필 그 모델이 안 걸리면 못 잡아낸다는
뜻이다. 300개는 각 브랜드 x 카테고리 조합의 실제 PDF 파일명(모델명)을 질문에
직접 포함시켜서, 60개 전부가 최소 5번씩은 검증되게 만든다.

질문은 특정 팀원의 청킹/DB 내부 구조(섹션 제목 등)에 의존하지 않고, "카테고리별
공통으로 있을 법한 주제 + 모델명"으로만 구성했다 - 4명이 서로 다른 방식으로
구현해도 공정하게 적용되도록 하기 위해서다. 앞으로 라운드를 늘릴 때도
(다른 유형의 질문 추가) 여기 `MODEL_QUESTIONS`에 계속 이어붙인다.

사용법 (experiments/{이름}_expNN.py 안에서):
    from pipeline.evaluate import MODEL_QUESTIONS, SYSTEM_PROMPT, run_and_save

    def my_answer(query: str) -> dict:
        ...

    if __name__ == "__main__":
        run_and_save(EXPERIMENT_NAME, NOTES, STRATEGY, my_answer)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

# ── 제품 모델 목록 (수정 금지 - data/lg + data/samsung 실제 PDF 파일명 기준) ──
# (브랜드, 카테고리, 모델명) - 모델명은 파일명에서 카테고리 접두어(AC_/RF_/DR_ 등)를
# 뺀 나머지 그대로다. 새 PDF가 추가되면 여기도 같이 업데이트한다(팀장 승인 후).
#
# **2026-09 카테고리 재정리(관리자 승인)**: 기존엔 폴더 구조 그대로 에어컨/냉장고/
# 세탁기 3개뿐이었는데, 실제 PDF 표지를 하나하나 열어보니 "세탁기" 폴더 안에
# 건조기·세탁건조기·스타일러(의류관리기)·미니워시가, "냉장고" 폴더 안에
# 김치냉장고·냉동고가 섞여 있었다 - 심지어 삼성 쪽엔 슈드레서(신발관리기)까지
# 있어서 "탈수/배수" 같은 세탁기 질문을 던지면 아예 말이 안 되는 제품도 있었다.
# 그래서 각 PDF 1페이지에 실제로 적힌 제품명을 기준으로 카테고리를 다시 나눴다
# (추측이 아니라 문서 원문 확인). 냉동고는 냉장고와 질문이 거의 겹쳐서 냉장고에
# 포함, 미니워시는 세탁기에 포함했고 나머지는 별도 카테고리로 분리했다.
MODEL_LIST: list[tuple[str, str, str]] = [
    # LG 에어컨 (스탠드형/벽걸이형)
    ("LG", "에어컨", "FQ16FV6EDN"), ("LG", "에어컨", "FQ17FN5BDN"),
    ("LG", "에어컨", "FQ18GC1EHN"), ("LG", "에어컨", "FQ18GN7BKN"),
    ("LG", "에어컨", "FQ18GU1BHN"), ("LG", "에어컨", "FQ25GN9BKN"),
    ("LG", "에어컨", "SQ06EJ1WAJ"), ("LG", "에어컨", "SQ07GA3WBN"),
    ("LG", "에어컨", "SQ07GJ1WEN"), ("LG", "에어컨", "SQ09GK1WEN"),
    # LG 냉장고 (일반/상냉장하냉동/양문형 + 냉동고 1개 포함)
    ("LG", "냉장고", "B242S32"), ("LG", "냉장고", "CA-H17DC"),
    ("LG", "냉장고", "GC-B40BSCQ"), ("LG", "냉장고", "GC-B414HG7M"),  # 냉동고
    ("LG", "냉장고", "J825MEE042"), ("LG", "냉장고", "M402ND"),
    ("LG", "냉장고", "S825AW35"), ("LG", "냉장고", "S835S31"),
    ("LG", "냉장고", "W826AAA492"),
    # LG 김치냉장고
    ("LG", "김치냉장고", "K135LW123"),
    # LG 세탁기 (드럼/전자동 + 미니워시 1개 포함)
    ("LG", "세탁기", "F21VDSK"), ("LG", "세탁기", "FC4KC"),  # 미니워시
    ("LG", "세탁기", "T17J4EFNTX"), ("LG", "세탁기", "TR16MV6"),
    # LG 건조기
    ("LG", "건조기", "RH10VTA"),
    # LG 세탁건조기 (일체형)
    ("LG", "세탁건조기", "RH21E5ATKWP"), ("LG", "세탁건조기", "FA25AJPB"),
    ("LG", "세탁건조기", "FC2521KX6C"), ("LG", "세탁건조기", "FH23KN"),
    # LG 스타일러 (의류관리기)
    ("LG", "스타일러", "SC3GBE50"),
    # 삼성 에어컨 (스탠드형/벽걸이형/시스템)
    ("삼성", "에어컨", "AF17B6474WSN"), ("삼성", "에어컨", "AF17B7538TZN"),
    ("삼성", "에어컨", "AF18RX975CAN"), ("삼성", "에어컨", "AF19TX978MZ3N"),
    ("삼성", "에어컨", "AM145JNPDBH1"), ("삼성", "에어컨", "AR-EH05"),
    ("삼성", "에어컨", "AR06A9170HNQ"), ("삼성", "에어컨", "AR06R1130HZN"),
    ("삼성", "에어컨", "AR09T9170HCN"), ("삼성", "에어컨", "AR11B9150HZT"),
    # 삼성 냉장고 (일반형/SBS)
    ("삼성", "냉장고", "RF85B90P1AP"), ("삼성", "냉장고", "RF85DB90B1AP"),
    ("삼성", "냉장고", "RH82M9152SL"), ("삼성", "냉장고", "RS63R557EB4"),
    ("삼성", "냉장고", "RS84B5041WW"), ("삼성", "냉장고", "RT42CG6024S9"),
    ("삼성", "냉장고", "RT53DG7A1CWW"), ("삼성", "냉장고", "SRS705IC"),
    # 삼성 김치냉장고 (김치플러스 포함)
    ("삼성", "김치냉장고", "RP20C3111S9"), ("삼성", "김치냉장고", "RQ49C9002S9"),
    # 삼성 세탁기 (드럼/일반)
    ("삼성", "세탁기", "WA19CG6745BV"), ("삼성", "세탁기", "WA80F19E8W"),
    ("삼성", "세탁기", "WF17M9100KG"), ("삼성", "세탁기", "WF21T6500KW"),
    # 삼성 건조기 (전기건조기)
    ("삼성", "건조기", "DV17T8520BV"), ("삼성", "건조기", "DV90TA040TE"),
    # 삼성 에어드레서 (의류관리기 - 스타일러와 같은 카테고리로 묶음)
    ("삼성", "스타일러", "DF60R8300WG"), ("삼성", "스타일러", "DF90H24R5C"),
    # 삼성 슈드레서 (신발관리기)
    ("삼성", "슈드레서", "DJ30T9500FE"), ("삼성", "슈드레서", "DJ40CB9600NE"),
]

# 카테고리별 공통 주제 (팀원 누구의 내부 구현에도 의존하지 않는, 실제 매뉴얼에
# 통상 있을 법한 주제 + 그 답변에 자연스럽게 나올 키워드). {model}에 모델명이 들어간다.
#
# **라운드 구조 및 질문 설계 기준**: 한 번에 다 만들지 않고, 회차(라운드)마다
# "다른 관점"의 질문을 5개씩 추가해서 점진적으로 커버리지와 케이스 다양성을
# 넓힌다. 라운드 사이에 주제가 겹치면 다양성을 넓힌다는 목적이 무너지므로,
# 각 라운드는 아래처럼 명확히 다른 축을 기준으로 설계했다:
#
# - **라운드1 - "가장 흔한 질문" 축**: 실제 A/S·고객센터 문의에서 가장 많이
#   나올 법한 기본 사용법·기본 고장 증상 (필터 청소, 이상한 소리, 기본 기능
#   안 됨, 에러 확인, 핵심 부품 관리). 카테고리를 막론하고 "이것부터 물어볼 것".
# - **라운드2 - "세부 기능·안전" 축**: 라운드1보다 한 단계 더 들어간 기능
#   (리모컨/무풍/제습, 정수필터/절전, 세제/통세척 등)과 안전 관련(냄새, 냉매 등)
#   질문. 매뉴얼 안에서도 조금 더 뒤쪽/세부 챕터에 있을 법한 내용.
# - **라운드3 - "스마트 기능·설치·에너지" 축**: 앞의 두 라운드와 성격이 다른
#   축 - 스마트폰 앱 연동, 설치 시 주의사항, 전력/에너지 소비 등 "제품을 쓰기
#   전/부가 기능" 관점. 매뉴얼의 도입부(설치)와 스마트 기능 챕터를 겨냥한다.
#
# 새 라운드를 추가할 때는 이렇게 "기존 라운드와 명확히 다른 축"을 먼저 정하고
# (예: 라운드4 - "복합/모호한 표현" 축, 라운드5 - "브랜드 특화 기능" 축 등),
# `_ROUNDS`에 새 항목을 추가하기만 하면 `MODEL_QUESTIONS`에 자동으로 누적된다
# (기존 라운드는 삭제/변경 금지 - 과거 실험 결과와의 비교 가능성이 깨진다).
_ROUND1_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "에어컨": [
        ("{model} 에어컨 필터 청소 방법 알려줘", "필터"),
        ("{model} 에어컨에서 이상한 소리가 나요", "소리"),
        ("{model} 에어컨이 냉방이 잘 안 돼요", "냉방"),
        ("{model} 에어컨 에러 메시지 확인하는 방법 알려줘", "에러"),
        ("{model} 에어컨 실외기 관리하는 방법 알려줘", "실외기"),
    ],
    "냉장고": [
        ("{model} 냉장고 온도 설정 방법 알려줘", "온도"),
        ("{model} 냉장고에 성에가 계속 생겨요", "성에"),
        ("{model} 냉장고에서 소음이 나는데 왜 그래요", "소음"),
        ("{model} 냉장고 문이 잘 안 닫혀요", "문"),
        ("{model} 냉장고 청소하는 방법 알려줘", "청소"),
    ],
    "김치냉장고": [
        ("{model} 김치냉장고 온도 설정 방법 알려줘", "온도"),
        ("{model} 김치냉장고 발효 기능이 뭐야", "발효"),
        ("{model} 김치냉장고에서 냄새가 나요", "냄새"),
        ("{model} 김치냉장고 문이 잘 안 닫혀요", "문"),
        ("{model} 김치냉장고 청소하는 방법 알려줘", "청소"),
    ],
    "세탁기": [
        ("{model} 세탁기 필터 청소 방법 알려줘", "필터"),
        ("{model} 세탁기 탈수할 때 소음이 심해요", "탈수"),
        ("{model} 세탁기 문이 안 열려요", "문"),
        ("{model} 세탁기 배수가 안 돼요", "배수"),
        ("{model} 세탁 코스 선택하는 방법 알려줘", "코스"),
    ],
    "건조기": [
        ("{model} 건조기 건조 코스 선택하는 방법 알려줘", "코스"),
        ("{model} 건조기 먼지 필터 청소 방법 알려줘", "필터"),
        ("{model} 건조기 건조 시간이 너무 오래 걸려요", "시간"),
        ("{model} 건조기에서 소음이 나는데 왜 그래요", "소음"),
        ("{model} 건조기 에러 코드는 어디서 확인해", "에러"),
    ],
    "세탁건조기": [
        ("{model} 세탁건조기 세탁 코스 선택하는 방법 알려줘", "코스"),
        ("{model} 세탁건조기 건조 기능은 어떻게 써", "건조"),
        ("{model} 세탁건조기 필터 청소 방법 알려줘", "필터"),
        ("{model} 세탁건조기 문이 안 열려요", "문"),
        ("{model} 세탁건조기 에러 코드는 어디서 확인해", "에러"),
    ],
    "스타일러": [
        ("{model} 스타일러 냄새 제거 기능이 뭐야", "냄새"),
        ("{model} 스타일러 트리트먼트 기능 알려줘", "트리트먼트"),
        ("{model} 스타일러 문이 안 열려요", "문"),
        ("{model} 스타일러 청소하는 방법 알려줘", "청소"),
        ("{model} 스타일러 에러 코드는 어디서 확인해", "에러"),
    ],
    "슈드레서": [
        ("{model} 슈드레서 신발 살균 기능이 뭐야", "살균"),
        ("{model} 슈드레서 냄새 제거는 어떻게 해", "냄새"),
        ("{model} 슈드레서 건조 기능 알려줘", "건조"),
        ("{model} 슈드레서 문이 안 열려요", "문"),
        ("{model} 슈드레서 청소하는 방법 알려줘", "청소"),
    ],
}

# 라운드2 - "세부 기능·안전" 축 (위쪽 라운드 구조 설명 참고).
_ROUND2_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "에어컨": [
        ("{model} 에어컨 리모컨 사용법 알려줘", "리모컨"),
        ("{model} 에어컨 무풍 기능이 뭐야", "무풍"),
        ("{model} 에어컨 제습 모드는 어떻게 써", "제습"),
        ("{model} 에어컨에서 냄새가 나는데 계속 써도 될까", "냄새"),
        ("{model} 에어컨 실내기 자동 건조 기능 알려줘", "건조"),
    ],
    "냉장고": [
        ("{model} 냉장고 냉동실 온도는 몇 도가 적당해", "냉동"),
        ("{model} 냉장고 야채실 사용법 알려줘", "야채"),
        ("{model} 냉장고 정수 필터 교체 방법 알려줘", "정수"),
        ("{model} 냉장고 절전하는 방법 알려줘", "절전"),
        ("{model} 냉장고 문이 오래 열려 있으면 알림이 오나", "알림"),
    ],
    "김치냉장고": [
        ("{model} 김치냉장고 보관 용기는 어떻게 써", "용기"),
        ("{model} 김치냉장고 절전 방법 알려줘", "절전"),
        ("{model} 김치냉장고 급속 냉동 기능이 뭐야", "급속"),
        ("{model} 김치냉장고 알림 기능 알려줘", "알림"),
        ("{model} 김치냉장고 설치 시 주의사항 알려줘", "설치"),
    ],
    "세탁기": [
        ("{model} 세탁기 세제는 어디에 넣어", "세제"),
        ("{model} 세탁기 예약 세탁 어떻게 해", "예약"),
        ("{model} 세탁기 통 세척 방법 알려줘", "통세척"),
        ("{model} 세탁기 울 코스는 어떻게 써", "코스"),
        ("{model} 세탁기 에러 코드는 어디서 확인해", "에러"),
    ],
    "건조기": [
        ("{model} 건조기 응축수 비우는 방법 알려줘", "응축수"),
        ("{model} 건조기에 세탁물을 얼마나 넣어도 돼", "용량"),
        ("{model} 건조기 사용하면 옷에 구김이 덜 가나", "구김"),
        ("{model} 건조기에서 냄새가 나요", "냄새"),
        ("{model} 건조기 정기 관리 방법 알려줘", "관리"),
    ],
    "세탁건조기": [
        ("{model} 세탁건조기 탈수할 때 소음이 심해요", "탈수"),
        ("{model} 세탁건조기 예약 세탁 어떻게 해", "예약"),
        ("{model} 세탁건조기 응축수 비우는 방법 알려줘", "응축수"),
        ("{model} 세탁건조기 세제는 어디에 넣어", "세제"),
        ("{model} 세탁건조기에서 소음이 나는데 왜 그래요", "소음"),
    ],
    "스타일러": [
        ("{model} 스타일러 정전기 방지 기능이 뭐야", "정전기"),
        ("{model} 스타일러 스팀 기능 알려줘", "스팀"),
        ("{model} 스타일러 한 번 사용하는데 시간이 얼마나 걸려", "시간"),
        ("{model} 스타일러에서 소음이 나는데 왜 그래요", "소음"),
        ("{model} 스타일러 필터 교체 방법 알려줘", "필터"),
    ],
    "슈드레서": [
        ("{model} 슈드레서 신발 트레이는 어떻게 써", "트레이"),
        ("{model} 슈드레서 한 번 사용하는데 시간이 얼마나 걸려", "시간"),
        ("{model} 슈드레서 에러 코드는 어디서 확인해", "에러"),
        ("{model} 슈드레서 습기 제거 기능 알려줘", "습기"),
        ("{model} 슈드레서 정기 관리 방법 알려줘", "관리"),
    ],
}

# 라운드3: 설치/스마트 기능/전력 관련 - round1(기본 단일 주제)·round2(세부 기능·안전)
# 와 겹치지 않는 세 번째 관점(스마트 앱 연동, 설치 시 주의사항, 전력/에너지 소비).
_ROUND3_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "에어컨": [
        ("{model} 에어컨 예약 기능 사용법 알려줘", "예약"),
        ("{model} 에어컨 취침 모드 어떻게 설정해", "취침"),
        ("{model} 에어컨 스마트폰 앱으로 제어하는 방법 알려줘", "앱"),
        ("{model} 에어컨 전력 소비량이 궁금해", "전력"),
        ("{model} 에어컨 설치할 때 주의사항 알려줘", "설치"),
    ],
    "냉장고": [
        ("{model} 냉장고 급속 냉동 기능이 뭐야", "급속"),
        ("{model} 냉장고 문 열림 알람 기능 알려줘", "알람"),
        ("{model} 냉장고 스마트폰 앱으로 확인하는 방법 알려줘", "앱"),
        ("{model} 냉장고 에너지 소비 효율 등급이 어떻게 돼", "등급"),
        ("{model} 냉장고 설치할 때 주의사항 알려줘", "설치"),
    ],
    "김치냉장고": [
        ("{model} 김치냉장고 김치 발효 코스 자세히 알려줘", "코스"),
        ("{model} 김치냉장고 스마트폰 앱 연동 방법 알려줘", "앱"),
        ("{model} 김치냉장고 정기 점검은 얼마나 자주 해야 해", "점검"),
        ("{model} 김치냉장고 전력 소비량이 궁금해", "전력"),
        ("{model} 김치냉장고 설치할 때 주의사항 알려줘", "설치"),
    ],
    "세탁기": [
        ("{model} 세탁기 스마트폰 앱으로 제어하는 방법 알려줘", "앱"),
        ("{model} 세탁기 물 온도 설정 방법 알려줘", "온도"),
        ("{model} 세탁기 소음을 줄이는 방법 알려줘", "소음"),
        ("{model} 세탁기 전기/수도 사용량이 궁금해", "사용량"),
        ("{model} 세탁기 설치할 때 수평 맞추는 방법 알려줘", "수평"),
    ],
    "건조기": [
        ("{model} 건조기 스마트폰 앱으로 제어하는 방법 알려줘", "앱"),
        ("{model} 건조기 자동 건조 센서가 뭐야", "센서"),
        ("{model} 건조기 전력 소비량이 궁금해", "전력"),
        ("{model} 건조기 설치할 때 환기구 주의사항 알려줘", "환기"),
        ("{model} 건조기 정기 점검은 얼마나 자주 해야 해", "점검"),
    ],
    "세탁건조기": [
        ("{model} 세탁건조기 스마트폰 앱으로 제어하는 방법 알려줘", "앱"),
        ("{model} 세탁건조기 물 온도 설정 방법 알려줘", "온도"),
        ("{model} 세탁건조기 전기/수도 사용량이 궁금해", "사용량"),
        ("{model} 세탁건조기 설치할 때 주의사항 알려줘", "설치"),
        ("{model} 세탁건조기 정기 점검은 얼마나 자주 해야 해", "점검"),
    ],
    "스타일러": [
        ("{model} 스타일러 스마트폰 앱으로 제어하는 방법 알려줘", "앱"),
        ("{model} 스타일러 항균 기능이 뭐야", "항균"),
        ("{model} 스타일러 전력 소비량이 궁금해", "전력"),
        ("{model} 스타일러 설치할 때 주의사항 알려줘", "설치"),
        ("{model} 스타일러 정기 점검은 얼마나 자주 해야 해", "점검"),
    ],
    "슈드레서": [
        ("{model} 슈드레서 스마트폰 앱으로 제어하는 방법 알려줘", "앱"),
        ("{model} 슈드레서 자외선 살균 기능 알려줘", "자외선"),
        ("{model} 슈드레서 전력 소비량이 궁금해", "전력"),
        ("{model} 슈드레서 설치할 때 주의사항 알려줘", "설치"),
        ("{model} 슈드레서 정기 점검은 얼마나 자주 해야 해", "점검"),
    ],
}

# 새 라운드를 추가할 땐 여기에 (라운드 이름, 카테고리별 템플릿)만 추가하면 된다.
_ROUNDS: list[tuple[str, dict[str, list[tuple[str, str]]]]] = [
    ("round1", _ROUND1_TEMPLATES),
    ("round2", _ROUND2_TEMPLATES),
    ("round3", _ROUND3_TEMPLATES),
]


def _build_model_questions() -> list[tuple[str, str, str, str, str]]:
    """(question, keyword, model, description, round) 형태로 라운드 전체 누적 생성.

    현재 라운드 2개 x 60모델 x 5문항 = 600개."""
    questions = []
    for round_name, templates in _ROUNDS:
        for brand, category, model in MODEL_LIST:
            for template, keyword in templates[category]:
                question = template.format(model=model)
                description = f"{brand} {category} {model} [{round_name}]"
                questions.append((question, keyword, model, description, round_name))
    return questions


# ── 공통 테스트 질문 (수정 금지 - RULES.md 참고, 관리자 승인 후에만 확장) ──────
MODEL_QUESTIONS: list[tuple[str, str, str, str, str]] = _build_model_questions()

# ── 공통 시스템 프롬프트 (수정 금지) ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 검색된 문서 내용만 근거로 자연스럽고 친절하게 답변해. "
    "문서에 없는 내용을 지어내면 안 돼. "
    "안전에 관련될 수 있는 내용(감전, 화재, 가스 누출 등)이면 "
    "전원을 끄고 서비스센터에 문의하라고 안내해."
)

MyAnswerFn = Callable[[str], dict]


def run_eval(my_answer: MyAnswerFn) -> list[dict]:
    """MODEL_QUESTIONS 전체에 대해 my_answer(query)를 호출하고 채점한다.

    my_answer는 {"answer": str, "candidates": [{"id","text","distance"}, ...]}
    형식을 반환해야 한다 (RULES.md 반환 형식 참고).
    """
    results = []
    for question, keyword, model, description, round_name in MODEL_QUESTIONS:
        start = time.time()
        try:
            out = my_answer(question)
        except Exception as e:
            out = {"answer": f"[ERROR] {e}", "candidates": []}
        elapsed = time.time() - start

        answer = out.get("answer", "")
        candidates = out.get("candidates", [])
        all_text = answer + " ".join(c.get("text", "") for c in candidates)
        keyword_hit = keyword in all_text
        distances = [c.get("distance", 0.0) for c in candidates]
        avg_dist = sum(distances) / len(distances) if distances else 1.0

        mark = "O" if keyword_hit else "X"
        print(f"[{mark}] {description}: {question}")
        print(f"    elapsed={elapsed:.1f}s | avg_dist={avg_dist:.4f}")

        results.append({
            "question": question,
            "model": model,
            "round": round_name,
            "description": description,
            "keyword_hit": keyword_hit,
            "elapsed_sec": round(elapsed, 2),
            "avg_distance": round(avg_dist, 4),
            "answer": answer,
            "candidates": [
                {"id": c.get("id", ""), "distance": c.get("distance", 0.0), "text": c.get("text", "")[:200]}
                for c in candidates
            ],
        })
    return results


def build_report(experiment_name: str, notes: str, strategy: dict, results: list[dict]) -> dict:
    hit_rate = sum(r["keyword_hit"] for r in results) / len(results)
    avg_dist = sum(r["avg_distance"] for r in results) / len(results)
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)

    # 모델별 히트율 - 300개짜리 세트로 바꾼 이유(60개 모델 전수 커버리지)가
    # 실제로 지켜지는지 보고서에서 바로 보이게 한다.
    per_model_hits: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        per_model_hits[r["model"]].append(r["keyword_hit"])
    models_with_zero_hits = sorted(
        model for model, hits in per_model_hits.items() if not any(hits)
    )

    # 라운드별 히트율 - 라운드마다 다른 관점(기본 주제/세부 기능·안전 등)으로
    # 질문을 쌓아가는 구조라, 어느 라운드(어떤 유형의 질문)가 약한지 바로 보인다.
    per_round_hits: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        per_round_hits[r["round"]].append(r["keyword_hit"])
    round_summary = {
        round_name: round(sum(hits) / len(hits), 4) for round_name, hits in per_round_hits.items()
    }

    print(f"\n{'=' * 55}")
    print(f"  키워드 히트율  : {hit_rate:.0%}")
    print(f"  평균 거리      : {avg_dist:.4f}")
    print(f"  평균 응답 시간 : {avg_time:.1f}s")
    print(f"  전수 검증된 모델: {len(per_model_hits)}/{len(MODEL_LIST)}개")
    print(f"  5문항 전부 놓친 모델: {len(models_with_zero_hits)}개 {models_with_zero_hits}")
    print("  라운드별 히트율:")
    for round_name, rate in round_summary.items():
        print(f"    {round_name}: {rate:.0%}")
    print(f"{'=' * 55}")

    return {
        "experiment_name": experiment_name,
        "notes": notes,
        "strategy": strategy,
        "summary": {
            "keyword_hit_rate": hit_rate,
            "avg_distance": avg_dist,
            "avg_elapsed_sec": avg_time,
            "models_tested": len(per_model_hits),
            "models_with_zero_hits": models_with_zero_hits,
            "round_hit_rates": round_summary,
        },
        "results": results,
    }


def save_report(report: dict, out_dir: str | Path = "experiments/results") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report['experiment_name']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")
    return out_path


def run_and_save(experiment_name: str, notes: str, strategy: dict, my_answer: MyAnswerFn) -> dict:
    """실험 파일의 `if __name__ == "__main__":` 블록에서 이 함수 하나만 호출하면
    질문 실행 -> 채점 -> 보고서 생성 -> experiments/results/{experiment_name}.json
    저장까지 전부 처리된다."""
    print(f"\n{'=' * 55}")
    print(f"  실험명  : {experiment_name}")
    print(f"  메모    : {notes}")
    print(f"  질문 수 : {len(MODEL_QUESTIONS)}개 (모델 {len(MODEL_LIST)}개 x {len(_ROUNDS)}라운드 x 5문항)")
    print(f"{'=' * 55}\n")
    results = run_eval(my_answer)
    report = build_report(experiment_name, notes, strategy, results)
    save_report(report)
    return report
