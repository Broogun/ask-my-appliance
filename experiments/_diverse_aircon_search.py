import sys, json
sys.path.insert(0, 'experiments')
import importlib.util
spec = importlib.util.spec_from_file_location('exp02', 'experiments/exp02_retrieval.py')
exp02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp02)
state = exp02._load_state()

QUESTIONS = {
    "FQ16FV6EDN": [
        "필터에 곰팡이가 핀 것 같은데 어떻게 없애야 해?",
        "리모컨 배터리는 어떤 걸 써야 하나요?",
        "설치할 때 벽에서 얼마나 띄워야 하나요?",
        "타이머 켜놓고 자도 되나요?",
        "실외기에서 물 떨어지는 소리가 계속 나요",
    ],
    "FQ17FN5BDN": [
        "제습 기능이랑 냉방 기능 뭐가 달라?",
        "리모컨이 반응을 안 해요",
        "겨울에 안 쓸 때 실외기 커버 씌워야 하나요?",
        "청소 알림이 계속 떠요, 어떻게 꺼요",
        "바람 방향 조절이 안 돼요",
    ],
    "FQ25GN9BKN": [
        "와이파이 연결이 자꾸 끊겨요",
        "실내기에서 삐- 소리가 나요",
        "인공지능 절전 기능이 뭔가요",
        "커버 벗기고 청소해도 되나요",
        "이사할 때 어떻게 옮겨야 하나요",
    ],
    "SQ06EJ1WAJ": [
        "전원 켰는데 아무 반응이 없어요",
        "공기청정 기능 따로 켤 수 있나요",
        "필터 교체 주기가 어떻게 되나요",
        "잠잘 때 좋은 모드가 있나요",
        "실외기 소음이 너무 커요",
    ],
    "SQ07GA3WBN": [
        "습도가 너무 높은데 뭘 눌러야 하나요",
        "리모컨 액정에 아무것도 안 떠요",
        "설치 후 첫 사용 시 주의할 점 있나요",
        "정기점검은 얼마 만에 받아야 하나요",
        "실내기에서 하얀 연기 같은 게 나와요",
    ],
    "SQ09GK1WEN": [
        "냉방 모드로 안 바뀌어요",
        "빨래 건조할 때도 쓸 수 있나요",
        "필터 청소 안 하면 어떻게 되나요",
        "예약 기능 어떻게 취소해요",
        "실외기 위에 뭘 올려놔도 되나요",
    ],
    "AR06A9170HNQ": [
        "무풍 기능 켰는데 그대로 시원한데 원래 그런가요",
        "리모컨 신호가 잘 안 잡혀요",
        "에어컨 냄새가 나요, 관리 어떻게 하나요",
        "설치 위치 바꿔도 되나요",
        "정기적으로 뭘 점검해야 하나요",
    ],
    "AR06R1130HZN": [
        "취침모드 켜면 온도가 자동으로 바뀌나요",
        "실외기에서 진동이 심해요",
        "필터 표시등이 계속 켜져 있어요",
        "빠른 냉방 어떻게 켜요",
        "설치 기사 없이 직접 설치해도 되나요",
    ],
    "AR09T9170HCN": [
        "제습 모드 켰는데 바람이 안 나와요",
        "리모컨 잃어버렸는데 어떻게 조작해요",
        "실내기 청소 주기가 어떻게 되나요",
        "에너지 절약 모드는 어떻게 켜요",
        "타이머 예약이 자꾸 풀려요",
    ],
    "AR11B9150HZT": [
        "전원 코드를 뽑아놨다가 다시 꽂으면 뭐 해줘야 하나요",
        "실외기 배수가 잘 안 되는 것 같아요",
        "와이파이로 제어하려면 뭐가 필요한가요",
        "바람이 너무 세게 나와요, 줄이는 법",
        "냄새 제거 기능이 따로 있나요",
    ],
    "AF17B6474WSN": [
        "실외기 팬이 안 돌아가는 것 같아요",
        "습도 조절 기능이 있나요",
        "필터 어디서 사나요",
        "제품 보증 기간이 얼마나 되나요",
        "이상한 진동이 느껴져요",
    ],
    "AF19TX978MZ3N": [
        "찬바람이 약하게 나와요",
        "리모컨 버튼이 눌리지가 않아요",
        "실내기에서 물이 새요",
        "정기적으로 청소해야 할 부분이 어디인가요",
        "여러 대를 하나의 리모컨으로 조작할 수 있나요",
    ],
}

out = []
for model, qs in QUESTIONS.items():
    for q in qs:
        cands = exp02._vector_search(q, state, top_k=60, where={"product_model": model})
        cands.sort(key=lambda c: c["distance"])
        top = cands[:8]
        out.append({
            "model": model, "question": q,
            "candidates": [
                {"id": c["id"], "distance": round(c["distance"], 4),
                 "section": c.get("metadata", {}).get("section_title"),
                 "text": c["text"][:450]}
                for c in top
            ],
        })
        print(f"done: {model} :: {q}", flush=True)

with open("experiments/results/_diverse_aircon_search_dump.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved", len(out), "questions")
