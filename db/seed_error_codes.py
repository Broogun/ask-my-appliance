"""data/lg_{aircon,fridge,washer}_errors.json 의 에러코드 항목들을 RDB(error_codes.db)에
시딩한다. `python -m db.seed_error_codes` 로 실행.

카테고리 간에 같은 코드 문자열이 다른 의미로 쓰이는 경우가 있다 (예: 'FF'는 냉장고에서는
냉동실 팬모터 이상, 세탁기에서는 동결 감지). 이는 아이컨의 F4(CH38/CH90-91 중의적)와 같은
패턴이라 그대로 둔다 - lookup_code()가 여러 건을 반환하고 _answer_from_rdb()가 각각 구분해서
설명하므로 문제 없다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.error_codes_db import add_solution, init_db

# 파일별 (해당 섹션의 실제 코드 별칭 전부, JSON 섹션 순서와 일치)
SOURCES = [
    (
        "data/lg_aircon_errors.json",
        [
            ["CH04", "FL"],
            ["CH05", "E0", "CH53"],
            ["CH07"],
            ["CH38", "F4"],
            ["CH54"],
            ["CH61", "P4", "P6", "P7", "P8", "CH34"],
            ["CH66"],
            ["CH10", "E6", "CH67", "EF"],
            ["CH90", "CH91", "F4"],
            ["CH93"],
            ["CH237", "CH238"],
        ],
    ),
    (
        "data/lg_fridge_errors.json",
        [
            ["DH", "FDH", "RDH"],
            ["FF", "RF"],
            ["CH", "CL"],
            ["H1", "H2"],
            ["CO"],
        ],
    ),
    (
        "data/lg_washer_errors.json",
        [
            ["IE"],
            ["LE", "LE1", "AE"],
            ["UE"],
            ["DE", "DE1", "DE2"],
            ["OE"],
            ["FE"],
            ["FF"],
            ["TE"],
            ["PE"],
            ["TCL"],
        ],
    ),
    (
        "data/samsung_aircon_errors.json",
        [
            ["C101", "E101", "C102", "E102", "C124", "E124"],
            ["C103", "E103"],
            ["C116", "E116"],
            ["C121", "E121", "C122", "E122", "C123", "E123", "C125", "E125", "C144", "E144", "C145", "E145"],
            ["C165", "E165"],
            ["C221", "E221", "C226", "E226"],
            ["C401", "E401", "C402", "E402", "C403", "E403"],
            ["C404", "E404", "C405", "E405", "C406", "E406"],
            ["C440", "E440"],
            ["C441", "E441"],
            ["C452", "E452"],
            ["C551", "E551"],
            ["C574", "E574", "C575", "E575"],
        ],
    ),
    (
        "data/samsung_fridge_errors.json",
        [
            ["E", "C"],
            ["HO", "NO"],
            ["12HWARN"],
            ["DOORALARM"],
        ],
    ),
    (
        "data/samsung_washer_errors.json",
        [
            ["1E", "IE", "1C", "IC"],
            ["LE", "LC"],
            ["DE", "DC"],
            ["UE", "UB", "U6"],
            ["OE", "OF", "OC"],
            ["FE", "FC"],
            ["5C", "5E", "SC", "SE", "E2"],
            ["3E", "3C"],
        ],
    ),
]


def seed():
    init_db()
    total = 0
    for path, code_aliases in SOURCES:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = data[0]["sections"]
        category = data[0]["category"]

        assert len(sections) == len(code_aliases), f"{path}: 섹션 수와 코드 별칭 목록 수가 안 맞음"

        for section, codes in zip(sections, code_aliases):
            add_solution(category=category, title=section["title"], content=section["content"], codes=codes)
        print(f"[seed] {path}: {category} {len(sections)}개 에러코드 항목을 RDB에 저장했습니다.")
        total += len(sections)

    print(f"[seed] 총 {total}개 에러코드 항목 저장 완료.")


if __name__ == "__main__":
    seed()
