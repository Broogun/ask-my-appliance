"""
에러코드 RDB (SQLite).

에러코드는 "이 코드를 입력하면 이 조치법이 정답"이라는 명확한 key-value 조회라서
벡터 검색(의미 유사도)이 아니라 정확 조회가 맞다. 벡터 검색은 두 가지 문제가 있었다:
  1) LLM이 검색된 텍스트를 잘못 해석해서 사실을 지어낼 위험 (예: "CH04랑 FL은 다른 코드"라고 오답)
  2) 긴 청크 속에 묻힌 별칭 코드를 못 찾는 검색 누락 (예: "P6" 검색 실패)
RDB 정확 조회는 이 두 문제를 원천적으로 없앤다.

스키마:
  error_solutions  1건당 1행. 실제 조치법 본문.
  error_codes      코드 별칭마다 1행 (예: CH61, P4, P6, P7, P8, CH34가 전부 같은
                    solution_id를 가리킴). 하나의 코드가 여러 solution을 가리킬
                    수도 있다 (예: F4는 CH38에도, CH90/91에도 쓰임 -> 중의적).
"""
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402  (Windows 콘솔 UTF-8 출력 설정을 위해 import)

DB_PATH = "data/error_codes.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS error_solutions (
    solution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT NOT NULL,       -- 예: '에어컨'
    title       TEXT NOT NULL,
    content     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS error_codes (
    code        TEXT NOT NULL,       -- 예: 'CH04', 'FL', 'P6'
    solution_id INTEGER NOT NULL REFERENCES error_solutions(solution_id),
    UNIQUE(code, solution_id)
);
CREATE INDEX IF NOT EXISTS idx_error_codes_code ON error_codes(code);
"""


@contextmanager
def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def add_solution(category: str, title: str, content: str, codes: list[str]) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO error_solutions (category, title, content) VALUES (?, ?, ?)",
            (category, title, content),
        )
        solution_id = cur.lastrowid
        for code in codes:
            conn.execute(
                "INSERT OR IGNORE INTO error_codes (code, solution_id) VALUES (?, ?)",
                (code.upper(), solution_id),
            )
        return solution_id


def lookup_code(code: str) -> list[dict]:
    """코드 하나로 조회. 중의적인 코드(F4 등)는 여러 건이 반환될 수 있다."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT s.solution_id, s.category, s.title, s.content,
                      GROUP_CONCAT(DISTINCT c2.code) AS all_aliases
               FROM error_codes c
               JOIN error_solutions s ON s.solution_id = c.solution_id
               JOIN error_codes c2 ON c2.solution_id = s.solution_id
               WHERE c.code = ?
               GROUP BY s.solution_id""",
            (code.upper(),),
        ).fetchall()
        return [dict(r) for r in rows]


def all_codes() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT DISTINCT code FROM error_codes").fetchall()
        return [r["code"] for r in rows]


def find_codes_in_text(text: str) -> list[str]:
    """질문 텍스트 안에 등록된 에러코드가 언급됐는지 확인한다 (대소문자 무시,
    단어 경계 기준 -> 'CH04'가 'CH040' 같은 다른 문자열의 일부로 오매칭되지 않게 함)."""
    import re

    text_upper = text.upper()
    found = []
    for code in all_codes():
        pattern = r"(?<![A-Z0-9])" + re.escape(code) + r"(?![A-Z0-9])"
        if re.search(pattern, text_upper):
            found.append(code)
    return found


if __name__ == "__main__":
    init_db()
    print(f"[error_codes_db] 초기화 완료 -> {DB_PATH}")
