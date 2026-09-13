"""자연어 질문 -> 벡터 검색 -> LLM 답변 생성 (가전제품 사용·문제해결 RAG).

CHAT_BACKEND=local  (기본값, 무료): transformers로 로컬 GPU/CPU에서 LLM 직접 구동
CHAT_BACKEND=ollama (무료): 별도 설치한 Ollama 서버 호출
CHAT_BACKEND=openai (유료): OpenAI Chat Completions API
"""
import argparse
import re

import config
from db.error_codes_db import find_codes_in_text, lookup_code
from pipeline.embed_store import embed_texts, get_collection, get_known_models

# 등록된 모델(정확 매칭)의 문서만 근거로 쓸 때: 그 모델 전용 정보라고 자신있게 답해도 됨
EXACT_MODEL_SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 검색 결과는 사용자가 말한 모델(제품)의 사용설명서에서 직접 찾은 내용이야. "
    "그 내용만 근거로 자연스럽고 친절하게 답변해. 문서에 없는 내용을 지어내면 안 돼. "
    "안전에 관련될 수 있는 내용(감전, 화재, 가스 누출 등)이면 임의로 판단하지 말고 "
    "전원을 끄고 제조사 서비스센터에 문의하라고 안내해."
)

# 모델이 특정되지 않았거나 등록 안 된 모델일 때: 비슷한 다른 모델 사례를 자연스럽게 답변
# (지난번 로컬 3B 테스트에서 이 경우 '정보가 없다'고 답한 게 오히려 별로였음 -> 자연스럽게
# 일반적인 가이드처럼 답하도록 바꿈. 단, 검색된 문서 내용 밖의 사실을 지어내는 건 여전히 금지.)
GENERAL_SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 검색 결과는 사용자가 말한 모델과 정확히 일치하지 않을 수 있는 유사 제품의 "
    "사용설명서 내용이야. 이걸 바탕으로 마치 일반적으로 잘 알고 있는 것처럼 자연스럽게 "
    "답변해. '이 정보는 없지만', '설명서에 없지만', '정확한 모델은 아니지만' 같은 "
    "양해를 구하는 표현은 쓰지 말고, 바로 유용한 답변부터 자연스럽게 시작해. "
    "다만 검색된 문서에 실제로 있는 내용만 근거로 삼고 사실을 지어내지는 마. "
    "안전에 관련될 수 있는 내용(감전, 화재, 가스 누출 등)이면 임의로 판단하지 말고 "
    "전원을 끄고 제조사 서비스센터에 문의하라고 안내해."
)

RDB_SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래 [공식 에러코드 정보]는 제조사 데이터베이스에서 정확히 조회한 내용이니, "
    "그 내용만 근거로 친절하게 설명해. 내용을 지어내거나 다른 내용과 헷갈리면 안 돼. "
    "같은 코드가 여러 항목에 해당될 수 있다면(예: F4처럼 원인이 여러 개인 코드) "
    "각 항목을 구분해서 전부 설명해. 안전 관련 내용(감전, 화재, 가스 누출 등)이면 "
    "임의로 판단하지 말고 전원을 끄고 서비스센터에 문의하라고 안내해."
)

COMBINED_SYSTEM_PROMPT = (
    "너는 가전제품 사용법과 문제 해결을 도와주는 어시스턴트야. 반드시 한국어로만 답해. "
    "아래에는 두 가지 출처의 정보가 있어. "
    "[공식 에러코드 정보]는 제조사 DB에서 정확히 조회한 내용이고, "
    "[사용설명서 검색 결과]는 관련 사용설명서에서 찾은 내용이야. "
    "사용자 질문의 모든 항목에 빠짐없이 답하되, 각 출처의 내용만 근거로 삼고 지어내지 마. "
    "같은 에러코드가 원인이 여러 개라면 전부 설명해. "
    "안전 관련 내용(감전, 화재, 가스 누출 등)이면 전원을 끄고 서비스센터에 문의하라고 안내해."
)

_local_model = None
_local_tokenizer = None


def find_model_in_text(text: str) -> str | None:
    """질문에 등록된 제품 모델명이 정확히(단어 경계 기준) 언급됐는지 확인한다.
    여러 개가 우연히 겹치면 가장 긴 것(더 구체적인 것)을 우선한다."""
    text_upper = text.upper()
    matched = [
        m for m in get_known_models()
        if re.search(r"(?<![A-Z0-9])" + re.escape(m) + r"(?![A-Z0-9])", text_upper)
    ]
    if not matched:
        return None
    return max(matched, key=len)


def _section_key(chunk_id: str) -> str:
    """청크 ID에서 섹션 키를 추출한다.
    ID 형식: {model}_manual_{section_idx}_{chunk_idx}
    마지막 _{chunk_idx} 를 제거한 부분이 섹션 키."""
    return chunk_id.rsplit("_", 1)[0]


def search(query: str, top_k: int = 3, model_filter: str | None = None) -> list[dict]:
    [q_embedding] = embed_texts([query])

    collection = get_collection()
    where = {"product_model": model_filter} if model_filter else None

    # top_k * 3 을 먼저 뽑고, 같은 섹션에서는 가장 유사한 청크 1개만 남긴다.
    # '에어컨 필터 청소' 같이 질문이 특정 섹션과 직접 매칭될 때
    # 동일 섹션의 오버랩 청크들이 top-k를 전부 차지하는 문제를 방지한다.
    fetch_k = max(top_k * 3, top_k + 5)
    result = collection.query(query_embeddings=[q_embedding], n_results=fetch_k, where=where)

    seen_sections: set[str] = set()
    candidates = []
    for i in range(len(result["ids"][0])):
        chunk_id = result["ids"][0][i]
        sec_key = _section_key(chunk_id)
        if sec_key in seen_sections:
            continue
        seen_sections.add(sec_key)
        candidates.append(
            {
                "id": chunk_id,
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            }
        )
        if len(candidates) == top_k:
            break

    return candidates


def _generate_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content


def _generate_local(system_prompt: str, user_message: str) -> str:
    """transformers로 로컬 GPU/CPU에서 LLM(기본: Qwen2.5-3B-Instruct)을 직접 구동한다.
    별도 서버(Ollama 등) 설치가 필요 없다. 모델은 최초 1회만 로드해서 재사용한다."""
    global _local_model, _local_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _local_model is None:
        print(f"[chat] 로컬 LLM 로딩 중: {config.LOCAL_CHAT_MODEL} (최초 1회는 다운로드 필요)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _local_tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_CHAT_MODEL)
        _local_model = AutoModelForCausalLM.from_pretrained(
            config.LOCAL_CHAT_MODEL,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt = _local_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _local_tokenizer(prompt, return_tensors="pt").to(_local_model.device)
    output_ids = _local_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=_local_tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_ollama(system_prompt: str, user_message: str) -> str:
    """Ollama(https://ollama.com)로 로컬에서 돌아가는 LLM을 호출한다.
    사전에 `ollama pull qwen2.5:7b` 로 모델을 받아두고 Ollama 앱이 실행 중이어야 한다.

    think=False로 고정한다: qwen3.5 같은 추론(thinking) 지원 모델은 기본으로 답변 전에
    안 보이는 "생각" 토큰을 대량 생성한다 (실측: 짧은 RAG 답변 하나에 1,756 think 토큰,
    43초). 이 프로젝트는 문서에서 찾은 내용을 그대로 정리해서 답하는 단순 RAG 챗봇이라
    긴 추론이 필요 없고, think=False로 끄면 같은 질문이 3초로 줄어든다 (약 17배)."""
    import requests

    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "think": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Ollama 서버에 연결할 수 없습니다. https://ollama.com 에서 설치 후 "
            f"`ollama pull {config.OLLAMA_MODEL}` 로 모델을 받고 Ollama 앱을 실행해두세요."
        ) from e

    return resp.json()["message"]["content"]


def _generate(system_prompt: str, user_message: str) -> str:
    if config.CHAT_BACKEND == "openai":
        return _generate_openai(system_prompt, user_message)
    if config.CHAT_BACKEND == "ollama":
        return _generate_ollama(system_prompt, user_message)
    return _generate_local(system_prompt, user_message)


def _lookup_rdb_solutions(matched_codes: list[str]) -> list[dict]:
    solutions = []
    seen_ids = set()
    for code in matched_codes:
        for sol in lookup_code(code):
            if sol["solution_id"] not in seen_ids:
                seen_ids.add(sol["solution_id"])
                solutions.append(sol)
    return solutions


def _answer_from_rdb(query: str, matched_codes: list[str]) -> dict:
    """에러코드만 있는 단순 질문: RDB만 사용."""
    solutions = _lookup_rdb_solutions(matched_codes)
    rdb_context = "\n\n".join(
        f"[{s['title']}] (관련 코드: {s['all_aliases']})\n{s['content']}" for s in solutions
    )
    user_message = f"[공식 에러코드 정보 (RDB 정확 조회)]\n{rdb_context}\n\n[사용자 질문]\n{query}"
    text = _generate(RDB_SYSTEM_PROMPT, user_message)
    candidates = [
        {
            "id": str(s["solution_id"]),
            "text": s["content"],
            "metadata": {"section_title": s["title"], "source": "RDB", "aliases": s["all_aliases"]},
            "distance": 0.0,
        }
        for s in solutions
    ]
    return {"answer": text, "candidates": candidates, "source": "rdb", "matched_codes": matched_codes}


def _answer_combined(
    query: str, matched_codes: list[str], solutions: list[dict], vector_candidates: list[dict]
) -> dict:
    """에러코드 + 일반 질문이 섞인 복합 질문: RDB + 벡터 컨텍스트를 함께 사용."""
    rdb_context = "\n\n".join(
        f"[{s['title']}] (관련 코드: {s['all_aliases']})\n{s['content']}" for s in solutions
    )
    vector_context = "\n".join(f"- {c['text']}" for c in vector_candidates)
    user_message = (
        f"[공식 에러코드 정보]\n{rdb_context}\n\n"
        f"[사용설명서 검색 결과]\n{vector_context}\n\n"
        f"[사용자 질문]\n{query}"
    )
    text = _generate(COMBINED_SYSTEM_PROMPT, user_message)

    rdb_cands = [
        {
            "id": str(s["solution_id"]),
            "text": s["content"],
            "metadata": {"section_title": s["title"], "source": "RDB", "aliases": s["all_aliases"]},
            "distance": 0.0,
        }
        for s in solutions
    ]
    return {
        "answer": text,
        "candidates": rdb_cands + vector_candidates,
        "source": "combined",
        "matched_codes": matched_codes,
    }


def _answer_from_vector(query: str, top_k: int, model_filter: str | None = None) -> dict:
    """질문에 등록된 모델명이 정확히 언급되면 그 모델 문서만으로 검색을 제한하고
    (핵심 키 매칭), 아니면 전체에서 검색해서 가장 비슷한 사례를 자연스럽게 답한다.

    model_filter가 명시적으로 주어지면(예: Streamlit UI에서 제품을 직접 선택한 경우)
    질문 텍스트에 모델명이 안 적혀 있어도 그 값을 우선한다."""
    matched_model = model_filter or find_model_in_text(query)
    candidates = search(query, top_k, model_filter=matched_model)

    # 특정 모델로 필터링했는데 그 모델 문서에 결과가 없으면(이론상 거의 없지만) 전체 검색으로 폴백
    if matched_model and not candidates:
        candidates = search(query, top_k)
        matched_model = None

    context = "\n".join(f"- {c['text']}" for c in candidates)
    user_message = f"[검색된 문서]\n{context}\n\n[사용자 질문]\n{query}"

    system_prompt = EXACT_MODEL_SYSTEM_PROMPT if matched_model else GENERAL_SYSTEM_PROMPT
    text = _generate(system_prompt, user_message)
    return {
        "answer": text,
        "candidates": candidates,
        "source": "vector",
        "matched_model": matched_model,
    }


def answer(query: str, top_k: int = 3, model_filter: str | None = None) -> dict:
    """에러코드가 있으면 RDB 조회를 기본으로 하되, 벡터 검색도 병행해서
    에러코드 외 다른 질문이 섞인 복합 질문도 커버한다.

    model_filter: UI 등에서 사용자가 제품을 미리 선택한 경우 그 모델명을 넘기면
    벡터 검색 범위를 해당 모델 문서로 제한한다."""
    matched_codes = find_codes_in_text(query)
    if matched_codes:
        solutions = _lookup_rdb_solutions(matched_codes)
        # 에러코드가 있어도 벡터 검색을 함께 돌려서 복합 질문을 커버한다.
        matched_model = model_filter or find_model_in_text(query)
        vector_candidates = search(query, top_k, model_filter=matched_model)
        if vector_candidates:
            return _answer_combined(query, matched_codes, solutions, vector_candidates)
        # 벡터 결과가 없으면 RDB만으로 답한다 (컬렉션 비어있는 경우 등).
        rdb_context = "\n\n".join(
            f"[{s['title']}] (관련 코드: {s['all_aliases']})\n{s['content']}" for s in solutions
        )
        user_message = f"[공식 에러코드 정보 (RDB 정확 조회)]\n{rdb_context}\n\n[사용자 질문]\n{query}"
        text = _generate(RDB_SYSTEM_PROMPT, user_message)
        candidates = [
            {
                "id": str(s["solution_id"]),
                "text": s["content"],
                "metadata": {"section_title": s["title"], "source": "RDB", "aliases": s["all_aliases"]},
                "distance": 0.0,
            }
            for s in solutions
        ]
        return {"answer": text, "candidates": candidates, "source": "rdb", "matched_codes": matched_codes}
    return _answer_from_vector(query, top_k, model_filter)


def main():
    parser = argparse.ArgumentParser(description="자연어 문서 검색/질의응답")
    parser.add_argument("query", help="자연어 질문")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    result = answer(args.query, args.top_k)
    print("\n=== 답변 ===")
    print(result["answer"])
    print("\n=== 근거 문서 ===")
    for c in result["candidates"]:
        print(f"- (distance={c['distance']:.4f}) {c['text'][:80]}")


if __name__ == "__main__":
    main()
