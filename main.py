"""
가전제품 사용·문제해결 RAG 서비스 CLI 진입점.

사용 예:
    python main.py ingest-manuals --input data/sample_manuals.json
    python main.py query "세탁기에서 E1 오류가 떴어요"
    python main.py ingest --input data/documents.json   # 이미 {id,text,metadata}로 만들어진 문서일 때
"""
import argparse


def cmd_ingest(args):
    from pipeline.embed_store import ingest

    ingest(args.input)


def cmd_ingest_manuals(args):
    from pipeline.build_documents import build_documents
    from pipeline.embed_store import ingest_documents

    docs = build_documents(args.input)
    ingest_documents(docs)


def cmd_query(args):
    from pipeline.query_engine import answer

    result = answer(args.text, args.top_k)
    print("\n=== 답변 ===")
    print(result["answer"])
    print("\n=== 근거 문서 ===")
    for c in result["candidates"]:
        print(f"- (distance={c['distance']:.4f}) {c['text'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="RAG 파이프라인")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="이미 {id,text,metadata}로 만들어진 문서 JSON을 저장")
    p_ingest.add_argument("--input", required=True, help="{id, text, metadata} 문서 JSON 파일 경로")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ingest_manuals = sub.add_parser("ingest-manuals", help="가전제품 설명서 JSON을 청킹 후 임베딩하여 저장")
    p_ingest_manuals.add_argument("--input", default="data/sample_manuals.json")
    p_ingest_manuals.set_defaults(func=cmd_ingest_manuals)

    p_query = sub.add_parser("query", help="자연어로 문서 검색/질의응답")
    p_query.add_argument("text")
    p_query.add_argument("--top-k", type=int, default=3)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
