"""LG 에어컨 매뉴얼 RAG (siyeon).

  chunking_lg.py   PDF 목차(TOC) 기반 청킹 + 에러코드 JSON 로더  →  list[dict] 청크
  retrieval.py  LGAirconRetriever: 벡터(본문+예상질문) + 조건부 BM25 + 리랭커 + SQLite 라우터

실험 실행 파일은 experiments/siyeon/exp/exp05_all.py, 탐색/시각화는 experiments/siyeon/notebook_lg_aircon.ipynb.
설계 설명은 experiments/siyeon/docs/overview_lg.md, 실험 기록은 experiments/siyeon/docs/experiment_notes.md.
"""
