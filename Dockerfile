# 찾아드림 웹 앱 — 공유 DB(Supabase) 모드 전용. 기본 포트 7860(Hugging Face Spaces), $PORT 로 변경 가능.
FROM python:3.12-slim

# torch 가 쓰는 OpenMP 런타임
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces 는 uid 1000 으로 실행한다
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8
WORKDIR /home/user/app

# CPU 전용 torch 를 먼저 깔아 requirements.txt 의 torch>=2.2 가 이미 충족되게 한다(CUDA 빌드 수 GB 방지)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 임베딩 모델을 이미지에 미리 받아 둔다 — 없으면 컨테이너가 뜰 때마다 수 GB 를 다시 받는다
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('dragonkue/BGE-m3-ko')"

COPY --chown=user . .
# PDF·페이지 이미지 캐시용(Git 제외 폴더). 컨테이너를 다시 만들면 비워지고, 필요할 때 Supabase Storage 에서 다시 받는다
RUN mkdir -p data

EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn web.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
