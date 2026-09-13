"""전역 설정. .env 파일에서 값을 읽어온다."""
import os
import sys
from dotenv import load_dotenv

# Windows 콘솔 기본 코드페이지(cp949 등)에서 한글 출력이 깨지는 것을 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# EMBEDDING_BACKEND: "local"(무료, sentence-transformers) | "openai"(유료)
# CHAT_BACKEND:      "local"(무료, transformers 직접 구동) | "ollama"(무료, Ollama 서버) | "openai"(유료)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
CHAT_BACKEND = os.getenv("CHAT_BACKEND", "local")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# local 백엔드용 (이 프로젝트 환경의 HuggingFace 캐시에 이미 받아져 있는 모델을 기본값으로 사용)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# CHAT_BACKEND=ollama 를 쓸 경우에만 사용
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
# 임베딩 백엔드마다 벡터 차원이 다르므로(local=bge-m3 1024차원, openai=1536차원) 컬렉션을 분리한다.
# 같은 이름의 컬렉션에 서로 다른 차원의 임베딩을 섞으면 Chroma가 에러를 낸다.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", f"appliance_manuals_{EMBEDDING_BACKEND}")

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw_documents.json")
