"""
Centralised configuration loaded from environment / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


# ── Endee Vector Database ────────────────────────────────────────────────────
ENDEE_HOST: str = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")
ENDEE_INDEX_NAME: str = os.getenv("ENDEE_INDEX_NAME", "rag_documents")
ENDEE_DIMENSION: int = int(os.getenv("ENDEE_DIMENSION", "384"))

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── LLM / Generation ────────────────────────────────────────────────────────
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID: str = os.getenv("HF_MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")
USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

# ── RAG settings ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── Paths ────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR: Path = Path(__file__).resolve().parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)
