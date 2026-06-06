"""
config/settings.py
──────────────────
Single source of truth for all environment variables.
Every other module imports from here — never reads os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class LLMSettings:
    # Groq as primary LLM
    GROQ_API_KEY: str  = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int    = 2048


class DatabaseSettings:
    HOST: str     = os.getenv("POSTGRES_HOST", "localhost")
    PORT: int     = int(os.getenv("POSTGRES_PORT", "5432"))
    DB: str       = os.getenv("POSTGRES_DB", "claims_db")
    USER: str     = os.getenv("POSTGRES_USER", "claims_user")
    PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "claims_pass")

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DB}"
        )


class RAGSettings:
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "rag/chroma_db")
    )
    EMBEDDING_MODEL: str  = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int       = 512
    CHUNK_OVERLAP: int    = 50
    TOP_K_RETRIEVAL: int  = 5
    COLLECTION_NAME: str  = "insurance_policies"


class MLSettings:
    MLFLOW_TRACKING_URI: str   = os.getenv(
        "MLFLOW_TRACKING_URI", str(BASE_DIR / "mlruns")
    )
    MODELS_DIR: str              = str(BASE_DIR / "ml_models/saved_models")
    FRAUD_MODEL_NAME: str        = "fraud_detector"
    LITIGATION_MODEL_NAME: str   = "litigation_predictor"
    RESOLUTION_MODEL_NAME: str   = "resolution_forecaster"


class APISettings:
    SECRET_KEY: str = os.getenv("API_SECRET_KEY", "dev-secret-change-in-prod")
    HOST: str       = os.getenv("API_HOST", "0.0.0.0")
    PORT: int       = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool     = os.getenv("DEBUG", "false").lower() == "true"


class RedisSettings:
    HOST: str = os.getenv("REDIS_HOST", "localhost")
    PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    @property
    def url(self) -> str:
        return f"redis://{self.HOST}:{self.PORT}"


# ── Instantiate once, import everywhere ──────────
llm_settings   = LLMSettings()
db_settings    = DatabaseSettings()
rag_settings   = RAGSettings()
ml_settings    = MLSettings()
api_settings   = APISettings()
redis_settings = RedisSettings()