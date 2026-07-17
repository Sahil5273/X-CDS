"""Stateful runtime settings for LangGraph and Gemini generation."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration for the X-CDS backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="X-CDS", alias="APP_NAME")
    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    data_dir: str = Field(default="./data", alias="DATA_DIR")
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="xcds_biomedical",
        alias="CHROMA_COLLECTION_NAME",
    )

    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="CROSS_ENCODER_MODEL_NAME",
    )
    cross_encoder_device: str = Field(default="cpu", alias="CROSS_ENCODER_DEVICE")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, alias="GEMINI_TEMPERATURE")
    langgraph_max_generation_attempts: int = Field(
        default=3,
        alias="LANGGRAPH_MAX_GENERATION_ATTEMPTS",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for the current process."""

    return Settings()
