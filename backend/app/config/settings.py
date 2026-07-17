"""Stateful runtime settings for LangGraph and Gemini generation."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# backend/app/config/settings.py -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"

# Free-tier friendly default. gemini-3.5-flash is currently supported.
_DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
_BLOCKED_GEMINI_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash",
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file (no export/interpolation)."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _resolve_gemini_model(raw: str | None) -> str:
    candidate = (raw or "").strip() or _DEFAULT_GEMINI_MODEL
    if (
        candidate in _BLOCKED_GEMINI_MODELS
        or "2.0-flash" in candidate
        or "2.5-flash" in candidate
        or "1.5-flash" in candidate
    ):
        return _DEFAULT_GEMINI_MODEL
    return candidate


class Settings(BaseSettings):
    """Environment-backed configuration for the X-CDS backend."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(
        default="X-CDS",
        validation_alias=AliasChoices("APP_NAME", "app_name"),
    )
    app_env: str = Field(
        default="local",
        validation_alias=AliasChoices("APP_ENV", "app_env"),
    )
    app_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("APP_HOST", "app_host"),
    )
    app_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("APP_PORT", "app_port"),
    )
    log_level: str = Field(
        default="info",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )

    data_dir: str = Field(
        default="./data",
        validation_alias=AliasChoices("DATA_DIR", "data_dir"),
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        validation_alias=AliasChoices("CHROMA_PERSIST_DIR", "chroma_persist_dir"),
    )
    chroma_collection_name: str = Field(
        default="xcds_biomedical",
        validation_alias=AliasChoices(
            "CHROMA_COLLECTION_NAME",
            "chroma_collection_name",
        ),
    )
    bm25_corpus_path: str = Field(
        default="./data/bm25_corpus.jsonl",
        validation_alias=AliasChoices("BM25_CORPUS_PATH", "bm25_corpus_path"),
    )

    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias=AliasChoices("EMBEDDING_MODEL_NAME", "embedding_model_name"),
    )
    eval_embedding_model: str = Field(
        default="models/text-embedding-004",
        validation_alias=AliasChoices("EVAL_EMBEDDING_MODEL", "eval_embedding_model"),
    )
    embedding_device: str = Field(
        default="cpu",
        validation_alias=AliasChoices("EMBEDDING_DEVICE", "embedding_device"),
    )

    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices(
            "CROSS_ENCODER_MODEL_NAME",
            "cross_encoder_model_name",
        ),
    )
    cross_encoder_device: str = Field(
        default="cpu",
        validation_alias=AliasChoices("CROSS_ENCODER_DEVICE", "cross_encoder_device"),
    )
    rerank_top_k: int = Field(
        default=5,
        validation_alias=AliasChoices("RERANK_TOP_K", "rerank_top_k"),
    )

    gcp_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GCP_PROJECT_ID", "gcp_project_id"),
    )
    gcp_region: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("GCP_REGION", "gcp_region"),
    )
    google_application_credentials: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "google_application_credentials",
        ),
    )
    gemini_model: str = Field(
        default=_DEFAULT_GEMINI_MODEL,
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"),
    )
    gemini_temperature: float = Field(
        default=0.1,
        validation_alias=AliasChoices("GEMINI_TEMPERATURE", "gemini_temperature"),
    )
    langgraph_max_generation_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "LANGGRAPH_MAX_GENERATION_ATTEMPTS",
            "langgraph_max_generation_attempts",
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Prefer project .env over ambient OS/shell environment variables.
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )

    @field_validator("gemini_model", mode="before")
    @classmethod
    def normalize_gemini_model(cls, value: Any) -> str:
        return _resolve_gemini_model(str(value) if value is not None else None)

    @model_validator(mode="after")
    def force_project_dotenv_gemini(self) -> Settings:
        """Final authority: project .env GEMINI_MODEL, then safe default."""

        file_values = _parse_dotenv(_ENV_FILE)
        if "GEMINI_MODEL" in file_values:
            self.gemini_model = _resolve_gemini_model(file_values["GEMINI_MODEL"])
        else:
            self.gemini_model = _resolve_gemini_model(self.gemini_model)
        if "GOOGLE_APPLICATION_CREDENTIALS" in file_values:
            self.google_application_credentials = file_values["GOOGLE_APPLICATION_CREDENTIALS"]
        if "EVAL_EMBEDDING_MODEL" in file_values:
            self.eval_embedding_model = file_values["EVAL_EMBEDDING_MODEL"]
        # Keep process env aligned for downstream libs and GCP configuration
        os.environ["GEMINI_MODEL"] = self.gemini_model
        os.environ["EVAL_EMBEDDING_MODEL"] = self.eval_embedding_model
        if self.gcp_project_id:
            os.environ["GCP_PROJECT_ID"] = self.gcp_project_id
        os.environ["GCP_REGION"] = self.gcp_region
        if self.google_application_credentials:
            credentials_path = Path(self.google_application_credentials)
            if not credentials_path.is_absolute():
                credentials_path = _PROJECT_ROOT / credentials_path
            resolved_path = credentials_path.resolve()
            import warnings
            if not resolved_path.exists():
                warnings.warn(f"[X-CDS] GOOGLE_APPLICATION_CREDENTIALS file not found at: {resolved_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved_path)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for the current process."""

    return Settings()


def clear_settings_cache() -> None:
    """Drop cached settings so a restarted process reloads `.env` values."""

    get_settings.cache_clear()
