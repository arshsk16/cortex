"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Cortex backend.

    Values are read from environment variables and an optional ``.env`` file.
    Nested configuration is intentionally flat so operators can set every
    knob with a single env var.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Cortex", description="Service display name")
    app_version: str = Field(default="0.1.0", description="Semantic version")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 URL prefix")

    # Server
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")

    # Database
    database_url: str = Field(
        ...,
        description="Async PostgreSQL connection URL (postgresql+asyncpg://...)",
        min_length=1,
    )
    database_pool_size: int = Field(default=5, ge=1, description="Connection pool size")
    database_max_overflow: int = Field(
        default=10,
        ge=0,
        description="Max overflow connections beyond pool_size",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements to the logger",
    )

    # CORS
    # NoDecode: allow comma-separated env values instead of JSON arrays.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
    )
    cors_allow_headers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level",
    )
    log_json: bool = Field(
        default=False,
        description="Emit logs as JSON (recommended in production)",
    )

    # Authentication / JWT
    jwt_secret_key: str = Field(
        ...,
        min_length=32,
        description="Secret key used to sign JWT access tokens",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="Access token lifetime in minutes",
    )

    # Document storage
    document_storage_path: str = Field(
        default="storage/documents",
        description="Directory where uploaded document files are stored",
    )
    document_max_file_size_bytes: int = Field(
        default=26_214_400,
        ge=1,
        description="Maximum upload size in bytes (default 25 MB)",
    )
    document_allowed_mime_type: str = Field(
        default="application/pdf",
        description="Allowed MIME type for document uploads",
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Ensure the URL uses an async PostgreSQL SQLAlchemy dialect."""
        normalized = value.strip()
        if not normalized.startswith("postgresql+asyncpg://"):
            msg = (
                "DATABASE_URL must use the asyncpg dialect "
                "(postgresql+asyncpg://user:pass@host:port/db)"
            )
            raise ValueError(msg)
        return normalized

    @field_validator(
        "cors_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        mode="before",
    )
    @classmethod
    def parse_comma_separated_list(cls, value: object) -> object:
        """Accept a comma-separated string or an existing list."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                import json

                return json.loads(stripped)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.app_env == "production"

    @property
    def database_url_str(self) -> str:
        """Return the database URL as a plain string for SQLAlchemy."""
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (safe for FastAPI dependency injection)."""
    return Settings()
