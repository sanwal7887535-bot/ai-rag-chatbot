"""Centralized configuration, validation, and secret sanitization."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

BASE_DIR = Path(__file__).resolve().parent

SUPPORTED_PROVIDERS = ("groq", "ollama", "gemini")

DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen2.5:3b",
    "gemini": "gemini-2.5-flash",
}

DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"


def _load_local_secrets(base_dir: Path) -> dict[str, Any]:
    """Load optional local `.streamlit/secrets.toml` without crashing when absent."""
    secrets_path = base_dir / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return {}
    try:
        with secrets_path.open("rb") as handle:
            data = tomllib.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Invalid secrets file at {secrets_path}: {error}") from error


def _get_setting(
    name: str,
    local_secrets: Mapping[str, Any],
    env: Mapping[str, str],
    default: str | None = None,
) -> str | None:
    """Resolve a setting with environment variables taking precedence over local secrets."""
    env_value = env.get(name)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    secret_value = local_secrets.get(name)
    if secret_value is not None:
        text = str(secret_value).strip()
        if text:
            return text

    return default


def _parse_bool(value: str | bool | None, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _parse_int(
    value: str | int | None,
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid integer value for {name}: {value!r}") from error

    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {parsed}.")
    return parsed


def _validate_url(url: str, name: str, *, allow_http: bool) -> str:
    parsed = urlparse(url)
    allowed_schemes = ("https", "http") if allow_http else ("https",)
    if parsed.scheme not in allowed_schemes or not parsed.netloc:
        schemes_label = " or ".join(allowed_schemes)
        raise ValueError(f"{name} must be a valid {schemes_label} URL.")
    return url


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def sanitize_error_message(message: str, secrets: Iterable[str | None]) -> str:
    """Redact any configured secret values from diagnostic or error strings."""
    cleaned = message
    for secret in secrets:
        if secret and len(secret) >= 4:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from env vars and optional local secrets."""

    llm_provider: str
    llm_model: str
    groq_api_key: str | None
    groq_api_url: str
    ollama_api_url: str
    gemini_api_key: str | None
    tavily_api_key: str | None
    chroma_db_dir: Path
    llm_log_path: Path
    enable_file_logging: bool
    log_redact_content: bool
    embedding_model_name: str
    max_pdf_mb: int
    max_tool_rounds: int
    max_chat_history: int
    max_prompt_chars: int
    max_context_chars: int
    request_timeout_sec: int
    default_timezone: str
    rag_top_k: int
    chunk_size: int
    chunk_overlap: int
    allow_provider_switch: bool

    @property
    def max_pdf_bytes(self) -> int:
        return self.max_pdf_mb * 1024 * 1024

    @property
    def secret_values(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.groq_api_key, self.gemini_api_key, self.tavily_api_key)
            if value
        )

    def available_providers(self) -> list[str]:
        """Return providers ready to use in this environment."""
        providers = [self.llm_provider]
        if self.groq_api_key and "groq" not in providers:
            providers.append("groq")
        if self.gemini_api_key and "gemini" not in providers:
            providers.append("gemini")
        return providers

    def sanitize_text(self, text: str) -> str:
        return sanitize_error_message(text, self.secret_values)


def load_settings(
    base_dir: Path = BASE_DIR,
    env: Mapping[str, str] | None = None,
    local_secrets: Mapping[str, Any] | None = None,
) -> Settings:
    """Build and validate Settings from environment variables and optional secrets.toml."""
    effective_env = os.environ if env is None else env
    effective_secrets = (
        _load_local_secrets(base_dir) if local_secrets is None else local_secrets
    )

    provider = (
        _get_setting("LLM_PROVIDER", effective_secrets, effective_env, "groq") or "groq"
    ).lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unsupported LLM_PROVIDER {provider!r}. Expected one of: {supported}."
        )

    llm_model = (
        _get_setting("LLM_MODEL", effective_secrets, effective_env)
        or DEFAULT_MODELS[provider]
    )

    groq_api_url = _validate_url(
        _get_setting(
            "GROQ_API_URL", effective_secrets, effective_env, DEFAULT_GROQ_URL
        )
        or DEFAULT_GROQ_URL,
        "GROQ_API_URL",
        allow_http=False,
    )
    ollama_api_url = _validate_url(
        _get_setting(
            "OLLAMA_API_URL", effective_secrets, effective_env, DEFAULT_OLLAMA_URL
        )
        or DEFAULT_OLLAMA_URL,
        "OLLAMA_API_URL",
        allow_http=True,
    )

    gemini_api_key = _get_setting(
        "GEMINI_API_KEY", effective_secrets, effective_env
    ) or _get_setting("GOOGLE_API_KEY", effective_secrets, effective_env)

    raw_db_dir = (
        _get_setting("CHROMA_DB_DIR", effective_secrets, effective_env)
        or str(base_dir / "chroma_db")
    )
    raw_log_path = (
        _get_setting("LLM_LOG_PATH", effective_secrets, effective_env)
        or str(base_dir / "llm_calls.jsonl")
    )

    timezone_name = (
        _get_setting("DEFAULT_TIMEZONE", effective_secrets, effective_env, "Asia/Karachi")
        or "Asia/Karachi"
    )
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Invalid DEFAULT_TIMEZONE: {timezone_name!r}") from error

    chunk_size = _parse_int(
        _get_setting("CHUNK_SIZE", effective_secrets, effective_env),
        "CHUNK_SIZE",
        500,
        minimum=100,
        maximum=4000,
    )
    chunk_overlap = _parse_int(
        _get_setting("CHUNK_OVERLAP", effective_secrets, effective_env),
        "CHUNK_OVERLAP",
        50,
        minimum=0,
        maximum=chunk_size - 1,
    )

    return Settings(
        llm_provider=provider,
        llm_model=llm_model,
        groq_api_key=_get_setting("GROQ_API_KEY", effective_secrets, effective_env),
        groq_api_url=groq_api_url,
        ollama_api_url=ollama_api_url,
        gemini_api_key=gemini_api_key,
        tavily_api_key=_get_setting("TAVILY_API_KEY", effective_secrets, effective_env),
        chroma_db_dir=_resolve_path(raw_db_dir, base_dir),
        llm_log_path=_resolve_path(raw_log_path, base_dir),
        enable_file_logging=_parse_bool(
            _get_setting("ENABLE_FILE_LOGGING", effective_secrets, effective_env),
            "ENABLE_FILE_LOGGING",
            True,
        ),
        log_redact_content=_parse_bool(
            _get_setting("LOG_REDACT_CONTENT", effective_secrets, effective_env),
            "LOG_REDACT_CONTENT",
            False,
        ),
        embedding_model_name=(
            _get_setting(
                "EMBEDDING_MODEL_NAME",
                effective_secrets,
                effective_env,
                "all-MiniLM-L6-v2",
            )
            or "all-MiniLM-L6-v2"
        ),
        max_pdf_mb=_parse_int(
            _get_setting("MAX_PDF_MB", effective_secrets, effective_env),
            "MAX_PDF_MB",
            25,
            minimum=1,
            maximum=200,
        ),
        max_tool_rounds=_parse_int(
            _get_setting("MAX_TOOL_ROUNDS", effective_secrets, effective_env),
            "MAX_TOOL_ROUNDS",
            5,
            minimum=1,
            maximum=10,
        ),
        max_chat_history=_parse_int(
            _get_setting("MAX_CHAT_HISTORY", effective_secrets, effective_env),
            "MAX_CHAT_HISTORY",
            20,
            minimum=2,
            maximum=100,
        ),
        max_prompt_chars=_parse_int(
            _get_setting("MAX_PROMPT_CHARS", effective_secrets, effective_env),
            "MAX_PROMPT_CHARS",
            8000,
            minimum=100,
            maximum=50000,
        ),
        max_context_chars=_parse_int(
            _get_setting("MAX_CONTEXT_CHARS", effective_secrets, effective_env),
            "MAX_CONTEXT_CHARS",
            15000,
            minimum=500,
            maximum=100000,
        ),
        request_timeout_sec=_parse_int(
            _get_setting("REQUEST_TIMEOUT_SEC", effective_secrets, effective_env),
            "REQUEST_TIMEOUT_SEC",
            60,
            minimum=5,
            maximum=300,
        ),
        default_timezone=timezone_name,
        rag_top_k=_parse_int(
            _get_setting("RAG_TOP_K", effective_secrets, effective_env),
            "RAG_TOP_K",
            5,
            minimum=1,
            maximum=20,
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        allow_provider_switch=_parse_bool(
            _get_setting("ALLOW_PROVIDER_SWITCH", effective_secrets, effective_env),
            "ALLOW_PROVIDER_SWITCH",
            True,
        ),
    )
