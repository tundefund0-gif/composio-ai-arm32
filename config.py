"""Application configuration — environment variables with .env support."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _load_dotenv():
    """Load .env file if present (lightweight, no dependency)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if len(val) > 1 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                if key not in os.environ:
                    os.environ[key] = val


_load_dotenv()


@dataclass
class Config:
    # LLM
    opencode_api_key: str = field(default_factory=lambda: os.environ.get("OPENGATE_API_KEY", ""))
    opencode_base_url: str = field(default_factory=lambda: os.environ.get("OPENGATE_BASE_URL", "https://opencode.ai/zen/v1"))
    opencode_model: str = field(default_factory=lambda: os.environ.get("OPENGATE_MODEL", "deepseek-v4-flash-free"))
    opencode_max_tokens: int = int(os.environ.get("OPENGATE_MAX_TOKENS", "131072"))
    opencode_fallback_model: str = field(default_factory=lambda: os.environ.get("OPENGATE_FALLBACK_MODEL", ""))
    opencode_fallback_base_url: str = field(default_factory=lambda: os.environ.get("OPENGATE_FALLBACK_BASE_URL", ""))
    opencode_fallback_api_key: str = field(default_factory=lambda: os.environ.get("OPENGATE_FALLBACK_API_KEY", ""))

    # Composio
    composio_api_key: str = field(default_factory=lambda: os.environ.get("COMPOSIO_API_KEY", ""))
    composio_base_url: str = field(default_factory=lambda: os.environ.get("COMPOSIO_BASE_URL", "https://backend.composio.dev"))

    # Server
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "9090"))
    cors_origins: str = os.environ.get("CORS_ORIGINS", "*")
    rate_limit_per_minute: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

    # Paths
    data_dir: str = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

    # Limits
    max_history_messages: int = int(os.environ.get("MAX_HISTORY_MESSAGES", "60"))
    max_tool_results_length: int = int(os.environ.get("MAX_TOOL_RESULTS_LENGTH", "15000"))
    request_timeout: float = float(os.environ.get("REQUEST_TIMEOUT", "300.0"))
    llm_timeout: float = float(os.environ.get("LLM_TIMEOUT", "300.0"))
    max_message_length: int = int(os.environ.get("MAX_MESSAGE_LENGTH", "100000"))

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    log_format: str = os.environ.get("LOG_FORMAT", "default")  # default or json


config = Config()


def validate_config() -> list[str]:
    """Validate configuration and return list of warnings."""
    warnings = []
    if not config.opencode_api_key:
        warnings.append("OPENGATE_API_KEY is not set — LLM calls will fail")
    if not config.composio_api_key:
        warnings.append("COMPOSIO_API_KEY is not set — Composio tools will fail")
    if config.max_tool_results_length < 1000:
        warnings.append("MAX_TOOL_RESULTS_LENGTH is very low (< 1000)")
    if config.max_history_messages < 5:
        warnings.append("MAX_HISTORY_MESSAGES is very low (< 5)")
    if config.port < 1 or config.port > 65535:
        warnings.append("PORT is invalid (must be 1-65535)")
    if config.rate_limit_per_minute < 1:
        warnings.append("RATE_LIMIT_PER_MINUTE must be >= 1")
    if config.max_message_length < 1000:
        warnings.append("MAX_MESSAGE_LENGTH is very low (< 1000)")
    if config.opencode_fallback_model and not config.opencode_fallback_api_key:
        warnings.append("OPENGATE_FALLBACK_MODEL set but OPENGATE_FALLBACK_API_KEY is missing")
    return warnings


# Validate on import
_validate_warnings = validate_config()
if _validate_warnings:
    import sys
    for w in _validate_warnings:
        print(f"\u26a0\ufe0f  Config warning: {w}", file=sys.stderr)


