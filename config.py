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

    # Composio
    composio_api_key: str = field(default_factory=lambda: os.environ.get("COMPOSIO_API_KEY", ""))
    composio_base_url: str = field(default_factory=lambda: os.environ.get("COMPOSIO_BASE_URL", "https://backend.composio.dev"))

    # Server
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "9090"))

    # Limits
    max_history_messages: int = int(os.environ.get("MAX_HISTORY_MESSAGES", "60"))
    max_tool_results_length: int = int(os.environ.get("MAX_TOOL_RESULTS_LENGTH", "15000"))
    request_timeout: float = float(os.environ.get("REQUEST_TIMEOUT", "300.0"))
    llm_timeout: float = float(os.environ.get("LLM_TIMEOUT", "300.0"))


config = Config()
