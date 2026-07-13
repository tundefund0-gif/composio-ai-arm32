"""Application configuration — uses environment variables only."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    opencode_api_key: str = field(default_factory=lambda: os.getenv("OPENGATE_API_KEY", ""))
    opencode_base_url: str = field(default_factory=lambda: os.getenv("OPENGATE_BASE_URL", "https://opencode.ai/zen/v1"))
    opencode_model: str = field(default_factory=lambda: os.getenv("OPENGATE_MODEL", "deepseek-v4-flash-free"))
    opencode_max_tokens: int = 131000
    composio_api_key: str = field(default_factory=lambda: os.getenv("COMPOSIO_API_KEY", ""))
    composio_base_url: str = field(default_factory=lambda: os.getenv("COMPOSIO_BASE_URL", "https://backend.composio.dev"))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

config = Config()
