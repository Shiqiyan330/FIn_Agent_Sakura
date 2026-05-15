"""Runtime configuration loaded from environment variables and optional .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if it exists."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """OpenAI-compatible LLM and embedding endpoint configuration."""

    api_key: str | None
    base_url: str | None
    chat_model: str
    embedding_model: str


@dataclass(frozen=True, slots=True)
class TushareConfig:
    """TuShare Pro endpoint configuration."""

    token: str | None
    http_url: str | None


def get_llm_config() -> LLMConfig:
    """Return LLM configuration, loading .env first."""

    load_dotenv()
    return LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def get_tushare_config() -> TushareConfig:
    """Return TuShare configuration, loading .env first."""

    load_dotenv()
    return TushareConfig(
        token=os.getenv("TUSHARE_TOKEN"),
        http_url=os.getenv("TUSHARE_HTTP_URL"),
    )
