from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # TODO: read env vars for one of the supported providers.
    # Example knobs:
    # - LLM_PROVIDER / LLM_MODEL
    # - OPENAI_API_KEY
    # - GEMINI_API_KEY
    # - ANTHROPIC_API_KEY
    # - OLLAMA_BASE_URL
    # - OPENROUTER_API_KEY
    # - CUSTOM_BASE_URL / CUSTOM_API_KEY

    # Try to load .env file if it exists
    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    # Resolve provider
    provider = normalize_provider(os.environ.get("LLM_PROVIDER", "openai"))
    model_name = os.environ.get("LLM_MODEL", _default_model(provider))
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

    api_key = _resolve_api_key(provider)
    base_url = os.environ.get("CUSTOM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge model defaults to same provider but can be overridden
    judge_provider = normalize_provider(os.environ.get("JUDGE_PROVIDER", provider))
    judge_model_name = os.environ.get("JUDGE_MODEL", model_name)
    judge_api_key = _resolve_api_key(judge_provider)
    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=base_url,
    )

    # TODO: create `root / "state"`.
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # TODO: choose sensible defaults for compact memory.
    # Threshold: ~2000 tokens triggers compaction (reasonable for most models)
    # Keep messages: last 6 messages stay verbatim after compaction
    compact_threshold_tokens = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", "2000"))
    compact_keep_messages = int(os.environ.get("COMPACT_KEEP_MESSAGES", "6"))

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_cfg,
        judge_model=judge_cfg,
    )


def _default_model(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-haiku-4-5-20251001",
        "ollama": "llama3",
        "openrouter": "openai/gpt-4o-mini",
        "custom": "gpt-4o-mini",
    }
    return defaults.get(provider, "gpt-4o-mini")


def _resolve_api_key(provider: str) -> str | None:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None
