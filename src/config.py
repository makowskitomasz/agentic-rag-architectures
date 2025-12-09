from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict
sys.path.append(os.path.join(os.getcwd(), '..'))

from dotenv import load_dotenv

load_dotenv(override=False)

SUPPORTED_PROVIDERS = {"openai", "gemini"}
DEFAULT_EMBEDDING_MODELS: Dict[str, str] = {
    "openai": "text-embedding-3-large",
    "gemini": "embeddings/embedding-001",
}
DEFAULT_LLM_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}


@dataclass(frozen=True)
class RoutingProfile:
    name: str
    embedding_provider: str
    embedding_model: str
    description: str = ""


DEFAULT_ROUTING_PROFILES: Dict[str, RoutingProfile] = {
    "balanced_openai": RoutingProfile(
        name="balanced_openai",
        embedding_provider="openai",
        embedding_model=DEFAULT_EMBEDDING_MODELS["openai"],
        description="High accuracy OpenAI embeddings",
    ),
    "fast_gemini": RoutingProfile(
        name="fast_gemini",
        embedding_provider="gemini",
        embedding_model=DEFAULT_EMBEDDING_MODELS["gemini"],
        description="Faster Gemini embeddings for exploratory questions",
    ),
}


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    api_key: str


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str


def _resolve_provider(provider: str | None, env_var: str, default: str = "openai") -> str:
    provider_name = (provider or os.getenv(env_var) or default).lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider_name}'. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}")
    return provider_name


def _require_api_key(provider: str) -> str:
    env_var = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} environment variable is required for provider '{provider}'.")
    return api_key


@lru_cache(maxsize=None)
def get_embedding_config(provider: str | None = None) -> EmbeddingConfig:
    provider_name = _resolve_provider(provider, "EMBEDDING_PROVIDER")
    model_env = f"{provider_name.upper()}_EMBEDDING_MODEL"
    model = os.getenv(model_env) or DEFAULT_EMBEDDING_MODELS[provider_name]
    api_key = _require_api_key(provider_name)
    return EmbeddingConfig(provider=provider_name, model=model, api_key=api_key)


@lru_cache(maxsize=None)
def get_llm_config(provider: str | None = None) -> LLMConfig:
    provider_name = _resolve_provider(provider, "LLM_PROVIDER")
    model_env = f"{provider_name.upper()}_LLM_MODEL"
    model = os.getenv(model_env) or DEFAULT_LLM_MODELS[provider_name]
    api_key = _require_api_key(provider_name)
    return LLMConfig(provider=provider_name, model=model, api_key=api_key)


@lru_cache(maxsize=None)
def get_routing_profiles() -> Dict[str, RoutingProfile]:
    return DEFAULT_ROUTING_PROFILES.copy()
