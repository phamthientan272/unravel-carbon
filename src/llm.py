"""LLM client factory and message helpers, shared by the graph and the tools.

The provider is chosen from the environment, so swapping models needs no code change:

1. `LLM_MODEL=anthropic:claude-sonnet-5-5` (provider prefix), or
2. `LLM_PROVIDER=openai` plus `LLM_MODEL=...`, or
3. nothing set: the first provider whose API key is in the environment is used
   (order: google_genai, anthropic, openai; ollama only when named explicitly).

Provider packages are imported only when used, so you install just the one you need.
Everything else in the app (router, tools, structured output) works with any chat model that supports tool calling.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage

from src.data import ROOT

load_dotenv(ROOT / ".env")

TIMEOUT_S = 120  # free-tier calls occasionally hang
MAX_RETRIES = 2


class LLMConfigError(RuntimeError):
    """The LLM provider or model cannot be determined or loaded. The message says how to fix it."""


@dataclass(frozen=True)
class Provider:
    name: str
    module: str
    cls: str
    package: str                       # pip/uv package that provides the class
    key_vars: tuple[str, ...]          # environment variables that hold its API key (any one)
    default_model: str | None = None   # None: LLM_MODEL must be set


PROVIDERS: dict[str, Provider] = {p.name: p for p in (
    Provider("google_genai", "langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai",
             ("GEMINI_API_KEY", "GOOGLE_API_KEY"), default_model="gemini-4-hydrogen"),
    Provider("anthropic", "langchain_anthropic", "ChatAnthropic", "langchain-anthropic",
             ("ANTHROPIC_API_KEY",), default_model="claude-sonnet-5-5"),
    Provider("openai", "langchain_openai", "ChatOpenAI", "langchain-openai", ("OPENAI_API_KEY",)),
    Provider("ollama", "langchain_ollama", "ChatOllama", "langchain-ollama", ()),
)}
ALIASES = {"google": "google_genai", "gemini": "google_genai", "google-genai": "google_genai", "claude": "anthropic"}
AUTO_ORDER = ("google_genai", "anthropic", "openai")  # ollama has no key, so it is never auto-selected


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    source: str  # how the provider was chosen, for the trace

    def __str__(self) -> str:
        return f"{self.provider}:{self.model} ({self.source})"


def _canonical(name: str) -> str | None:
    n = name.strip().lower()
    n = ALIASES.get(n, n)
    return n if n in PROVIDERS else None


def _key_var(p: Provider) -> str | None:
    return next((v for v in p.key_vars if os.getenv(v)), None)


def resolve_llm() -> LLMConfig:
    """Decide provider and model from the environment (read at call time)."""
    model = (os.getenv("LLM_MODEL") or "").strip()
    provider_env = (os.getenv("LLM_PROVIDER") or "").strip()
    provider: str | None = None
    source = ""

    if ":" in model and _canonical(model.split(":", 1)[0]):  # "anthropic:claude-..."; "llama3.1:8b" is a model name, not a prefix
        prefix, model = model.split(":", 1)
        provider, source = _canonical(prefix), "from LLM_MODEL prefix"
    elif provider_env:
        provider = _canonical(provider_env)
        if provider is None:
            raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider_env}'. Choose one of: {', '.join(PROVIDERS)}.")
        source = "from LLM_PROVIDER"
    else:
        for name in AUTO_ORDER:
            var = _key_var(PROVIDERS[name])
            if var:
                provider, source = name, f"auto-detected from {var}"
                break
    if provider is None:
        raise LLMConfigError(
            "No LLM configured. Set one API key (GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY), or set "
            "LLM_PROVIDER and LLM_MODEL (for example LLM_PROVIDER=ollama LLM_MODEL=llama3.1).")

    spec = PROVIDERS[provider]
    model = model or spec.default_model or ""
    if not model:
        raise LLMConfigError(f"Provider '{provider}' has no default model. Set LLM_MODEL (for example LLM_MODEL={provider}:<model name>).")
    return LLMConfig(provider, model, source)


def describe_llm() -> str:
    """Which provider and model are in use, or why none is."""
    try:
        return str(resolve_llm())
    except LLMConfigError as e:
        return f"not configured: {e}"


def get_llm(temperature: float = 0.2):
    """Build the chat model for the configured provider."""
    cfg = resolve_llm()
    spec = PROVIDERS[cfg.provider]
    try:
        cls = getattr(importlib.import_module(spec.module), spec.cls)
    except ImportError as e:
        raise LLMConfigError(f"Provider '{cfg.provider}' needs the {spec.package} package. Install it with `uv add {spec.package}`.") from e

    kwargs: dict = {"model": cfg.model, "temperature": temperature}
    if cfg.provider == "google_genai":
        kwargs |= {"google_api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), "max_retries": MAX_RETRIES, "timeout": TIMEOUT_S}
    elif cfg.provider == "anthropic":
        kwargs |= {"max_retries": MAX_RETRIES, "timeout": TIMEOUT_S, "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096"))}
    elif cfg.provider == "openai":
        kwargs |= {"max_retries": MAX_RETRIES, "timeout": TIMEOUT_S}
    elif cfg.provider == "ollama" and os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = os.environ["OLLAMA_BASE_URL"]
    return cls(**kwargs)


def text_of(msg: AnyMessage) -> str:
    """Message content as plain text. Some providers return a list of content blocks (text, thinking, tool use)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
