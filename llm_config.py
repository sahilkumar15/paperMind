"""
llm_config.py — Central LLM provider configuration for PaperMind
=================================================================

Supports two providers selectable from the .env or the sidebar:
  • groq    — FREE tier, fast, runs Llama / Qwen models
  • openai  — Paid, ~$0.025/run with gpt-4o-mini

CrewAI uses LiteLLM under the hood. With litellm installed the
format  "groq/llama-3.3-70b-versatile"  works natively.

KatzBot (RAG) uses LangChain — we use langchain-groq or
langchain-openai depending on the provider, and we fall back
to free HuggingFace embeddings when no OpenAI key is present.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Groq model catalogue (updated April 2026) ─────────────────
GROQ_MODELS = {
    # Production
    "llama-3.3-70b-versatile":  "Best quality · 300K TPM · 1K RPM (recommended)",
    "llama-3.1-8b-instant":     "Fastest · 250K TPM · 1K RPM · great for demos",
    "openai/gpt-oss-120b":      "GPT-class 120B · 250K TPM · 1K RPM",
    "openai/gpt-oss-20b":       "Fastest GPT-class · 1000 t/s · 250K TPM",
    # Preview
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout · vision · 300K TPM",
    "qwen/qwen3-32b":           "Qwen3 32B · 400 t/s · reasoning",
}

OPENAI_MODELS = {
    "gpt-4o-mini":  "Cheapest · ~$0.025/run · fast (recommended)",
    "gpt-4o":       "Best quality · ~$0.10/run",
    "gpt-3.5-turbo":"Cheapest OpenAI · lower quality",
}


# ── Provider helpers ──────────────────────────────────────────

def get_provider() -> str:
    return "groq" if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" else "openai"


def get_model_name() -> str:
    env = os.getenv("MODEL_NAME", "").strip()
    if env:
        return env
    return "llama-3.3-70b-versatile" if get_provider() == "groq" else "gpt-4o-mini"


def get_api_key() -> str:
    return (os.getenv("GROQ_API_KEY", "") if get_provider() == "groq"
            else os.getenv("OPENAI_API_KEY", ""))


def is_api_key_set() -> bool:
    key = get_api_key()
    if get_provider() == "groq":
        return bool(key and key.startswith("gsk_"))
    return bool(key and key.startswith("sk-"))


def get_crewai_llm_string() -> str:
    """
    LLM string for CrewAI agents.
    CrewAI uses LiteLLM, which accepts:
        groq/llama-3.3-70b-versatile
        openai/gpt-4o-mini
    The GROQ_API_KEY env var is picked up automatically by LiteLLM.
    """
    provider = get_provider()
    model    = get_model_name()

    if provider == "groq":
        # LiteLLM expects "groq/<model>" — env var GROQ_API_KEY is used automatically
        return f"groq/{model}"
    else:
        return f"openai/{model}"


def get_openai_client():
    """
    OpenAI-compatible client for chatbot / PaperBot calls.
    Groq exposes an OpenAI-compatible REST API so we reuse the same client class.
    """
    from openai import OpenAI
    if get_provider() == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def get_langchain_llm(temperature: float = 0.3):
    """LangChain LLM object for KatzBot RAG chain."""
    provider = get_provider()
    model    = get_model_name()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )


def get_langchain_embeddings():
    """
    Embeddings for KatzBot vector store.
    - OpenAI key present  → text-embedding-3-small (best, costs ~$0.001)
    - No OpenAI key       → all-MiniLM-L6-v2 via HuggingFace (free, ~80 MB)
    Groq does NOT provide an embeddings API.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key.startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )
    # Free local fallback
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def get_provider_display() -> dict:
    provider = get_provider()
    model    = get_model_name()
    key_ok   = is_api_key_set()
    if provider == "groq":
        return {
            "name": "Groq", "model": model, "key_ok": key_ok,
            "color": "#F56A00", "note": "Free tier",
            "key_var": "GROQ_API_KEY", "signup": "https://console.groq.com",
        }
    return {
        "name": "OpenAI", "model": model, "key_ok": key_ok,
        "color": "#10A37F", "note": "~$0.025/run",
        "key_var": "OPENAI_API_KEY", "signup": "https://platform.openai.com/api-keys",
    }
