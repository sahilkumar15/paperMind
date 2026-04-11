"""
llm_config.py
==============
GROQ TPM LIMITS (on_demand / free tier, April 2026):
  llama-3.1-8b-instant          250,000 TPM  ← DEFAULT (safe)
  openai/gpt-oss-120b           250,000 TPM  ← safe
  openai/gpt-oss-20b            250,000 TPM  ← safe
  meta-llama/llama-4-scout-*    300,000 TPM  ← safe
  llama-3.3-70b-versatile        12,000 TPM  ← hits limit with 6 agents
  qwen/qwen3-32b                  6,000 TPM  ← always fails with agents
"""

import os
from dotenv import load_dotenv

load_dotenv()

# TPM reference for sidebar display
GROQ_MODEL_TPM = {
    "llama-3.1-8b-instant":                      250_000,
    "llama-3.3-70b-versatile":                    12_000,
    "openai/gpt-oss-120b":                       250_000,
    "openai/gpt-oss-20b":                        250_000,
    "meta-llama/llama-4-scout-17b-16e-instruct": 300_000,
    "qwen/qwen3-32b":                              6_000,
}

GROQ_MODELS = {
    "llama-3.1-8b-instant":                      "✅ RECOMMENDED · 250K TPM · always works",
    "openai/gpt-oss-120b":                       "🔥 GPT-class 120B · 250K TPM · safe",
    "openai/gpt-oss-20b":                        "🚀 GPT-class fast · 250K TPM · safe",
    "meta-llama/llama-4-scout-17b-16e-instruct": "🦙 Llama 4 Scout · 300K TPM · safe",
    "llama-3.3-70b-versatile":                   "⚠️ Best quality · 12K TPM · may rate-limit",
    "qwen/qwen3-32b":                            "🚫 6K TPM · will fail · avoid",
}

OPENAI_MODELS = {
    "gpt-4o-mini":   "✅ Recommended · cheap · fast",
    "gpt-4o":        "⭐ Best quality · higher cost",
    "gpt-3.5-turbo": "Budget option",
}

# Models that are too low TPM for the 6-agent pipeline
GROQ_UNSAFE_MODELS = {"qwen/qwen3-32b", "llama-3.3-70b-versatile"}
GROQ_SAFE_FALLBACK  = "llama-3.1-8b-instant"


def get_provider() -> str:
    return "groq" if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" else "openai"


def get_model_name() -> str:
    env = os.getenv("MODEL_NAME", "").strip()
    if env:
        return env
    return GROQ_SAFE_FALLBACK if get_provider() == "groq" else "gpt-4o-mini"


def get_api_key() -> str:
    return (
        os.getenv("GROQ_API_KEY", "")
        if get_provider() == "groq"
        else os.getenv("OPENAI_API_KEY", "")
    )


def is_api_key_set() -> bool:
    key = get_api_key()
    if get_provider() == "groq":
        return bool(key and key.startswith("gsk_"))
    return bool(key and key.startswith("sk-"))


def get_crewai_llm_string() -> str:
    """
    LiteLLM routing string.
    Always prepend 'groq/' for Groq models (regardless of sub-prefix).
    """
    provider = get_provider()
    model    = get_model_name()

    if provider == "groq":
        return f"groq/{model}" if not model.startswith("groq/") else model
    return f"openai/{model}" if not model.startswith("openai/") else model


def get_safe_crewai_llm_string() -> str:
    """
    Same as get_crewai_llm_string() but auto-overrides low-TPM Groq models.
    Used by agents to guarantee the pipeline completes.
    """
    provider = get_provider()
    model    = get_model_name()

    if provider == "groq" and model in GROQ_UNSAFE_MODELS:
        print(
            f"[llm_config] ⚠ '{model}' has insufficient TPM for 6 agents. "
            f"Auto-switching to '{GROQ_SAFE_FALLBACK}' (250K TPM)."
        )
        return f"groq/{GROQ_SAFE_FALLBACK}"

    return get_crewai_llm_string()


def get_openai_client():
    """OpenAI-compatible client for direct API calls (PaperBot, drafts)."""
    from openai import OpenAI
    if get_provider() == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def get_langchain_llm(temperature: float = 0.3):
    """LangChain LLM for KatzBot RAG chain."""
    provider = get_provider()
    # Strip any groq/ prefix — langchain_groq wants bare model name
    model = get_model_name().replace("groq/", "")

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
    OpenAI key present → text-embedding-3-small (best quality).
    No OpenAI key      → all-MiniLM-L6-v2 via HuggingFace (free, no API needed).
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key.startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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