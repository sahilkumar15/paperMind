"""
llm_config.py — Central LLM provider configuration for PaperMind

Patched so model/provider mismatches are visible and KatzBot can still use
free local embeddings while OpenAI/Groq remain optional for chat.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODELS = {
    "llama-3.3-70b-versatile":                   "⭐ Best quality · 300K TPM · recommended",
    "llama-3.1-8b-instant":                      "⚡ Fastest · 250K TPM · 14,400 req/day",
    "openai/gpt-oss-120b":                       "🔥 GPT-class 120B · 250K TPM",
    "openai/gpt-oss-20b":                        "🚀 GPT-class fast · 1000 t/s",
    "meta-llama/llama-4-scout-17b-16e-instruct": "🦙 Llama 4 Scout · 300K TPM",
    "qwen/qwen3-32b":                            "🧠 Qwen3 32B · strong reasoning",
}

OPENAI_MODELS = {
    "gpt-4o-mini": "Cheapest · recommended",
    "gpt-4o": "Best quality",
    "gpt-3.5-turbo": "Budget option",
}


def get_provider() -> str:
    raw = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if raw in {"groq", "openai"}:
        return raw
    return "groq"


def get_model_name() -> str:
    env = os.getenv("MODEL_NAME", "").strip()
    if env:
        return env
    return "llama-3.3-70b-versatile" if get_provider() == "groq" else "gpt-4o-mini"


def get_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "") if get_provider() == "groq" else os.getenv("OPENAI_API_KEY", "")


def is_api_key_set() -> bool:
    key = get_api_key().strip()
    if get_provider() == "groq":
        return key.startswith("gsk_")
    return key.startswith("sk-")


def validate_provider_model() -> tuple[bool, str]:
    provider = get_provider()
    model = get_model_name()
    if provider == "groq" and model not in GROQ_MODELS:
        return False, f"MODEL_NAME '{model}' is not a Groq model"
    if provider == "openai" and model not in OPENAI_MODELS:
        return False, f"MODEL_NAME '{model}' is not a supported OpenAI model"
    return True, f"provider={provider} model={model}"


def get_crewai_llm_string() -> str:
    provider = get_provider()
    model = get_model_name()
    if provider == "groq":
        return model if model.startswith("groq/") else f"groq/{model}"
    return model if model.startswith("openai/") else f"openai/{model}"


def get_openai_client():
    from openai import OpenAI
    if get_provider() == "groq":
        return OpenAI(api_key=os.getenv("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def get_langchain_llm(temperature: float = 0.3):
    provider = get_provider()
    model = get_model_name()

    ok, reason = validate_provider_model()
    print(f"[LLM] get_langchain_llm → provider={provider} model={model}")
    if not ok:
        raise ValueError(reason)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model.replace("groq/", ""),
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
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key.startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)

    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
