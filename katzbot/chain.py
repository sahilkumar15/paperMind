"""
katzbot/chain.py
=================
Conversational RAG chain — callable class pattern.
Works with any LangChain version (0.1 through 0.3+).

This version is fully offline-safe:
- Uses Groq/OpenAI only when provider/model/key are valid
- Uses local HuggingFace only if available locally
- Falls back to a deterministic extractive answerer
- Filters out noisy QA-pair docs for important question types
"""

import os
import re
from collections import Counter

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate


RAG_PROMPT = """\
You are KatzBot, the official AI assistant for the Yeshiva University \
Katz School of Science and Health.

Answer the question using ONLY the context below (from the Katz School \
website, faculty database, events, and QA knowledge base).

Guidelines:
- Be specific and helpful. Include faculty emails, program URLs, deadlines.
- For faculty questions: always include name, title, email, and expertise.
- If the answer is NOT in the context, say:
  "I don't have that specific information. Please check yu.edu/katz \
  or email katz@yu.edu"
- Never invent information not present in the context.
- Keep answers concise but informative.

Context:
{context}

Conversation history:
{history}

Question: {question}

Answer:"""


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at",
    "is", "are", "was", "were", "be", "been", "being", "what", "who", "how", "when",
    "where", "which", "that", "this", "it", "as", "by", "from", "about", "into",
    "does", "do", "did", "can", "i", "me", "my", "we", "you", "your", "our",
    "school", "katz", "yeshiva", "university"
}


def _format_docs(docs: list) -> str:
    if not docs:
        return "No relevant information found."
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "") or src.split("/")[-1]
        parts.append(f"[{title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-6:]:
        role = "User" if msg.get("role") == "user" else "KatzBot"
        content = str(msg.get("content", ""))[:400]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _want_local_only() -> bool:
    return os.getenv("KATZBOT_LOCAL_ONLY", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _provider_config_is_valid() -> tuple[bool, str]:
    try:
        from llm_config import (
            get_provider,
            get_model_name,
            is_api_key_set,
            GROQ_MODELS,
            OPENAI_MODELS,
        )

        provider = get_provider()
        model = get_model_name()

        if provider == "groq":
            if model not in GROQ_MODELS:
                return False, (
                    f"MODEL_NAME='{model}' is not a valid Groq model. "
                    f"Pick one of: {', '.join(GROQ_MODELS.keys())}"
                )
        elif provider == "openai":
            if model not in OPENAI_MODELS:
                return False, (
                    f"MODEL_NAME='{model}' is not a supported OpenAI model. "
                    f"Pick one of: {', '.join(OPENAI_MODELS.keys())}"
                )
        else:
            return False, f"Unsupported provider '{provider}'"

        if not is_api_key_set():
            return False, f"Missing/invalid API key for provider '{provider}'"

        return True, f"provider={provider} model={model}"
    except Exception as e:
        return False, f"llm_config validation failed: {e}"


def _question_intent(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["event", "events", "upcoming", "info session", "workshop", "symposium"]):
        return "events"
    if any(x in q for x in ["who is", "prof", "professor", "faculty", "contact"]):
        return "faculty"
    if any(x in q for x in ["tuition", "financial aid", "scholarship", "cost"]):
        return "money"
    if any(x in q for x in ["apply", "admission", "deadline", "requirements"]):
        return "admissions"
    if any(x in q for x in ["curriculum", "course", "credits", "program", "programs"]):
        return "programs"
    return "general"


def _filter_docs_for_question(question: str, docs: list) -> list:
    if not docs:
        return []

    intent = _question_intent(question)

    if intent in {"faculty", "events", "money", "admissions", "programs"}:
        official = [d for d in docs if d.metadata.get("type") != "qa_pair"]
        if official:
            return official[:4]

    return docs[:4]


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]*", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def _clean_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" -•\n\t")


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\r", " ")
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    seen = set()
    for s in raw:
        s = _clean_sentence(s)
        if len(s) < 25:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_faculty_answer(context: str, question: str) -> str | None:
    q = question.lower()
    if not any(x in q for x in ["prof", "professor", "faculty", "contact", "who is"]):
        return None

    blocks = re.split(r"\n\n---\n\n", context)
    candidates = []

    for block in blocks:
        if "FACULTY PROFILE" not in block and "honggang wang" not in block.lower():
            continue

        score = 0
        for tok in _tokenize(question):
            if tok in block.lower():
                score += 2

        if score <= 0:
            continue

        name = re.search(r"Name:\s*(.+)", block)
        title = re.search(r"Title:\s*(.+)", block)
        email = re.search(r"Email:\s*(.+)", block)
        exp = re.search(r"Research Expertise:\s*(.+)", block)
        note = re.search(r"About:\s*(.+)", block)

        parts = []
        if name:
            parts.append(name.group(1).strip())
        if title:
            parts.append(title.group(1).strip())
        if email:
            parts.append(f"Email: {email.group(1).strip()}")
        if exp:
            exp_items = [e.strip() for e in exp.group(1).split(",")[:5] if e.strip()]
            if exp_items:
                parts.append("Expertise: " + ", ".join(exp_items))
        if note:
            parts.append(note.group(1).strip())

        if parts:
            candidates.append((score, " ".join(parts)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _best_sentences(context: str, question: str, k: int = 4) -> list[str]:
    q_tokens = _tokenize(question)
    q_counts = Counter(q_tokens)
    sents = _split_sentences(context)

    scored = []
    for sent in sents:
        s_low = sent.lower()
        s_tokens = _tokenize(sent)
        if not s_tokens:
            continue

        overlap = sum(q_counts[t] for t in s_tokens if t in q_counts)
        bonus = 0
        if "http" in s_low or "yu.edu" in s_low:
            bonus += 1
        if any(x in s_low for x in [
            "email:", "deadline", "tuition", "credits", "curriculum",
            "contact", "apply", "financial aid"
        ]):
            bonus += 1

        score = overlap + bonus
        if score > 0:
            scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    seen = set()
    for _, sent in scored:
        norm = sent.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(sent)
        if len(out) >= k:
            break
    return out


class ExtractiveFallbackLLM:
    def __init__(self):
        self._katzbot_local = True
        self._katzbot_extractive = True

    def invoke(self, filled_prompt: str):
        context_match = re.search(
            r"Context:\n(.*?)\n\nConversation history:\n",
            filled_prompt,
            flags=re.DOTALL,
        )
        question_match = re.search(
            r"Question:\s*(.*?)\n\nAnswer:",
            filled_prompt,
            flags=re.DOTALL,
        )

        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else ""

        if not context or context == "No relevant information found.":
            return "I don't have that specific information. Please check yu.edu/katz or email katz@yu.edu"

        faculty = _extract_faculty_answer(context, question)
        if faculty:
            return faculty

        top = _best_sentences(context, question, k=4)
        if not top:
            return "I don't have that specific information. Please check yu.edu/katz or email katz@yu.edu"

        answer = " ".join(top)
        answer = re.sub(r"\s+", " ", answer).strip()
        answer = answer.replace("Q: ", "").replace("A: ", "")
        return answer[:1200]


def get_llm(temperature: float = 0.2):
    if _want_local_only():
        print("[Chain] KATZBOT_LOCAL_ONLY=1 → local/offline mode enabled")
        try:
            return _get_hf_llm()
        except Exception as e:
            print(f"[Chain] Local HuggingFace unavailable ({e}) → using extractive fallback")
            return ExtractiveFallbackLLM()

    ok, reason = _provider_config_is_valid()
    if ok:
        try:
            from llm_config import get_langchain_llm
            llm = get_langchain_llm(temperature=temperature)
            print(f"[Chain] LLM: {type(llm).__name__}")
            return llm
        except Exception as e:
            print(f"[Chain] API LLM setup failed ({e})")
    else:
        print(f"[Chain] API LLM disabled ({reason})")

    try:
        return _get_hf_llm()
    except Exception as e:
        print(f"[Chain] Local HuggingFace unavailable ({e}) → using extractive fallback")
        return ExtractiveFallbackLLM()


def _get_hf_llm():
    from transformers import pipeline as hf_pipeline
    try:
        from langchain_community.llms import HuggingFacePipeline
    except ImportError:
        from langchain.llms import HuggingFacePipeline

    model_name = os.getenv("KATZBOT_LOCAL_MODEL", "google/flan-t5-base")
    local_files_only = os.getenv("KATZBOT_HF_LOCAL_FILES_ONLY", "1").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    print(
        f"[Chain] Loading local HuggingFace model: {model_name} "
        f"(local_files_only={local_files_only})"
    )

    task = "text2text-generation" if "t5" in model_name.lower() else "text-generation"
    pipe = hf_pipeline(
        task,
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        local_files_only=local_files_only,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    setattr(llm, "_katzbot_local", True)
    return llm


class KatzRAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "history", "question"],
            template=RAG_PROMPT,
        )

    def _invoke_with_local_retry(self, filled: str):
        try:
            response = self.llm.invoke(filled)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            if getattr(self.llm, "_katzbot_extractive", False):
                raise
            print(f"[Chain] Current LLM failed ({e}) → retrying with extractive fallback")
            self.llm = ExtractiveFallbackLLM()
            response = self.llm.invoke(filled)
            return response.content if hasattr(response, "content") else str(response)

    def __call__(self, inputs: dict) -> dict:
        question = inputs.get("question", "")
        history = inputs.get("history", [])

        try:
            docs = self.retriever.invoke(question)
        except AttributeError:
            try:
                docs = self.retriever.get_relevant_documents(question)
            except Exception as e:
                docs = []
                print(f"[Chain] Retrieval error: {e}")

        docs = _filter_docs_for_question(question, docs)

        context = _format_docs(docs)
        history_str = _format_history(history)
        filled = self.prompt.format(
            context=context,
            history=history_str,
            question=question,
        )

        try:
            answer = self._invoke_with_local_retry(filled)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        return {
            "answer": str(answer).strip(),
            "sources": [d.metadata.get("source", "") for d in docs],
            "docs": docs,
        }


def build_chain(retriever, llm) -> KatzRAGChain:
    chain = KatzRAGChain(retriever, llm)
    print("[Chain] RAG chain ready")
    return chain