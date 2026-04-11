"""
katzbot/chain.py
=================
Conversational RAG chain. Works with Groq, OpenAI, or HuggingFace.
Uses a simple callable class pattern — no LangChain version issues.
"""

from typing import Optional

# ── Prompts (version-safe) ────────────────────────────────────
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

# ── Output parser ─────────────────────────────────────────────
try:
    from langchain_core.output_parsers import StrOutputParser
    _HAS_PARSER = True
except ImportError:
    _HAS_PARSER = False

# ── Runnables for LCEL ────────────────────────────────────────
try:
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    _HAS_LCEL = True
except ImportError:
    _HAS_LCEL = False


RAG_PROMPT = """\
You are KatzBot, the official AI assistant for the Yeshiva University \
Katz School of Science and Health.

Answer the question using ONLY the context below from the Katz School website.
- Be specific and helpful — include emails, URLs, deadlines when available
- If the answer is not in the context, say exactly:
  "I don't have that specific information. Please check yu.edu/katz or email katz@yu.edu"
- Keep answers to 2-5 sentences

Context:
{context}

Conversation so far:
{history}

Question: {question}

Answer:"""


def _format_docs(docs: list) -> str:
    """Turn retrieved docs into a context string."""
    if not docs:
        return "No relevant information found."
    parts = []
    for doc in docs:
        src   = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "") or src.split("/")[-1]
        parts.append(f"[{title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: list) -> str:
    """Format chat history to string."""
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-6:]:   # last 3 turns only
        role    = "User" if msg.get("role") == "user" else "KatzBot"
        content = str(msg.get("content", ""))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def get_llm(temperature: float = 0.2):
    """
    Get LLM — Groq (free) preferred, HuggingFace pipeline as fallback.
    Mirrors the notebook's HuggingFace approach but uses API instead of local model.
    """
    try:
        from llm_config import get_langchain_llm
        llm = get_langchain_llm(temperature=temperature)
        print(f"[Chain] LLM: {type(llm).__name__}")
        return llm
    except Exception as e:
        print(f"[Chain] llm_config failed ({e}), trying HuggingFace pipeline...")
        return _get_hf_llm()


def _get_hf_llm():
    """HuggingFace pipeline fallback — same spirit as notebook's Zephyr-7B but CPU-friendly."""
    try:
        from transformers import pipeline as hf_pipeline
        try:
            from langchain_community.llms import HuggingFacePipeline
        except ImportError:
            from langchain.llms import HuggingFacePipeline

        print("[Chain] Loading HuggingFace phi-2 (CPU, ~5GB)...")
        pipe = hf_pipeline(
            "text-generation",
            model="microsoft/phi-2",
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        raise RuntimeError(
            f"No LLM available. Set GROQ_API_KEY in .env or install transformers. Error: {e}"
        )


class ConversationalRAGChain:
    """
    Simple callable RAG chain with conversation memory.
    Avoids all LangChain version compatibility issues by being explicit.
    
    Call like: result = chain({"question": "...", "history": [...]})
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm       = llm
        self.prompt    = PromptTemplate(
            input_variables=["context", "history", "question"],
            template=RAG_PROMPT,
        )

    def __call__(self, inputs: dict) -> dict:
        question = inputs.get("question", "")
        history  = inputs.get("history", [])

        # Step 1: Retrieve relevant documents
        try:
            docs = self.retriever.invoke(question)
        except Exception:
            # Older LangChain uses .get_relevant_documents()
            try:
                docs = self.retriever.get_relevant_documents(question)
            except Exception as e:
                docs = []
                print(f"[Chain] Retrieval failed: {e}")

        # Step 2: Format context and history
        context      = _format_docs(docs)
        history_str  = _format_history(history)

        # Step 3: Fill prompt
        filled = self.prompt.format(
            context=context,
            history=history_str,
            question=question,
        )

        # Step 4: Call LLM
        try:
            response = self.llm.invoke(filled)
            # Handle both string and AIMessage responses
            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        return {
            "answer":  answer.strip(),
            "sources": [d.metadata.get("source", "") for d in docs],
            "docs":    docs,
        }


def build_conversational_chain(retriever, llm) -> ConversationalRAGChain:
    """Factory function — returns a ready-to-call chain."""
    chain = ConversationalRAGChain(retriever, llm)
    print("[Chain] Conversational RAG chain ready")
    return chain