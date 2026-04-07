"""
katzbot/rag_engine.py

KatzBot — RAG chatbot grounded in the Katz School website (yu.edu/katz).
Works with Groq (free) or OpenAI — controlled by LLM_PROVIDER in .env.

Uses modern LangChain LCEL instead of deprecated RetrievalQA / langchain.chains.
Uses langchain-chroma instead of deprecated langchain_community.vectorstores.Chroma.

Pipeline:
  1. Crawl YU sitemap → filter /katz URLs (from your DataManager code)
  2. Load pages with UnstructuredURLLoader
  3. Chunk → embed → store in ChromaDB (persisted to disk)
  4. Answer with LCEL retrieval chain (Groq or OpenAI)
"""

import os
import pickle
import logging
import requests
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)

BASE_DIR  = Path(__file__).parent
INDEX_DIR = BASE_DIR / "chroma_index"
DATA_FILE = BASE_DIR / "katz_pages.pkl"

SITEMAP_URL = "https://www.yu.edu/sitemap.xml"
KATZ_PREFIX = "https://www.yu.edu/katz"

# ── Official Katz Faculty (from yu.edu/katz/faculty) ─────────
KATZ_FACULTY = [
    {
        "name": "Dr. Honggang Wang",
        "title": "Professor & Founding Chair, IEEE Fellow",
        "dept": "Computer Science & Engineering",
        "expertise": ["AI", "IoT", "healthcare AI", "autonomous vehicles",
                      "wireless networks", "cybersecurity", "multi-agent systems"],
        "email": "Honggang.wang@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty/honggang-wang",
        "note": "$5M+ NSF/NIH grants. Best for: AI systems, IoT, health AI.",
    },
    {
        "name": "Dr. Iddo Drori",
        "title": "Associate Professor",
        "dept": "M.S. in Artificial Intelligence",
        "expertise": ["AGI", "agentic AI", "LLMs", "deep learning",
                      "computer vision", "education AI", "NLP"],
        "email": "iddo.drori@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "Runs superintelligence lab. 80+ papers, 7k+ citations. Best for: LLMs, agentic AI.",
    },
    {
        "name": "Dr. Youshan Zhang",
        "title": "Assistant Professor of AI & Computer Science",
        "dept": "M.S. in Artificial Intelligence",
        "expertise": ["deep learning", "neural networks", "computer vision",
                      "medical imaging", "transfer learning", "self-driving"],
        "email": "youshan.zhang@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "NSF-funded. Best for: deep learning, medical AI, object detection.",
    },
    {
        "name": "Dr. Yucheng Xie",
        "title": "Assistant Professor of Computer Science",
        "dept": "Computer Science & Engineering",
        "expertise": ["machine learning", "healthcare AI", "IoT security",
                      "privacy-preserving AI", "backdoor attacks", "cybersecurity"],
        "email": "yucheng.xie@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "Best for: AI security, smart healthcare, adversarial ML.",
    },
    {
        "name": "Dr. Ming Ma",
        "title": "Assistant Professor of Computer Science",
        "dept": "Computer Science & Engineering",
        "expertise": ["AI", "geometric modeling", "medical imaging"],
        "email": "ming.ma@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "40+ publications. PhD Stony Brook, postdoc Stanford.",
    },
    {
        "name": "Prof. Jiang Zhou",
        "title": "Professor of Artificial Intelligence",
        "dept": "M.S. in Artificial Intelligence",
        "expertise": ["artificial intelligence", "machine learning", "applied AI"],
        "email": "jiang.zhou@yu.edu",
        "profile": "https://www.yu.edu/katz/ai",
        "note": "AI course instructor. Best for: AI coursework, ML projects.",
    },
    {
        "name": "Adam Faulkner",
        "title": "Industry Professor — Generative AI",
        "dept": "M.S. in Artificial Intelligence",
        "expertise": ["LLMs", "NLP", "generative AI", "multi-agent systems",
                      "large language models", "agentic AI"],
        "email": "adam.faulkner@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "Senior Manager GenAI at Capital One. Best for: LLMs, GenAI industry projects.",
    },
    {
        "name": "Dr. David Li",
        "title": "Director, M.S. in Data Analytics & Visualization",
        "dept": "Data Analytics & Visualization",
        "expertise": ["data science", "probabilistic models", "statistics",
                      "visualization", "fuzzy ranking"],
        "email": "david.li@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "Best for: data analytics, statistical modeling, visualization.",
    },
    {
        "name": "Dr. Sivan Tehila",
        "title": "Director, M.S. in Cybersecurity",
        "dept": "Cybersecurity",
        "expertise": ["cybersecurity", "information security", "security systems"],
        "email": "sivan.tehila@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "Best for: cybersecurity research and security projects.",
    },
    {
        "name": "Dr. Marian Gidea",
        "title": "Associate Dean for STEM Research",
        "dept": "Mathematical Sciences",
        "expertise": ["mathematics", "topology", "dynamical systems",
                      "interdisciplinary research"],
        "email": "marian.gidea@yu.edu",
        "profile": "https://www.yu.edu/katz/faculty",
        "note": "Best for: math foundations, interdisciplinary STEM.",
    },
]


def match_faculty(topic: str) -> list:
    """Match up to 3 faculty members to a research topic string."""
    topic_lower = topic.lower()
    scored = []
    for f in KATZ_FACULTY:
        score = sum(
            1 for kw in f["expertise"]
            if kw.lower() in topic_lower
            or any(w in kw.lower() for w in topic_lower.split())
        )
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:3]] or KATZ_FACULTY[:2]


# ── Sitemap crawler (adapted from your DataManager) ───────────

def fetch_katz_urls(max_pages: int = 60) -> list:
    """Crawl YU sitemap and return Katz-related URLs only."""
    try:
        resp = requests.get(SITEMAP_URL, timeout=15)
        soup = BeautifulSoup(resp.content, "xml")
        all_urls = [loc.text for loc in soup.find_all("loc")]
        katz_urls = [u for u in all_urls if KATZ_PREFIX in u]
    except Exception as e:
        print(f"[KatzBot] Sitemap fetch failed: {e}. Using fallback URLs.")
        katz_urls = []

    # Always include key pages
    extra = [
        "https://www.yu.edu/katz",
        "https://www.yu.edu/katz/faculty",
        "https://www.yu.edu/katz/ai",
        "https://www.yu.edu/katz/computer-science-engineering",
        "https://www.yu.edu/katz/computer-science",
        "https://www.yu.edu/katz/cybersecurity",
        "https://www.yu.edu/katz/research",
        "https://www.yu.edu/katz/staff",
        "https://www.yu.edu/katz/faculty/honggang-wang",
    ]
    for e in extra:
        if e not in katz_urls:
            katz_urls.append(e)

    return katz_urls[:max_pages]


def build_index(force_rebuild: bool = False):
    """
    Build or load the ChromaDB vector index.
    Uses modern langchain-chroma (not deprecated langchain_community.vectorstores.Chroma).
    """
    from langchain_community.document_loaders import UnstructuredURLLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma          # ← new, not deprecated
    from llm_config import get_langchain_embeddings

    embeddings = get_langchain_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        print("[KatzBot] Loading existing ChromaDB index...")
        return Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=embeddings,
        )

    print("[KatzBot] Building new index from Katz School website...")
    urls = fetch_katz_urls()
    print(f"[KatzBot] Crawling {len(urls)} URLs...")

    try:
        loader = UnstructuredURLLoader(urls=urls)
        docs   = loader.load()
        print(f"[KatzBot] Loaded {len(docs)} documents.")
        with open(DATA_FILE, "wb") as f:
            pickle.dump(docs, f)
    except Exception as e:
        print(f"[KatzBot] Loader error: {e}. Checking cache...")
        if DATA_FILE.exists():
            with open(DATA_FILE, "rb") as f:
                docs = pickle.load(f)
        else:
            raise RuntimeError(f"No cached data and URL loader failed: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks   = splitter.split_documents(docs)
    print(f"[KatzBot] Split into {len(chunks)} chunks.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    print("[KatzBot] Index built and saved to disk.")
    return vectorstore


def get_rag_chain(vectorstore):
    """
    Build RAG chain using modern LangChain LCEL.
    Replaces deprecated RetrievalQA / langchain.chains.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from llm_config import get_langchain_llm

    llm       = get_langchain_llm(temperature=0.3)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    prompt = ChatPromptTemplate.from_template("""You are KatzBot, the official AI assistant \
for Yeshiva University's Katz School of Science and Health. Answer questions about \
programs, faculty, research, admissions, and student life using the context below.

If the question is about a research topic, mention relevant faculty and their contact info.
If you cannot answer from the context, say: "I don't have that information — \
please check yu.edu/katz or email katz@yu.edu"

Context:
{context}

Question: {question}

Answer (be specific, helpful, mention relevant contacts when appropriate):""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Modern LCEL chain — no deprecated imports
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever   # return retriever too so we can get source docs


class KatzBotEngine:
    """Main KatzBot engine. Simple ask() interface over the RAG chain."""

    def __init__(self):
        self._vectorstore = None
        self._chain       = None
        self._retriever   = None

    def _ensure_loaded(self):
        if self._chain is None:
            self._vectorstore         = build_index()
            self._chain, self._retriever = get_rag_chain(self._vectorstore)

    def ask(self, question: str, extra_context: str = "") -> dict:
        self._ensure_loaded()

        full_q = question
        if extra_context:
            full_q = f"{question}\n\nAdditional context: {extra_context}"

        try:
            answer  = self._chain.invoke(full_q)
            # Fetch source docs separately for display
            src_docs = self._retriever.invoke(question)
            sources  = list({
                doc.metadata.get("source", "")
                for doc in src_docs
                if doc.metadata.get("source", "")
            })
        except Exception as e:
            answer  = f"KatzBot error: {e}"
            sources = []

        return {
            "answer":          answer,
            "sources":         sources,
            "faculty_matches": match_faculty(question),
        }

    def rebuild_index(self):
        self._vectorstore         = build_index(force_rebuild=True)
        self._chain, self._retriever = get_rag_chain(self._vectorstore)
        return "Index rebuilt successfully."


_engine: Optional[KatzBotEngine] = None

def get_engine() -> KatzBotEngine:
    global _engine
    if _engine is None:
        _engine = KatzBotEngine()
    return _engine
