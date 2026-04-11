"""
katzbot/rag_engine.py
======================
FIX: _ensure_loaded() now auto-builds instead of raising.
     No more "Call engine.build() first" error on first chat message.
"""

import os
from pathlib import Path
from typing import Optional

from katzbot.crawler import crawl_katz
from katzbot.indexer import build_faiss_index, get_retriever, FAISS_DIR
from katzbot.chain   import build_conversational_chain, get_llm
from katzbot.faculty import KATZ_FACULTY, match_faculty, get_faculty_documents

CHROMA_DIR = Path(__file__).parent / "chroma_index"
CACHE_FILE = Path(__file__).parent / "pages_cache.json"


class KatzRAGEngine:
    def __init__(self):
        self._db        = None
        self._retriever = None
        self._chain     = None
        self._ready     = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _try_load_existing_faiss(self) -> bool:
        """Load pre-built FAISS index from disk without crawling."""
        if not FAISS_DIR.exists():
            return False
        try:
            from llm_config import get_langchain_embeddings
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError:
                from langchain.vectorstores import FAISS

            emb = get_langchain_embeddings()
            db  = FAISS.load_local(
                str(FAISS_DIR),
                emb,
                allow_dangerous_deserialization=True,
            )
            n = db.index.ntotal
            if n > 20:
                print(f"[KatzBot] ✓ Loaded FAISS index ({n} vectors) from disk")
                self._db        = db
                self._retriever = get_retriever(db, search_type="mmr")
                llm             = get_llm()
                self._chain     = build_conversational_chain(self._retriever, llm)
                self._ready     = True
                return True
            print(f"[KatzBot] Index too small ({n} vectors) — will rebuild")
        except Exception as e:
            print(f"[KatzBot] Could not load FAISS index: {e}")
        return False

    def _ensure_loaded(self):
        """
        FIX: Auto-build the index if not ready instead of raising.
        Called automatically on every ask() so users never see the
        'Call engine.build() first' error.
        """
        if self._ready:
            return
        # Try loading existing persisted index first (fast path)
        if self._try_load_existing_faiss():
            return
        # Nothing on disk — build from scratch
        print("[KatzBot] No index found — building automatically...")
        self.build(force_refresh=False)

    def build(self, force_refresh: bool = False) -> dict:
        """
        Full index build pipeline:
          1. Crawl Katz website (cached 7 days)
          2. Add faculty structured documents
          3. Build / load FAISS index
          4. Wire up conversational RAG chain
        """
        print("\n[KatzBot] ══════ Building KatzBot Index ══════")

        # Step 1 — try loading existing index unless forced
        if not force_refresh and self._try_load_existing_faiss():
            n = self._db.index.ntotal
            return {
                "web_pages":     0,
                "faculty_docs":  0,
                "total_docs":    0,
                "index_vectors": n,
                "index_dir":     str(FAISS_DIR),
                "source":        "cache",
            }

        # Step 2 — crawl
        web_docs     = crawl_katz(force_refresh=force_refresh)
        faculty_docs = get_faculty_documents()
        all_docs     = web_docs + faculty_docs
        print(f"[KatzBot] {len(web_docs)} web + {len(faculty_docs)} faculty = {len(all_docs)} docs")

        # Step 3 — build FAISS
        self._db = build_faiss_index(all_docs, force_rebuild=force_refresh)

        # Step 4 — retriever + chain
        self._retriever = get_retriever(self._db, search_type="mmr")
        llm             = get_llm()
        self._chain     = build_conversational_chain(self._retriever, llm)
        self._ready     = True

        stats = {
            "web_pages":     len(web_docs),
            "faculty_docs":  len(faculty_docs),
            "total_docs":    len(all_docs),
            "index_vectors": self._db.index.ntotal,
            "index_dir":     str(FAISS_DIR),
            "source":        "fresh",
        }
        print(f"[KatzBot] ══════ Ready: {stats['index_vectors']} vectors ══════\n")
        return stats

    def ask(self, question: str, history: list = None, extra_context: str = "") -> dict:
        """
        Answer a question. Auto-builds index on first call if needed.
        """
        self._ensure_loaded()   # ← auto-build instead of raise

        q = (f"{question}\n[Context: {extra_context[:200]}]"
             if extra_context else question)

        try:
            result  = self._chain({"question": q, "history": history or []})
            answer  = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])
        except Exception as e:
            answer  = f"Error: {e}"
            sources = []

        return {
            "answer":          answer,
            "sources":         [s for s in sources if s],
            "faculty_matches": match_faculty(question),
        }

    def similarity_search(self, query: str, k: int = 5) -> list:
        """Direct vector search — useful for debugging."""
        self._ensure_loaded()
        return self._db.similarity_search(query, k=k)


# ── Singleton ──────────────────────────────────────────────────
_engine: Optional[KatzRAGEngine] = None


def get_engine() -> KatzRAGEngine:
    global _engine
    if _engine is None:
        _engine = KatzRAGEngine()
    return _engine