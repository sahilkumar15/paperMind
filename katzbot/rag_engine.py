"""
katzbot/rag_engine.py
======================
KatzBot RAG engine.

This version fixes three practical issues:
1. Backward compatibility for code that still expects a pickle path.
2. Faster reloads by using FAISS on disk as the primary runtime artifact.
3. Better coverage by explicitly indexing faculty and event documents in
   addition to crawled pages and notebook-derived QA pairs.
"""

from __future__ import annotations

from typing import Optional

from katzbot.chain import build_chain, get_llm
from katzbot.crawler import crawl_katz
from katzbot.events_fetcher import get_events_documents
from katzbot.faculty import get_faculty_documents, match_faculty
from katzbot.indexer import (
    FAISS_DIR,
    DOCS_PICKLE,
    INDEX_META_PICKLE,
    build_faiss_index,
    get_retriever,
    load_index_metadata,
)


class KatzRAGEngine:
    def __init__(self):
        self._db = None
        self._retriever = None
        self._chain = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _wire_runtime(self, db) -> None:
        self._db = db
        self._retriever = get_retriever(db, search_type="mmr")
        self._chain = build_chain(self._retriever, get_llm())
        self._ready = True

    def _try_load_disk(self) -> bool:
        if not FAISS_DIR.exists():
            return False
        try:
            from llm_config import get_langchain_embeddings
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError:
                from langchain.vectorstores import FAISS

            embeddings = get_langchain_embeddings()
            db = FAISS.load_local(
                str(FAISS_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            n = int(db.index.ntotal)
            if n <= 20:
                print(f"[KatzBot] Index too small ({n} vectors) — rebuilding")
                return False

            self._wire_runtime(db)
            print(f"[KatzBot] ✓ Loaded FAISS index ({n:,} vectors) from disk")

            meta = load_index_metadata()
            if meta:
                print(
                    f"[KatzBot] Sources: {meta.get('source_doc_count', 0)} docs | "
                    f"Chunks: {meta.get('chunk_count', 0)} | "
                    f"PKL: {DOCS_PICKLE}"
                )
            return True
        except Exception as e:
            print(f"[KatzBot] Disk load failed: {e}")
            return False

    def _ensure_loaded(self):
        if self._ready:
            return
        if self._try_load_disk():
            return
        print("[KatzBot] No usable index found — building automatically…")
        self.build(force_refresh=False)

    def build(self, force_refresh: bool = False) -> dict:
        print("\n[KatzBot] ══════ Building Index ══════")

        if not force_refresh and self._try_load_disk():
            meta = load_index_metadata()
            return {
                "web_pages": 0,
                "faculty_docs": meta.get("faculty_docs", 0),
                "events_docs": meta.get("events_docs", 0),
                "total_docs": meta.get("source_doc_count", 0),
                "index_vectors": self._db.index.ntotal,
                "index_dir": str(FAISS_DIR),
                "pkl_path": str(DOCS_PICKLE),
                "meta_path": str(INDEX_META_PICKLE),
                "source": "disk_cache",
            }

        web_docs = crawl_katz(force_refresh=force_refresh)
        faculty_docs = get_faculty_documents()
        event_docs = get_events_documents()
        all_docs = web_docs + faculty_docs + event_docs

        print(
            f"[KatzBot] {len(web_docs)} web + {len(faculty_docs)} faculty + "
            f"{len(event_docs)} events = {len(all_docs)} total docs"
        )

        self._db = build_faiss_index(all_docs, force_rebuild=force_refresh)
        self._wire_runtime(self._db)

        meta = load_index_metadata()
        meta.update(
            {
                "web_pages": len(web_docs),
                "faculty_docs": len(faculty_docs),
                "events_docs": len(event_docs),
                "total_docs": len(all_docs),
            }
        )
        # write merged metadata back through indexer pickle helper would be ideal,
        # but keeping this file self-contained avoids another public helper.
        import pickle
        with open(INDEX_META_PICKLE, "wb") as f:
            pickle.dump(meta, f)

        stats = {
            "web_pages": len(web_docs),
            "faculty_docs": len(faculty_docs),
            "events_docs": len(event_docs),
            "total_docs": len(all_docs),
            "index_vectors": self._db.index.ntotal,
            "index_dir": str(FAISS_DIR),
            "pkl_path": str(DOCS_PICKLE),
            "meta_path": str(INDEX_META_PICKLE),
            "source": "fresh_build",
        }
        print(f"[KatzBot] ══════ Ready: {stats['index_vectors']:,} vectors ══════\n")
        return stats

    def ask(self, question: str, history: list = None, extra_context: str = "") -> dict:
        self._ensure_loaded()
        q = f"{question}\n[Extra context: {extra_context[:300]}]" if extra_context else question
        try:
            result = self._chain({"question": q, "history": history or []})
            answer = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"Error: {e}"
            sources = []

        return {
            "answer": answer,
            "sources": [s for s in sources if s],
            "faculty_matches": match_faculty(question),
        }

    def get_index_info(self) -> dict:
        if not self._ready:
            return {"status": "not_built"}
        meta = load_index_metadata()
        return {
            "status": "ready",
            "vectors": self._db.index.ntotal,
            "index_dir": str(FAISS_DIR),
            "pkl_path": str(DOCS_PICKLE),
            "meta": meta,
        }


_engine: Optional[KatzRAGEngine] = None


def get_engine() -> KatzRAGEngine:
    global _engine
    if _engine is None:
        _engine = KatzRAGEngine()
    return _engine
