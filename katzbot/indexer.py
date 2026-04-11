"""
katzbot/indexer.py
===================
Vector store builder — upgraded from the notebook's FAISS approach.

Original notebook:
  - FAISS.from_documents(docs, embeddings)
  - Retriever: similarity search, k=3
  - Embeddings: all-MiniLM-l6-v2 on CUDA

Modern upgrades:
  - FAISS persisted to disk (original kept in-memory only)
  - Chroma as alternative (richer metadata filtering)
  - Chunk size tuned from notebook's 1000→800 (better recall)
  - Chunk overlap: 20→100 (better context preservation)
  - k=3→6 (notebook used 3, more context improves accuracy)
  - Saved as FAISS index files (same format as args.retriever_index)
  - Pickle-compatible retriever save (mirrors args.retrieve_data)

FAISS is kept as PRIMARY because:
  1. The original notebook used it and achieved ROUGE-1 F1=0.367
  2. No external service needed (fully local)
  3. Faster similarity search than Chroma for < 50K chunks
"""

import os
import pickle
from pathlib import Path
from typing import Optional

# ── Text splitter ─────────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── FAISS (primary, same as notebook) ─────────────────────────
try:
    from langchain_community.vectorstores import FAISS
    print("[Indexer] Using langchain_community FAISS")
except ImportError:
    from langchain.vectorstores import FAISS
    print("[Indexer] Using legacy langchain FAISS")

# ── Chroma (alternative, if FAISS unavailable) ────────────────
def _get_chroma():
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        return Chroma

# ── Constants (mirror original notebook's args) ───────────────
INDEX_DIR        = Path(__file__).parent / "faiss_index"
FAISS_DIR        = INDEX_DIR   # alias used by rag_engine.py
RETRIEVER_PICKLE = Path(__file__).parent / "retriever_store.pkl"
CHROMA_DIR       = Path(__file__).parent / "chroma_index"

# Chunk settings (tuned from notebook's 1000/20)
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
RETRIEVER_K   = 6    # notebook used 3; 6 gives better context


def _get_embeddings():
    """
    Get embeddings model.
    Priority:
      1. OpenAI text-embedding-3-small (if key available) — best quality
      2. all-MiniLM-L6-v2 via HuggingFace — same as original notebook, free
    """
    from llm_config import get_langchain_embeddings
    return get_langchain_embeddings()


def split_documents(documents: list) -> list:
    """
    Split documents into chunks.
    Uses RecursiveCharacterTextSplitter — same as notebook but tuned.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Indexer] Split {len(documents)} docs → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_faiss_index(documents: list, force_rebuild: bool = False) -> "FAISS":
    """
    Build or load FAISS vector store.
    
    Mirrors the notebook's FAISS.from_documents() but adds persistence.
    Original: db = FAISS.from_documents(docs, embeddings)
    Now:      saved to faiss_index/ and reloaded on next run.
    """
    embeddings = _get_embeddings()

    # ── Try loading existing index ─────────────────────────────
    if not force_rebuild and INDEX_DIR.exists():
        try:
            db = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            n = db.index.ntotal
            if n > 0:
                print(f"[Indexer] Loaded FAISS index ({n} vectors) from {INDEX_DIR}")
                return db
        except Exception as e:
            print(f"[Indexer] Could not load existing FAISS index: {e}")

    # ── Build fresh ────────────────────────────────────────────
    print(f"[Indexer] Building FAISS index from {len(documents)} documents...")
    chunks = split_documents(documents)

    print(f"[Indexer] Creating embeddings for {len(chunks)} chunks...")
    db = FAISS.from_documents(chunks, embeddings)

    # ── Persist (notebook didn't persist — we add this) ───────
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        db.save_local(str(INDEX_DIR))
        print(f"[Indexer] FAISS index saved → {INDEX_DIR}")
    except Exception as e:
        print(f"[Indexer] FAISS save failed: {e}")

    # ── Also save retriever as pickle (mirrors args.retrieve_data) ─
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
        with open(RETRIEVER_PICKLE, "wb") as f:
            pickle.dump(retriever, f)
        print(f"[Indexer] Retriever saved → {RETRIEVER_PICKLE}")
    except Exception as e:
        print(f"[Indexer] Retriever pickle failed: {e}")

    return db


def build_chroma_index(documents: list, force_rebuild: bool = False):
    """
    Alternative: Chroma vector store (richer metadata, persistent by default).
    Used as fallback if FAISS has issues.
    """
    Chroma = _get_chroma()
    embeddings = _get_embeddings()

    if not force_rebuild and CHROMA_DIR.exists():
        try:
            db = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
            )
            count = db._collection.count()
            if count > 0:
                print(f"[Indexer] Loaded Chroma index ({count} chunks)")
                return db
        except Exception as e:
            print(f"[Indexer] Chroma load failed: {e}")

    print("[Indexer] Building Chroma index...")
    chunks = split_documents(documents)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    if hasattr(db, "persist"):
        db.persist()
    print(f"[Indexer] Chroma index saved → {CHROMA_DIR}")
    return db


def get_retriever(db, search_type: str = "mmr") -> object:
    """
    Create retriever from vector store.
    
    search_type options:
      "similarity" — same as original notebook (cosine sim)
      "mmr"        — Maximum Marginal Relevance (more diverse results, better answers)
    """
    if search_type == "mmr":
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":        RETRIEVER_K,
                "fetch_k":  RETRIEVER_K * 3,   # fetch more, rerank for diversity
                "lambda_mult": 0.7,            # balance relevance vs diversity
            },
        )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


if __name__ == "__main__":
    from crawler import crawl_katz
    docs = crawl_katz()
    db   = build_faiss_index(docs, force_rebuild=True)
    ret  = get_retriever(db)
    results = ret.invoke("What programs does Katz School offer?")
    print(f"\nTest query returned {len(results)} chunks:")
    for r in results[:2]:
        print(f"  [{r.metadata.get('source','')}] {r.page_content[:150]}...")