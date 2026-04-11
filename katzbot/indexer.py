"""
katzbot/indexer.py
===================
FAISS vector store with persistent disk caching.

Why this version works:
- FAISS binary index is the primary fast-load artifact.
- A companion .pkl stores serializable source/chunk metadata for debugging,
  rebuild assistance, and notebook-style persistence.
- We DO NOT pickle the retriever object itself, because LangChain retrievers
  often contain thread locks (e.g. _thread.RLock) that are not picklable.

Saved artifacts:
  katzbot/faiss_index/
    ├── index.faiss
    ├── index.pkl              # LangChain FAISS sidecar created by save_local
    ├── source_docs.pkl        # serializable cached chunks/docs (ours)
    └── index_meta.pkl         # build metadata (ours)
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

INDEX_DIR = Path(__file__).parent / "faiss_index"
FAISS_DIR = INDEX_DIR
DOCS_PICKLE = INDEX_DIR / "source_docs.pkl"
INDEX_META_PICKLE = INDEX_DIR / "index_meta.pkl"

# Backward-compatible alias used by older imports.
# This is NOT a pickled retriever object.
RETRIEVER_PICKLE = DOCS_PICKLE

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 6


def _serialize_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
        for doc in documents
    ]


def _deserialize_documents(payload: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for item in payload:
        docs.append(
            Document(
                page_content=item.get("page_content", ""),
                metadata=item.get("metadata", {}),
            )
        )
    return docs


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(
        f"[Indexer] {len(documents)} docs → {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def save_index_artifacts(source_documents: List[Document], chunk_documents: List[Document], db) -> None:
    payload = {
        "source_documents": _serialize_documents(source_documents),
        "chunk_documents": _serialize_documents(chunk_documents),
    }
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_doc_count": len(source_documents),
        "chunk_count": len(chunk_documents),
        "vector_count": int(getattr(db.index, "ntotal", 0)),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retriever_k": RETRIEVER_K,
        "faiss_dir": str(INDEX_DIR),
        "docs_pickle": str(DOCS_PICKLE),
    }
    _save_pickle(DOCS_PICKLE, payload)
    _save_pickle(INDEX_META_PICKLE, meta)
    print(f"[Indexer] ✓ Document cache saved → {DOCS_PICKLE}")
    print(f"[Indexer] ✓ Metadata saved → {INDEX_META_PICKLE}")


def load_index_metadata() -> Dict[str, Any]:
    return _load_pickle(INDEX_META_PICKLE, default={}) or {}


def load_cached_documents(kind: str = "source") -> List[Document]:
    payload = _load_pickle(DOCS_PICKLE, default={}) or {}
    if kind == "chunk":
        return _deserialize_documents(payload.get("chunk_documents", []))
    return _deserialize_documents(payload.get("source_documents", []))


def build_faiss_index(documents: List[Document], force_rebuild: bool = False):
    """
    Build or load FAISS index.

    Fast path:
      - load FAISS from disk in ~seconds
    Fresh build:
      - split docs
      - embed chunks
      - save FAISS + companion .pkl artifacts
    """
    from llm_config import get_langchain_embeddings

    embeddings = get_langchain_embeddings()

    if not force_rebuild and INDEX_DIR.exists():
        try:
            db = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            n = int(db.index.ntotal)
            if n > 0:
                print(f"[Indexer] ✓ Loaded FAISS ({n:,} vectors) from disk")
                meta = load_index_metadata()
                if meta:
                    print(
                        f"[Indexer] Cached metadata: {meta.get('source_doc_count', 0)} source docs, "
                        f"{meta.get('chunk_count', 0)} chunks"
                    )
                return db
        except Exception as e:
            print(f"[Indexer] Could not load existing index: {e}")

    print(f"[Indexer] Building FAISS from {len(documents)} docs…")
    chunks = split_documents(documents)
    print(f"[Indexer] Embedding {len(chunks)} chunks…")
    db = FAISS.from_documents(chunks, embeddings)

    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        db.save_local(str(INDEX_DIR))
        save_index_artifacts(documents, chunks, db)
        print(f"[Indexer] ✓ FAISS index saved → {INDEX_DIR}")
    except Exception as e:
        print(f"[Indexer] Save failed: {e}")

    return db


def get_retriever(db, search_type: str = "mmr"):
    if search_type == "mmr":
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_K * 3,
                "lambda_mult": 0.7,
            },
        )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
