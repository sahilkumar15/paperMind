"""
katzbot/__init__.py
====================
Clean public API for the katzbot package.
"""
from katzbot.rag_engine import KatzRAGEngine, get_engine
from katzbot.faculty    import KATZ_FACULTY, match_faculty, get_faculty_documents
from katzbot.crawler    import crawl_katz
from katzbot.indexer    import build_faiss_index, get_retriever

__all__ = [
    "KatzRAGEngine",
    "get_engine",
    "KATZ_FACULTY",
    "match_faculty",
    "get_faculty_documents",
    "crawl_katz",
    "build_faiss_index",
    "get_retriever",
]