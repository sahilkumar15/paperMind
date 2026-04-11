from katzbot.rag_engine import KatzRAGEngine, get_engine
from katzbot.faculty import KATZ_FACULTY, match_faculty, get_faculty_documents
from katzbot.events_fetcher import (
    fetch_events,
    get_events_documents,
    match_events_to_topic,
    STATIC_EVENTS,
)
from katzbot.smart_advisor import get_smart_advice, format_email_template
from katzbot.crawler import crawl_katz
from katzbot.indexer import (
    build_faiss_index,
    get_retriever,
    FAISS_DIR,
    DOCS_PICKLE,
    INDEX_META_PICKLE,
    RETRIEVER_PICKLE,  # backward-compatible alias
)

__all__ = [
    "KatzRAGEngine",
    "get_engine",
    "KATZ_FACULTY",
    "match_faculty",
    "get_faculty_documents",
    "fetch_events",
    "get_events_documents",
    "match_events_to_topic",
    "STATIC_EVENTS",
    "get_smart_advice",
    "format_email_template",
    "crawl_katz",
    "build_faiss_index",
    "get_retriever",
    "FAISS_DIR",
    "DOCS_PICKLE",
    "INDEX_META_PICKLE",
    "RETRIEVER_PICKLE",
]
