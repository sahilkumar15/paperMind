"""
katzbot/rag_engine.py
======================
KatzBot RAG engine.
"""

from pathlib import Path
from typing import Optional

from katzbot.crawler import crawl_katz
from katzbot.indexer import build_faiss_index, get_retriever, FAISS_DIR
from katzbot.chain import build_chain, get_llm
from katzbot.faculty import match_faculty, get_faculty_documents
from katzbot.events_fetcher import fetch_events, get_events_documents, match_events_to_topic


_CLUB_CONTACTS = {
    "cs": {
        "name": "CS & AI Club / Katz Student Life",
        "email": "katzgrad@yu.edu",
        "url": "https://www.yu.edu/katz/clubs",
        "note": "For CS & AI Club participation, start with Katz admissions/student-life and ask to be connected to the CS & AI Club organizers."
    },
    "cybersecurity": {
        "name": "Cybersecurity Club / Katz Student Life",
        "email": "katzgrad@yu.edu",
        "url": "https://www.yu.edu/katz/clubs",
        "note": "For Cybersecurity Club participation, start with Katz admissions/student-life and ask to be connected to the Cybersecurity Club organizers."
    },
    "data": {
        "name": "Data Science Society / Katz Student Life",
        "email": "katzgrad@yu.edu",
        "url": "https://www.yu.edu/katz/clubs",
        "note": "For Data Science Society participation, start with Katz admissions/student-life and ask to be connected to the club organizers."
    },
}


def _is_club_question(question: str) -> bool:
    q = question.lower()
    return any(x in q for x in ["club", "clubs", "student club", "organization", "society"])


def _is_faculty_question(question: str) -> bool:
    q = question.lower()
    if _is_club_question(q):
        return False
    return any(x in q for x in ["who is", "prof", "professor", "faculty"])


def _is_events_question(question: str) -> bool:
    q = question.lower()
    return any(x in q for x in ["event", "events", "upcoming", "info session", "workshop", "symposium", "ideathon"])


def _direct_faculty_answer(question: str):
    matches = match_faculty(question, top_k=1)
    if not matches:
        return None

    f = matches[0]
    return {
        "answer": (
            f"{f['name']} is {f['title']}. "
            f"Email: {f['email']}. "
            f"Expertise: {', '.join(f['expertise'][:5])}. "
            f"{f['note']}"
        ),
        "sources": [f["profile"]],
        "faculty_matches": matches,
    }


def _direct_events_answer(question: str):
    events = fetch_events()

    matched = match_events_to_topic(events, question, top_k=5)
    if not matched:
        matched = events[:5]

    lines = []
    for ev in matched:
        title = ev.get("title", "").strip()
        date = ev.get("date", "TBD")
        url = ev.get("url", "").strip()
        if title:
            line = f"{title} — {date}"
            if url:
                line += f" — {url}"
            lines.append(line)

    if not lines:
        return {
            "answer": "I don't have specific upcoming Katz event details right now. Please check https://www.yu.edu/katz/events and https://www.yu.edu/katz/info-sessions.",
            "sources": ["https://www.yu.edu/katz/events"],
            "faculty_matches": [],
        }

    return {
        "answer": "Relevant Katz-related events include: " + "; ".join(lines) + ".",
        "sources": [ev.get("url", "https://www.yu.edu/katz/events") for ev in matched[:3]],
        "faculty_matches": [],
    }


def _direct_club_answer(question: str):
    q = question.lower()

    if "indian" in q:
        return {
            "answer": (
                "I do not currently have a verified Katz or Yeshiva contact for an Indian student club in the indexed data. "
                "The safest contact is Katz graduate admissions/student life at katzgrad@yu.edu, and you can also check https://www.yu.edu/katz/clubs."
            ),
            "sources": ["https://www.yu.edu/katz/clubs"],
            "faculty_matches": [],
        }

    if "cs" in q or "computer science" in q or "ai" in q:
        info = _CLUB_CONTACTS["cs"]
        return {
            "answer": f"{info['name']}: contact {info['email']}. {info['note']}",
            "sources": [info["url"]],
            "faculty_matches": [],
        }

    if "cyber" in q:
        info = _CLUB_CONTACTS["cybersecurity"]
        return {
            "answer": f"{info['name']}: contact {info['email']}. {info['note']}",
            "sources": [info["url"]],
            "faculty_matches": [],
        }

    if "data" in q:
        info = _CLUB_CONTACTS["data"]
        return {
            "answer": f"{info['name']}: contact {info['email']}. {info['note']}",
            "sources": [info["url"]],
            "faculty_matches": [],
        }

    return {
        "answer": (
            "Katz clubs and organizations are listed at https://www.yu.edu/katz/clubs. "
            "For participation or organizer contact details, email katzgrad@yu.edu and ask to be connected to the relevant student club."
        ),
        "sources": ["https://www.yu.edu/katz/clubs"],
        "faculty_matches": [],
    }


class KatzRAGEngine:
    def __init__(self):
        self._db = None
        self._retriever = None
        self._chain = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _wire_runtime(self, db):
        self._db = db
        self._retriever = get_retriever(db, search_type="mmr")
        self._chain = build_chain(self._retriever, get_llm())
        self._ready = True

    def _stats_payload(self, source: str) -> dict:
        docs_pkl = str(Path(FAISS_DIR) / "source_docs.pkl")
        meta_pkl = str(Path(FAISS_DIR) / "index_meta.pkl")
        return {
            "index_vectors": self._db.index.ntotal if self._db is not None else 0,
            "index_dir": str(FAISS_DIR),
            "pkl_path": docs_pkl,
            "meta_pkl_path": meta_pkl,
            "source": source,
        }

    def _try_load_disk(self) -> bool:
        if not FAISS_DIR.exists():
            return False

        try:
            from llm_config import get_langchain_embeddings
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError:
                from langchain.vectorstores import FAISS

            emb = get_langchain_embeddings()
            db = FAISS.load_local(
                str(FAISS_DIR),
                emb,
                allow_dangerous_deserialization=True,
            )

            n = db.index.ntotal
            if n > 20:
                print(f"[KatzBot] ✓ Loaded FAISS index ({n:,} vectors) from disk")
                self._wire_runtime(db)
                return True

            print(f"[KatzBot] Index too small ({n} vectors) — rebuilding")
        except Exception as e:
            print(f"[KatzBot] Disk load failed: {e}")

        return False

    def _ensure_loaded(self):
        if self._ready:
            return
        if self._try_load_disk():
            return
        print("[KatzBot] No index found — building automatically…")
        self.build(force_refresh=False)

    def build(self, force_refresh: bool = False) -> dict:
        print("\n[KatzBot] ══════ Building Index ══════")

        if not force_refresh and self._try_load_disk():
            stats = self._stats_payload("disk_cache")
            stats.update({
                "web_pages": 0,
                "faculty_docs": 0,
                "event_docs": 0,
                "total_docs": 0,
            })
            return stats

        web_docs = crawl_katz(force_refresh=force_refresh)
        faculty_docs = get_faculty_documents()
        event_docs = get_events_documents()

        all_docs = web_docs + faculty_docs + event_docs

        print(
            f"[KatzBot] {len(web_docs)} web + {len(faculty_docs)} faculty + "
            f"{len(event_docs)} events = {len(all_docs)} total docs"
        )

        db = build_faiss_index(all_docs, force_rebuild=force_refresh)
        self._wire_runtime(db)

        stats = self._stats_payload("fresh_build")
        stats.update({
            "web_pages": len(web_docs),
            "faculty_docs": len(faculty_docs),
            "event_docs": len(event_docs),
            "total_docs": len(all_docs),
        })

        print(f"[KatzBot] ══════ Ready: {stats['index_vectors']:,} vectors ══════\n")
        return stats

    def ask(self, question: str, history: list = None, extra_context: str = "") -> dict:
        self._ensure_loaded()

        if _is_club_question(question):
            return _direct_club_answer(question)

        if _is_faculty_question(question):
            direct = _direct_faculty_answer(question)
            if direct:
                return direct

        if _is_events_question(question):
            return _direct_events_answer(question)

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
        return {
            "status": "ready",
            "vectors": self._db.index.ntotal,
            "index_dir": str(FAISS_DIR),
        }


_engine: Optional[KatzRAGEngine] = None


def get_engine() -> KatzRAGEngine:
    global _engine
    if _engine is None:
        _engine = KatzRAGEngine()
    return _engine