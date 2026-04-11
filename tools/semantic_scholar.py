"""
tools/semantic_scholar.py
==========================
Semantic Scholar search tool for CrewAI agents.
Simple string-in / string-out signature to avoid Groq function-call schema errors.
"""

import time
import requests
from crewai.tools import tool

SS_URL  = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS  = ("title,authors,year,abstract,citationCount,"
           "externalIds,url,venue,publicationTypes")
HEADERS = {"User-Agent": "ScholarMind/1.0"}


def _search(query: str, limit: int = 20) -> list:
    try:
        resp = requests.get(
            SS_URL,
            params={"query": query, "limit": min(limit, 25), "fields": FIELDS},
            headers=HEADERS, timeout=15,
        )
        if resp.status_code == 429:
            time.sleep(15)
            resp = requests.get(
                SS_URL,
                params={"query": query, "limit": min(limit, 25), "fields": FIELDS},
                headers=HEADERS, timeout=15,
            )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        return [{"error": str(e)}]


def _format(papers: list) -> str:
    if not papers:
        return "No papers found."
    lines = []
    for i, p in enumerate(papers, 1):
        if "error" in p:
            lines.append(f"Error: {p['error']}")
            continue
        title   = p.get("title", "Unknown")
        year    = p.get("year", "N/A")
        cites   = p.get("citationCount", 0)
        authors = ", ".join(a.get("name","") for a in p.get("authors",[])[:2])
        if len(p.get("authors",[])) > 2:
            authors += " et al."
        abstract = (p.get("abstract") or "No abstract.")[:200]
        url      = p.get("url", "")
        lines.append(
            f"{i}. \"{title}\" ({year}) — {authors} — Citations: {cites}\n"
            f"   Summary: {abstract}\n"
            f"   URL: {url}"
        )
    return "\n\n".join(lines)


@tool("semantic_scholar_search")
def semantic_scholar_search(query: str) -> str:
    """
    Search Semantic Scholar for academic papers on a given topic.
    Input: a search query string.
    Output: numbered list of papers with titles, authors, abstracts, citation counts, URLs.
    Use this to find real academic papers for literature reviews.
    """
    papers = _search(query, limit=20)
    return _format(papers)
