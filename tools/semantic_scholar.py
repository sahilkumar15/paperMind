"""
tools/semantic_scholar.py
===========================
Plain Python tool for Semantic Scholar API.
Uses @tool decorator from crewai but WITHOUT complex JSON schema
that causes Groq's GroqException "Failed to call a function".

KEY FIX: We use a simple string-in / string-out tool signature
so LiteLLM/Groq never tries to generate a complex function-call JSON blob.
"""

import requests
import json
from crewai.tools import tool


SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

FIELDS = (
    "title,authors,year,abstract,citationCount,"
    "externalIds,url,venue,publicationTypes"
)


def _search_papers(query: str, limit: int = 20) -> list[dict]:
    """Internal search — returns list of paper dicts."""
    params = {
        "query": query,
        "limit": min(limit, 25),
        "fields": FIELDS,
    }
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        return [{"error": str(e)}]


def _format_papers(papers: list[dict]) -> str:
    """Format paper list to readable markdown string."""
    if not papers:
        return "No papers found."

    lines = []
    for i, p in enumerate(papers, 1):
        if "error" in p:
            lines.append(f"Error: {p['error']}")
            continue
        title   = p.get("title", "Unknown title")
        year    = p.get("year", "N/A")
        cites   = p.get("citationCount", 0)
        authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        abstract = (p.get("abstract") or "No abstract.")[:300]
        url      = p.get("url", "")
        venue    = p.get("venue", "")

        lines.append(
            f"**{i}. {title}** ({year})\n"
            f"   Authors: {authors}\n"
            f"   Venue: {venue} | Citations: {cites}\n"
            f"   Abstract: {abstract}...\n"
            f"   URL: {url}\n"
        )
    return "\n".join(lines)


# ── CrewAI Tool ───────────────────────────────────────────────
# IMPORTANT: Keep signature as (query: str) -> str  
# Complex type annotations cause Groq function-call failures

@tool("semantic_scholar_search")
def semantic_scholar_search(query: str) -> str:
    """
    Search Semantic Scholar for academic papers on a given topic.
    Input: a search query string (e.g. 'transformer attention mechanisms').
    Output: formatted list of up to 20 relevant papers with titles, authors,
    abstracts, citation counts, and URLs.
    Use this to find real academic papers for literature reviews.
    """
    papers = _search_papers(query, limit=20)
    return _format_papers(papers)


@tool("semantic_scholar_details")
def semantic_scholar_details(paper_title: str) -> str:
    """
    Get more details about a specific paper by searching its title.
    Input: the title or partial title of the paper.
    Output: detailed information about the matching paper.
    """
    papers = _search_papers(paper_title, limit=5)
    return _format_papers(papers)