"""
tools/semantic_scholar.py

Free academic paper search using the Semantic Scholar API.
No API key required for basic usage (200M+ papers).
Rate limit: 100 requests/5min unauthenticated.
"""

import requests
import time
from typing import Optional


BASE_URL = "https://api.semanticscholar.org/graph/v1"

FIELDS = "paperId,title,abstract,authors,year,citationCount,referenceCount,url,externalIds,tldr"


def search_papers(query: str, limit: int = 20) -> list[dict]:
    """Search Semantic Scholar for papers matching a query."""
    try:
        resp = requests.get(
            f"{BASE_URL}/paper/search",
            params={"query": query, "limit": limit, "fields": FIELDS},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        print(f"[SemanticScholar] search error: {e}")
        return []


def get_paper_details(paper_id: str) -> Optional[dict]:
    """Get full details for a specific paper by ID."""
    try:
        resp = requests.get(
            f"{BASE_URL}/paper/{paper_id}",
            params={"fields": FIELDS + ",references,citations"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[SemanticScholar] details error: {e}")
        return None


def get_recommendations(paper_id: str, limit: int = 10) -> list[dict]:
    """Get papers recommended based on a seed paper."""
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
            params={"limit": limit, "fields": FIELDS},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("recommendedPapers", [])
    except Exception as e:
        print(f"[SemanticScholar] recommendations error: {e}")
        return []


def format_paper(p: dict) -> str:
    """Format a paper dict into a readable string for agents."""
    authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:3])
    if len(p.get("authors", [])) > 3:
        authors += " et al."
    tldr = ""
    if p.get("tldr") and p["tldr"].get("text"):
        tldr = f"\n  TL;DR: {p['tldr']['text']}"
    abstract = (p.get("abstract") or "")[:400]
    return (
        f"Title: {p.get('title', 'Unknown')}\n"
        f"  Authors: {authors}\n"
        f"  Year: {p.get('year', 'N/A')} | Citations: {p.get('citationCount', 0)}\n"
        f"  ID: {p.get('paperId', '')}\n"
        f"  URL: {p.get('url', '')}{tldr}\n"
        f"  Abstract: {abstract}..."
    )


def search_and_format(query: str, limit: int = 15) -> str:
    """Search and return formatted results as a single string for agents."""
    papers = search_papers(query, limit)
    if not papers:
        return f"No papers found for query: {query}"
    lines = [f"Found {len(papers)} papers for '{query}':\n"]
    for i, p in enumerate(papers, 1):
        lines.append(f"[{i}] {format_paper(p)}\n")
    return "\n".join(lines)
