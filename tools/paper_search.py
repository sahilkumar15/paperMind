"""
tools/paper_search.py
=====================
Deterministic paper search for KatzScholarMind.

Why this exists:
- Groq tool-calling can hallucinate tool names (for example brave_search)
- CrewAI research runs should not depend on LLM-chosen web tools
- Semantic Scholar is great when available, but rate limits or empty results happen
- arXiv is a strong free fallback for CS / AI topics

Strategy:
1. Query Semantic Scholar first
2. If results are sparse or fail, query arXiv
3. Deduplicate and format results once in Python
4. Pass the paper list into the agents as plain text context
"""

from __future__ import annotations

import html
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

SS_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = (
    "title,authors,year,abstract,citationCount,externalIds,url,venue,publicationTypes"
)
USER_AGENT = "KatzScholarMind/1.0"
DEFAULT_TIMEOUT = 20


def _clean_text(text: str, max_len: int = 260) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _semantic_scholar_search(query: str, limit: int = 12) -> List[Dict]:
    headers = {"User-Agent": USER_AGENT}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(limit, 20),
        "fields": SS_FIELDS,
    }

    waits = [2, 5, 10]
    for attempt in range(len(waits) + 1):
        try:
            resp = requests.get(
                SS_URL,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                papers = []
                for p in data:
                    title = _clean_text(p.get("title", ""), 220)
                    if not title:
                        continue
                    authors = [a.get("name", "") for a in p.get("authors", [])[:4] if a.get("name")]
                    summary = _clean_text(p.get("abstract") or "No abstract available.", 240)
                    papers.append(
                        {
                            "title": title,
                            "authors": authors,
                            "year": p.get("year") or "N/A",
                            "citations": int(p.get("citationCount") or 0),
                            "summary": summary,
                            "url": p.get("url") or "",
                            "source": "Semantic Scholar",
                        }
                    )
                return papers

            if resp.status_code in {429, 500, 502, 503, 504} and attempt < len(waits):
                retry_after = resp.headers.get("Retry-After")
                sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else waits[attempt]
                time.sleep(sleep_s)
                continue

            return []
        except requests.RequestException:
            if attempt < len(waits):
                time.sleep(waits[attempt])
                continue
            return []

    return []


def _arxiv_search(query: str, limit: int = 12) -> List[Dict]:
    # arXiv API works well for AI / CS topics and does not require an API key.
    encoded = urllib.parse.quote(query)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}&start=0&max_results={min(limit, 20)}"
        "&sortBy=relevance&sortOrder=descending"
    )
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    papers: List[Dict] = []
    for entry in root.findall("atom:entry", ns):
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns), 220)
        if not title:
            continue

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", default="", namespaces=ns).strip()
            if name:
                authors.append(name)
        summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns), 240)
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = published[:4] if published else "N/A"
        entry_id = entry.findtext("atom:id", default="", namespaces=ns)

        papers.append(
            {
                "title": title,
                "authors": authors[:4],
                "year": year,
                "citations": 0,
                "summary": summary or "No abstract available.",
                "url": entry_id,
                "source": "arXiv",
            }
        )
    return papers


def _dedupe_papers(papers: List[Dict], max_results: int) -> List[Dict]:
    seen = set()
    out = []
    for paper in papers:
        norm = _norm_title(paper.get("title", ""))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(paper)
        if len(out) >= max_results:
            break
    return out


def search_papers(query: str, max_results: int = 12) -> List[Dict]:
    """Deterministically retrieve papers without LLM tool-calling."""
    ss_results = _semantic_scholar_search(query, limit=max_results)

    # If SS works well, keep it primary but still enrich with a few arXiv results.
    need = max(0, max_results - len(ss_results))
    arxiv_results = []
    if need > 0:
        arxiv_results = _arxiv_search(query, limit=max(max_results, 8))
    elif len(ss_results) < max_results // 2:
        arxiv_results = _arxiv_search(query, limit=max_results)

    merged = _dedupe_papers(ss_results + arxiv_results, max_results=max_results)
    return merged


def format_papers_for_display(papers: List[Dict]) -> str:
    if not papers:
        return (
            "No papers found from Semantic Scholar or arXiv. "
            "Try a broader topic or check your network/API setup."
        )

    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors", [])[:2]) or "Unknown authors"
        if len(p.get("authors", [])) > 2:
            authors += " et al."
        cites = p.get("citations", 0)
        cite_txt = f"Citations: {cites}" if cites else "Citations: N/A"
        lines.append(
            f"{i}. \"{p.get('title', 'Untitled')}\" ({p.get('year', 'N/A')}) — "
            f"{authors} — {cite_txt} — Source: {p.get('source', 'Unknown')}\n"
            f"   Summary: {p.get('summary', 'No abstract available.')}\n"
            f"   URL: {p.get('url', '')}"
        )
    return "\n\n".join(lines)


def format_papers_for_prompt(papers: List[Dict]) -> str:
    """Compact context string for the LLM tasks."""
    if not papers:
        return "No papers found."

    blocks = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors", [])[:3]) or "Unknown authors"
        blocks.append(
            f"[{i}] {p.get('title', 'Untitled')}\n"
            f"Year: {p.get('year', 'N/A')}\n"
            f"Authors: {authors}\n"
            f"Source: {p.get('source', 'Unknown')}\n"
            f"Citations: {p.get('citations', 0)}\n"
            f"Summary: {p.get('summary', 'No abstract available.')}\n"
            f"URL: {p.get('url', '')}"
        )
    return "\n\n".join(blocks)
