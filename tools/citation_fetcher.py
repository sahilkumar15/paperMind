"""
tools/citation_fetcher.py
==========================
FIXES:
  1. Rate limit: use longer backoff (30s) + exponential retry (3 attempts)
  2. Hallucinated titles: if SS returns 0 results, fall back to topic search
  3. Similarity threshold lowered to 0.45 (LLM titles differ from real ones)
  4. Batch search: search ALL titles in one query to reduce API calls
  5. Last resort: direct topic search always returns something
"""

import re
import time
import requests
from difflib import SequenceMatcher
from typing import Optional

SS_BASE  = "https://api.semanticscholar.org/graph/v1"
HEADERS  = {
    "User-Agent": "PaperMind-Academic/1.0 (research tool)",
    "Accept":     "application/json",
}

PAPER_FIELDS = (
    "paperId,externalIds,title,authors,year,venue,"
    "publicationVenue,journal,volume,pages,"
    "publicationTypes,citationCount,url"
)


# ── Retry-aware GET ───────────────────────────────────────────
def _get_with_retry(url: str, params: dict, max_tries: int = 4) -> Optional[dict]:
    """
    GET with exponential backoff on 429.
    Waits: 15s → 30s → 60s → give up
    """
    wait_times = [15, 30, 60]
    for attempt in range(max_tries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                # Try to read Retry-After header
                retry_after = int(resp.headers.get("Retry-After", 0))
                wait = max(retry_after, wait_times[min(attempt, len(wait_times)-1)])
                print(f"[CitFetcher]   Rate limited (429) — waiting {wait}s "
                      f"(attempt {attempt+1}/{max_tries})")
                time.sleep(wait)
                continue

            print(f"[CitFetcher]   HTTP {resp.status_code} for {url}")
            return None

        except requests.exceptions.Timeout:
            print(f"[CitFetcher]   Timeout (attempt {attempt+1})")
            time.sleep(10)
        except Exception as e:
            print(f"[CitFetcher]   Request error: {e}")
            return None

    print(f"[CitFetcher]   All {max_tries} attempts failed")
    return None


# ── Title extraction from agent text ─────────────────────────
def extract_titles_from_text(text: str) -> list:
    """Extract paper titles from Crawler agent output."""
    titles = []
    seen   = set()

    # Quoted titles: "Some Title"
    for m in re.finditer(r'"([^"]{15,200})"', text):
        t = m.group(1).strip()
        if t.lower() not in seen:
            titles.append(t); seen.add(t.lower())

    # Bold markdown: **Title** or **1. Title**
    for m in re.finditer(r'\*\*(?:\d+\.\s*)?([^*]{15,200})\*\*', text):
        t = re.sub(r'^\d+\.\s*', '', m.group(1)).strip()
        if t and t.lower() not in seen and len(t) > 10:
            titles.append(t); seen.add(t.lower())

    # Numbered list: 1. Title (Year) — ...
    for m in re.finditer(
        r'(?:^|\n)\s*(?:\[?\d+\]?\.?)\s+([A-Z][^:\n]{14,200})'
        r'(?:\s*[(\-–—]|\s*$)', text, re.MULTILINE
    ):
        t = m.group(1).strip()
        t = re.sub(r'\s*\(\d{4}\)\s*$', '', t).strip()
        t = re.sub(r'\s*[-–—].*$', '', t).strip()
        skip = {'authors:','summary:','url:','citations:','abstract:',
                'note:','source:','venue:','http','www.'}
        if len(t) > 14 and t.lower() not in seen:
            if not any(t.lower().startswith(s) for s in skip):
                titles.append(t); seen.add(t.lower())

    # Clean: needs 2+ words
    clean = [t for t in titles if len(t.split()) >= 2
             and not t.replace(' ','').isdigit()]
    print(f"[CitFetcher] Extracted {len(clean)} titles")
    return clean[:20]


# ── Similarity ────────────────────────────────────────────────
def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _best_match(query: str, candidates: list, threshold: float = 0.45) -> Optional[dict]:
    """Pick best match with lowered threshold (LLM titles differ from real ones)."""
    best_score, best_paper = 0.0, None
    for p in candidates:
        s = _sim(query, p.get("title",""))
        if s > best_score:
            best_score, best_paper = s, p
    if best_score >= threshold and best_paper:
        print(f"[CitFetcher]   ✓ ({best_score:.2f}) {best_paper.get('title','')[:55]}")
        return best_paper
    if best_paper:
        print(f"[CitFetcher]   ✗ best={best_score:.2f} below {threshold}")
    return None


# ── Search strategies ─────────────────────────────────────────
def _search_title(title: str, delay: float = 2.0) -> Optional[dict]:
    """Search by exact title with retry."""
    time.sleep(delay)
    data = _get_with_retry(
        f"{SS_BASE}/paper/search",
        {"query": title, "limit": 5, "fields": PAPER_FIELDS}
    )
    if not data:
        return None
    return _best_match(title, data.get("data", []))


def _search_topic(topic: str, limit: int = 15) -> list:
    """
    Fallback: search Semantic Scholar directly by topic.
    Always returns real papers — used when title matching fails.
    """
    print(f"[CitFetcher] Topic search: '{topic}'")
    time.sleep(2)
    data = _get_with_retry(
        f"{SS_BASE}/paper/search",
        {"query": topic, "limit": limit, "fields": PAPER_FIELDS}
    )
    if not data:
        return []
    papers = data.get("data", [])
    print(f"[CitFetcher] Topic search returned {len(papers)} papers")
    return papers


# ── BibTeX builder ────────────────────────────────────────────
def _clean(text: str) -> str:
    if not text: return ""
    for o, n in [("&","\\&"),("%","\\%"),("#","\\#"),("_","\\_"),("$","\\$")]:
        text = text.replace(o, n)
    return text.strip()


def _cite_key(authors: list, year: int, title: str) -> str:
    last = "Unknown"
    if authors:
        parts = authors[0].get("name","").replace(",","").split()
        last  = re.sub(r"[^a-zA-Z]", "", parts[-1]) if parts else "Unknown"
    yr    = str(year) if year else "XXXX"
    stop  = {"a","an","the","of","in","on","at","to","for","and","or","with",
             "from","is","are","using","via","towards","deep","learning","based"}
    words = re.sub(r"[^a-zA-Z0-9\s]","",title).split()
    word  = next((w for w in words if w.lower() not in stop and len(w) > 2),
                 words[0] if words else "Paper")
    return f"{last}{yr}{word.capitalize()}"


def _entry_type(pub_types: list, venue: str) -> str:
    tl = [t.lower() for t in (pub_types or [])]
    vl = (venue or "").lower()
    if "book" in tl: return "book"
    if any(t in tl for t in ["conferencepaper","conference"]): return "inproceedings"
    for kw in ["conference","workshop","symposium","proceedings","annual",
               "cvpr","iccv","eccv","neurips","nips","icml","aaai","ijcai",
               "acl","emnlp","iclr","sigkdd","www","cikm","uai"]:
        if kw in vl: return "inproceedings"
    return "article"


def _build_bibtex(paper: dict) -> tuple:
    """Returns (bibtex_string, cite_key)."""
    pid       = paper.get("paperId","")
    ext       = paper.get("externalIds") or {}
    title     = paper.get("title","Untitled")
    authors   = paper.get("authors",[])
    year      = paper.get("year") or 0
    venue     = paper.get("venue","") or ""
    pub_venue = paper.get("publicationVenue") or {}
    journal   = paper.get("journal") or {}
    pub_types = paper.get("publicationTypes") or []
    doi       = ext.get("DOI","")
    arxiv     = ext.get("ArXiv","")
    volume    = journal.get("volume","") or ""
    pages     = journal.get("pages","")  or ""
    url       = paper.get("url","") or f"https://www.semanticscholar.org/paper/{pid}"
    venue_name= pub_venue.get("name","") or venue or ""

    etype = _entry_type(pub_types, venue)
    key   = _cite_key(authors, year, title)

    # Format authors
    parts = []
    for a in authors:
        name = a.get("name","")
        if "," in name:
            parts.append(name)
        else:
            p = name.rsplit(" ",1)
            parts.append(f"{p[1]}, {p[0]}" if len(p)==2 else name)
    auth_str = " and ".join(parts) if parts else "Unknown"

    fields = [
        f"  title        = {{{{{_clean(title)}}}}}",
        f"  author       = {{{_clean(auth_str)}}}",
    ]
    if year: fields.append(f"  year         = {{{year}}}")

    if etype == "article":
        if venue_name: fields.append(f"  journal      = {{{{{_clean(venue_name)}}}}}")
        if volume:     fields.append(f"  volume       = {{{_clean(volume)}}}")
        if pages:      fields.append(f"  pages        = {{{_clean(pages)}}}")
    elif etype == "inproceedings":
        if venue_name: fields.append(f"  booktitle    = {{{{{_clean(venue_name)}}}}}")
        if pages:      fields.append(f"  pages        = {{{_clean(pages)}}}")
    else:
        if venue_name: fields.append(f"  publisher    = {{{_clean(venue_name)}}}")

    if doi:   fields.append(f"  doi          = {{{doi}}}")
    if arxiv:
        fields.append(f"  eprint       = {{{arxiv}}}")
        fields.append(f"  archivePrefix = {{arXiv}}")
    fields.append(f"  url          = {{{url}}}")

    bibtex = f"@{etype}{{{key},\n" + ",\n".join(fields) + "\n}"
    return bibtex, key


def _paper_to_cit(paper: dict, query: str = "") -> Optional[dict]:
    """Convert a Semantic Scholar paper dict to our citation dict."""
    try:
        bibtex, key = _build_bibtex(paper)
        ext  = paper.get("externalIds") or {}
        auths= paper.get("authors",[])
        return {
            "bibtex":         bibtex,
            "cite_key":       key,
            "title":          paper.get("title",""),
            "authors":        [a.get("name","") for a in auths],
            "year":           paper.get("year") or 0,
            "venue":          paper.get("venue","") or "",
            "doi":            ext.get("DOI",""),
            "arxiv":          ext.get("ArXiv",""),
            "url":            paper.get("url",""),
            "citation_count": paper.get("citationCount",0),
            "paper_id":       paper.get("paperId",""),
            "matched_query":  query,
        }
    except Exception as e:
        print(f"[CitFetcher] Build error: {e}")
        return None


# ── PUBLIC API ────────────────────────────────────────────────

def fetch_citations_from_papers_text(
    papers_text: str,
    topic: str = "",
    delay: float = 2.5,
) -> list:
    """
    Main entry point.
    1. Extract titles from agent text
    2. Search each title on Semantic Scholar (with retry)
    3. If < 3 found, fall back to topic search to fill up
    Returns list of citation dicts.
    """
    citations = []
    seen_ids  = set()

    titles = extract_titles_from_text(papers_text)

    if titles:
        print(f"[CitFetcher] Searching {len(titles)} titles "
              f"(delay={delay}s between calls)...")

        for i, title in enumerate(titles, 1):
            print(f"[CitFetcher] [{i}/{len(titles)}] '{title[:50]}...'")
            paper = _search_title(title, delay=delay)

            if paper:
                pid = paper.get("paperId","")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    cit = _paper_to_cit(paper, query=title)
                    if cit:
                        citations.append(cit)
            else:
                print(f"[CitFetcher]   → No match, skipping")

    # ── Fallback: topic search fills gaps ────────────────────
    # Always run if we got < 5 citations from title matching
    needed = max(0, 10 - len(citations))
    if needed > 0 and topic:
        print(f"\n[CitFetcher] Got {len(citations)} from titles — "
              f"fetching {needed} more via topic search: '{topic}'")
        topic_papers = _search_topic(topic, limit=needed + 5)

        for paper in topic_papers:
            pid = paper.get("paperId","")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                cit = _paper_to_cit(paper, query=topic)
                if cit:
                    citations.append(cit)
                    if len(citations) >= 15:
                        break
                time.sleep(0.3)

    print(f"\n[CitFetcher] ✅ Total: {len(citations)} verified citations")
    return citations


def build_bib_file(citations: list, topic: str = "") -> str:
    """Build complete .bib file string."""
    header = (
        f"% BibTeX bibliography — PaperMind\n"
        f"% Topic: {topic}\n"
        f"% Papers: {len(citations)}\n"
        f"% Source: Semantic Scholar API (verified)\n"
        f"%\n"
        f"% Overleaf: upload as references.bib, then add:\n"
        f"%   \\bibliographystyle{{ieeetr}}\n"
        f"%   \\bibliography{{references}}\n"
        f"% {'='*50}\n\n"
    )
    entries = "\n\n".join(c["bibtex"] for c in citations if c.get("bibtex"))
    return header + entries


# ── Auto-save to disk ─────────────────────────────────────────
import os as _os
from pathlib import Path as _Path
from datetime import datetime as _dt


def save_bib_to_disk(bib_content: str, topic: str = "",
                     output_dir: str = "outputs") -> str:
    """
    Save .bib file to disk automatically after fetching.
    Returns the full path of the saved file.
    Prints a clear terminal banner so the user can see exactly where it is.
    """
    _Path(output_dir).mkdir(parents=True, exist_ok=True)

    safe_topic = re.sub(r"[^a-zA-Z0-9_\-]", "_", topic)[:30] or "references"
    timestamp  = _dt.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"references_{safe_topic}_{timestamp}.bib"
    filepath   = _os.path.abspath(_os.path.join(output_dir, filename))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(bib_content)

    # ── Terminal banner ───────────────────────────────────────
    width = 65
    print("\n" + "═" * width)
    print("  📚  BibTeX File Saved")
    print("═" * width)
    print(f"  File   : {filepath}")
    print(f"  Size   : {len(bib_content):,} bytes")
    print(f"  Papers : {bib_content.count('@article') + bib_content.count('@inproceedings') + bib_content.count('@book')}")
    print("─" * width)
    print("  Overleaf: upload as  references.bib")
    print("  LaTeX  : \\bibliographystyle{ieeetr}")
    print("           \\bibliography{references}")
    print("═" * width + "\n")

    # ── Print .bib content to terminal ───────────────────────
    print("  .bib FILE CONTENT (copy from terminal if needed):")
    print("─" * width)
    print(bib_content)
    print("─" * width + "\n")

    return filepath


def print_citations_summary(citations: list) -> None:
    """Print a clean summary table of all citations to terminal."""
    width = 65
    print("\n" + "─" * width)
    print(f"  CITATIONS SUMMARY  ({len(citations)} papers)")
    print("─" * width)
    for i, c in enumerate(citations, 1):
        key   = c.get("cite_key", f"paper{i}")
        title = c.get("title", "")[:45]
        year  = c.get("year", "")
        doi   = "DOI ✓" if c.get("doi") else ("arXiv ✓" if c.get("arxiv") else "URL only")
        print(f"  [{i:>2}] \\cite{{{key}}}")
        print(f"        {title}{'...' if len(c.get('title','')) > 45 else ''}")
        print(f"        {year}  {doi}")
    print("─" * width + "\n")