"""
tools/citation_fetcher.py
==========================
Fetches EXACT BibTeX citations from Semantic Scholar API.
Extracts titles from agent text → searches SS by title → builds verified BibTeX.
Falls back to topic search if title matching fails.
"""

import re
import time
import requests
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from datetime import datetime

SS_BASE  = "https://api.semanticscholar.org/graph/v1"
HEADERS  = {"User-Agent": "ScholarMind-BibTeX/1.0", "Accept": "application/json"}
FIELDS   = ("paperId,externalIds,title,authors,year,venue,"
            "publicationVenue,journal,volume,pages,"
            "publicationTypes,citationCount,url")


# ── Retry-aware GET ───────────────────────────────────────────
def _get(url: str, params: dict, tries: int = 3) -> Optional[dict]:
    waits = [15, 30, 60]
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                w = int(resp.headers.get("Retry-After", waits[min(attempt, 2)]))
                print(f"[CitFetcher] Rate limit — waiting {w}s")
                time.sleep(w)
                continue
        except requests.exceptions.Timeout:
            time.sleep(10)
        except Exception as e:
            print(f"[CitFetcher] Request error: {e}")
            return None
    return None


# ── Title extraction ──────────────────────────────────────────
def extract_titles_from_text(text: str) -> list:
    titles = []
    seen   = set()

    for m in re.finditer(r'"([^"]{15,200})"', text):
        t = m.group(1).strip()
        if t.lower() not in seen:
            titles.append(t); seen.add(t.lower())

    for m in re.finditer(r'\*\*(?:\d+\.\s*)?([^*]{15,200})\*\*', text):
        t = re.sub(r'^\d+\.\s*', '', m.group(1)).strip()
        if t and t.lower() not in seen and len(t) > 10:
            titles.append(t); seen.add(t.lower())

    for m in re.finditer(
        r'(?:^|\n)\s*(?:\[?\d+\]?\.?)\s+([A-Z][^:\n]{14,200})'
        r'(?:\s*[(\-–—]|\s*$)', text, re.MULTILINE
    ):
        t = re.sub(r'\s*\(\d{4}\)\s*$', '', m.group(1)).strip()
        t = re.sub(r'\s*[-–—].*$', '', t).strip()
        skip = {'authors:','summary:','url:','citations:','http','www.'}
        if len(t) > 14 and t.lower() not in seen:
            if not any(t.lower().startswith(s) for s in skip):
                titles.append(t); seen.add(t.lower())

    clean = [t for t in titles if len(t.split()) >= 2]
    print(f"[CitFetcher] Extracted {len(clean)} titles")
    return clean[:20]


# ── Matching ──────────────────────────────────────────────────
def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _best_match(query: str, candidates: list, thresh: float = 0.45) -> Optional[dict]:
    best_s, best_p = 0.0, None
    for p in candidates:
        s = _sim(query, p.get("title",""))
        if s > best_s:
            best_s, best_p = s, p
    if best_s >= thresh and best_p:
        print(f"[CitFetcher]   ✓ ({best_s:.2f}) {best_p.get('title','')[:50]}")
        return best_p
    if best_p:
        print(f"[CitFetcher]   ✗ best={best_s:.2f}")
    return None


# ── BibTeX builder ────────────────────────────────────────────
def _clean(t: str) -> str:
    if not t: return ""
    for o, n in [("&","\\&"),("%","\\%"),("#","\\#"),("_","\\_"),("$","\\$")]:
        t = t.replace(o, n)
    return t.strip()


def _cite_key(authors: list, year: int, title: str) -> str:
    last = "Unknown"
    if authors:
        parts = authors[0].get("name","").replace(",","").split()
        last  = re.sub(r"[^a-zA-Z]","", parts[-1]) if parts else "Unknown"
    yr   = str(year) if year else "XXXX"
    stop = {"a","an","the","of","in","on","to","for","and","or","with",
            "from","is","are","using","via","towards","based","deep","learning"}
    words = re.sub(r"[^a-zA-Z0-9\s]","",title).split()
    word  = next((w for w in words if w.lower() not in stop and len(w)>2),
                 words[0] if words else "Paper")
    return f"{last}{yr}{word.capitalize()}"


def _entry_type(pub_types: list, venue: str) -> str:
    tl = [t.lower() for t in (pub_types or [])]
    vl = (venue or "").lower()
    if "book" in tl: return "book"
    if any(t in tl for t in ["conferencepaper","conference"]): return "inproceedings"
    for kw in ["conference","workshop","symposium","proceedings",
               "cvpr","iccv","eccv","neurips","icml","aaai","acl","emnlp","iclr"]:
        if kw in vl: return "inproceedings"
    return "article"


def _build_bibtex(paper: dict) -> tuple:
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
    vname     = pub_venue.get("name","") or venue or ""

    etype = _entry_type(pub_types, venue)
    key   = _cite_key(authors, year, title)

    parts = []
    for a in authors:
        name = a.get("name","")
        if "," in name:
            parts.append(name)
        else:
            p = name.rsplit(" ", 1)
            parts.append(f"{p[1]}, {p[0]}" if len(p)==2 else name)
    auth = " and ".join(parts) if parts else "Unknown"

    fields = [
        f"  title        = {{{{{_clean(title)}}}}}",
        f"  author       = {{{_clean(auth)}}}",
    ]
    if year: fields.append(f"  year         = {{{year}}}")
    if etype == "article":
        if vname:  fields.append(f"  journal      = {{{{{_clean(vname)}}}}}")
        if volume: fields.append(f"  volume       = {{{_clean(volume)}}}")
        if pages:  fields.append(f"  pages        = {{{_clean(pages)}}}")
    elif etype == "inproceedings":
        if vname:  fields.append(f"  booktitle    = {{{{{_clean(vname)}}}}}")
        if pages:  fields.append(f"  pages        = {{{_clean(pages)}}}")
    else:
        if vname:  fields.append(f"  publisher    = {{{_clean(vname)}}}")
    if doi:   fields.append(f"  doi          = {{{doi}}}")
    if arxiv:
        fields.append(f"  eprint       = {{{arxiv}}}")
        fields.append(f"  archivePrefix = {{arXiv}}")
    fields.append(f"  url          = {{{url}}}")

    return f"@{etype}{{{key},\n" + ",\n".join(fields) + "\n}}", key


def _paper_to_cit(paper: dict, query: str = "") -> Optional[dict]:
    try:
        bibtex, key = _build_bibtex(paper)
        ext   = paper.get("externalIds") or {}
        auths = paper.get("authors",[])
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


# ── Public API ────────────────────────────────────────────────
def fetch_citations_from_papers_text(
    papers_text: str, topic: str = "", delay: float = 2.5
) -> list:
    """
    Main entry point.
    Extracts titles from agent output → searches SS by title → builds BibTeX.
    Falls back to topic search if < 5 citations matched.
    """
    citations = []
    seen_ids  = set()
    titles    = extract_titles_from_text(papers_text)

    if titles:
        print(f"[CitFetcher] Searching {len(titles)} titles…")
        for i, title in enumerate(titles, 1):
            print(f"[CitFetcher] [{i}/{len(titles)}] '{title[:50]}…'")
            time.sleep(delay)
            data = _get(f"{SS_BASE}/paper/search",
                        {"query": title, "limit": 5, "fields": FIELDS})
            if data:
                paper = _best_match(title, data.get("data",[]))
                if paper:
                    pid = paper.get("paperId","")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        cit = _paper_to_cit(paper, query=title)
                        if cit:
                            citations.append(cit)

    # Fallback: topic search fills gaps
    needed = max(0, 10 - len(citations))
    if needed > 0 and topic:
        print(f"[CitFetcher] Filling {needed} via topic search: '{topic}'")
        time.sleep(2)
        data = _get(f"{SS_BASE}/paper/search",
                    {"query": topic, "limit": needed+5, "fields": FIELDS})
        if data:
            for paper in data.get("data",[]):
                pid = paper.get("paperId","")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    cit = _paper_to_cit(paper, query=topic)
                    if cit:
                        citations.append(cit)
                    if len(citations) >= 15:
                        break

    print(f"[CitFetcher] ✅ {len(citations)} verified citations")
    return citations


def build_bib_file(citations: list, topic: str = "") -> str:
    header = (
        f"% BibTeX bibliography — ScholarMind\n"
        f"% Topic: {topic}\n"
        f"% Papers: {len(citations)}\n"
        f"% Source: Semantic Scholar API (verified)\n"
        f"%\n"
        f"% Overleaf: upload as references.bib, then add:\n"
        f"%   \\bibliographystyle{{ieeetr}}\n"
        f"%   \\bibliography{{references}}\n"
        f"% {'='*50}\n\n"
    )
    return header + "\n\n".join(c["bibtex"] for c in citations if c.get("bibtex"))


def save_bib_to_disk(bib_content: str, topic: str = "",
                     output_dir: str = "outputs") -> str:
    import os
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe  = re.sub(r"[^a-zA-Z0-9_\-]", "_", topic)[:30] or "references"
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"references_{safe}_{ts}.bib"
    fpath = os.path.abspath(os.path.join(output_dir, fname))
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(bib_content)
    n = bib_content.count("@article") + bib_content.count("@inproceedings") + bib_content.count("@book")
    print("\n" + "═"*60)
    print("  📚  BibTeX File Saved")
    print("═"*60)
    print(f"  File  : {fpath}")
    print(f"  Papers: {n}")
    print("═"*60 + "\n")
    print(bib_content)
    print("─"*60 + "\n")
    return fpath


def print_citations_summary(citations: list) -> None:
    print("\n" + "─"*60)
    print(f"  CITATIONS SUMMARY  ({len(citations)} papers)")
    print("─"*60)
    for i, c in enumerate(citations, 1):
        key  = c.get("cite_key", f"paper{i}")
        title = c.get("title","")[:45]
        year  = c.get("year","")
        doi   = "DOI ✓" if c.get("doi") else ("arXiv ✓" if c.get("arxiv") else "URL only")
        print(f"  [{i:>2}] \\cite{{{key}}}")
        print(f"        {title}{'…' if len(c.get('title',''))>45 else ''}")
        print(f"        {year}  {doi}")
    print("─"*60 + "\n")
