"""
katzbot/crawler.py
===================
Faithful port of the original notebook's data loading strategy — upgraded for 2026.

Original approach (notebook):
  1. Parse yu.edu/sitemap.xml → 1,262 URLs
  2. Filter to katz + inject extra katz URLs
  3. UnstructuredURLLoader → load all pages
  4. Cache as pickle

Modern upgrades:
  - Multiple sitemap sources (sitemap.xml + sitemap_index.xml)
  - BeautifulSoup XML parser (same as notebook, works offline)
  - Parallel fetching with ThreadPoolExecutor (was sequential in notebook)
  - Smart content extraction (main/article tags, strips boilerplate)
  - LangChain Document objects with rich metadata
  - JSON cache instead of pickle (portable, inspectable)
  - 7-day cache expiry (same cadence as original)
"""

import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── LangChain Document (version-safe import) ──────────────────
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# ── Constants ─────────────────────────────────────────────────
CACHE_FILE   = Path(__file__).parent / "pages_cache.json"
CACHE_TTL    = 7 * 24 * 3600          # 7 days (same as original)
MAX_WORKERS  = 8                       # parallel fetchers
REQUEST_TO   = 12                      # seconds
MAX_CONTENT  = 6000                    # chars per page (original used full text)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PaperMind-KatzBot/2.0; "
        "+https://www.yu.edu/katz)"
    )
}

# ── Sitemap sources (the original only used sitemap.xml) ──────
SITEMAP_URLS = [
    "https://www.yu.edu/sitemap.xml",
    "https://www.yu.edu/sitemap_index.xml",
    "https://www.yu.edu/katz/sitemap.xml",
]

# ── Extra Katz URLs not always in sitemap ─────────────────────
# (mirrors the notebook's manual URL injection)
EXTRA_KATZ_URLS = [
    "https://www.yu.edu/katz",
    "https://www.yu.edu/katz/about",
    "https://www.yu.edu/katz/programs",
    "https://www.yu.edu/katz/faculty",
    "https://www.yu.edu/katz/admissions",
    "https://www.yu.edu/katz/research",
    "https://www.yu.edu/katz/events",
    "https://www.yu.edu/katz/news",
    "https://www.yu.edu/katz/student-life",
    "https://www.yu.edu/katz/career-services",
    "https://www.yu.edu/katz/alumni",
    # AI program (was manually added in notebook)
    "https://www.yu.edu/katz/ai",
    "https://www.yu.edu/katz/ai/curriculum",
    "https://www.yu.edu/katz/ai/admissions",
    "https://www.yu.edu/katz/ai/faculty",
    # Other programs
    "https://www.yu.edu/katz/cs",
    "https://www.yu.edu/katz/cs/curriculum",
    "https://www.yu.edu/katz/cybersecurity",
    "https://www.yu.edu/katz/cybersecurity/curriculum",
    "https://www.yu.edu/katz/data-analytics",
    "https://www.yu.edu/katz/data-analytics/curriculum",
    "https://www.yu.edu/katz/math",
    "https://www.yu.edu/katz/biotechnology",
    "https://www.yu.edu/katz/health-informatics",
    "https://www.yu.edu/katz/speech-language-pathology",
    "https://www.yu.edu/katz/occupational-therapy",
    "https://www.yu.edu/katz/physician-assistant",
    # Admissions detail
    "https://www.yu.edu/katz/admissions/how-to-apply",
    "https://www.yu.edu/katz/admissions/tuition-financial-aid",
    "https://www.yu.edu/katz/admissions/scholarships",
    "https://www.yu.edu/katz/admissions/international-students",
]

# ── QA dataset URL from original notebook ─────────────────────
QA_DATASET_URL = (
    "https://raw.githubusercontent.com/sahilkumar15/Research_Work/"
    "main/New_Train_QA_Pairs.csv"
)


def _parse_sitemap(url: str) -> list[str]:
    """Parse sitemap XML and return all <loc> URLs (handles sitemap index too)."""
    urls = []
    try:
        resp = requests.get(url, timeout=REQUEST_TO, headers=HEADERS)
        if resp.status_code != 200:
            return urls

        # Try XML parse
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return urls

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Sitemap index → recurse
        for child_sm in root.findall("sm:sitemap/sm:loc", ns):
            urls.extend(_parse_sitemap(child_sm.text.strip()))

        # Regular sitemap
        for loc in root.findall("sm:url/sm:loc", ns):
            urls.append(loc.text.strip())

        # Also try BeautifulSoup XML (same method as original notebook)
        if not urls:
            soup = BeautifulSoup(resp.content, "xml")
            urls = [loc.text.strip() for loc in soup.find_all("loc")]

        print(f"[Crawler] Sitemap {url} → {len(urls)} URLs")
    except Exception as e:
        print(f"[Crawler] Sitemap failed {url}: {e}")
    return urls


def _collect_urls() -> list[str]:
    """
    Collect all Katz-relevant URLs.
    Strategy (matches original notebook):
      1. Parse all sitemaps → filter to /katz URLs
      2. Add manually curated extra URLs
      3. Deduplicate
    """
    all_sitemap_urls: list[str] = []
    for sm_url in SITEMAP_URLS:
        all_sitemap_urls.extend(_parse_sitemap(sm_url))

    # Filter to Katz URLs (original notebook used all 1262 but we focus on Katz)
    katz_urls = [
        u for u in all_sitemap_urls
        if "/katz" in u.lower()
        and not u.endswith((".pdf", ".jpg", ".png", ".zip", ".docx", ".xlsx"))
    ]
    print(f"[Crawler] {len(katz_urls)} Katz URLs from sitemaps")

    # Merge with extras (deduplicated)
    seen  = set(katz_urls)
    final = list(katz_urls)
    for u in EXTRA_KATZ_URLS:
        if u not in seen:
            final.append(u)
            seen.add(u)

    print(f"[Crawler] {len(final)} total URLs after adding extras")
    return final


def _fetch_page(url: str) -> Optional[Document]:
    """
    Fetch a single URL and return a LangChain Document.
    Content extraction strategy: prefer <main>/<article>, strip boilerplate.
    (Improved over original UnstructuredURLLoader which was noisy)
    """
    try:
        resp = requests.get(url, timeout=REQUEST_TO, headers=HEADERS, allow_redirects=True)
        if resp.status_code != 200:
            return None
        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Strip boilerplate (same intent as original's text cleaning)
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "noscript", "iframe"]):
            tag.decompose()

        # Prefer semantic main content areas
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="main-content")
            or soup.find(id="content")
            or soup.find(class_="main-content")
            or soup.find(class_="field--type-text-with-summary")
            or soup.body
        )

        text = " ".join(
            (main or soup).get_text(separator=" ", strip=True).split()
        )

        if len(text) < 100:   # skip near-empty pages
            return None

        # Extract page title for metadata
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        return Document(
            page_content=text[:MAX_CONTENT],
            metadata={
                "source": url,
                "title":  title.strip()[:200],
                "length": len(text),
            },
        )
    except Exception as e:
        print(f"[Crawler] Failed {url}: {e}")
        return None


def _fetch_qa_dataset() -> list[Document]:
    """
    Load the QA pairs from GitHub (same URL as original notebook).
    Converts each QA pair into a Document for indexing.
    This was one of the key reasons the original performed well on evaluation.
    """
    docs = []
    try:
        import pandas as pd
        resp = requests.get(QA_DATASET_URL, timeout=15, headers=HEADERS)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            for _, row in df.iterrows():
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if q and a:
                    docs.append(Document(
                        page_content=f"Q: {q}\nA: {a}",
                        metadata={
                            "source": QA_DATASET_URL,
                            "title":  "KatzBot QA Dataset",
                            "type":   "qa_pair",
                        },
                    ))
            print(f"[Crawler] Loaded {len(docs)} QA pairs from GitHub dataset")
    except Exception as e:
        print(f"[Crawler] QA dataset load failed: {e}")
    return docs


def crawl_katz(force_refresh: bool = False) -> list[Document]:
    """
    Main entry point. Returns list of LangChain Documents.
    
    Cache strategy:
      - Load from JSON cache if < 7 days old (same as original pickle)
      - Rebuild with parallel fetching if expired or forced
      - Always include static QA dataset
    
    Args:
        force_refresh: bypass cache and re-crawl
    """
    # ── Check cache ───────────────────────────────────────────
    if not force_refresh and CACHE_FILE.exists():
        try:
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < CACHE_TTL:
                with open(CACHE_FILE) as f:
                    cached = json.load(f)
                if len(cached) >= 20:
                    print(f"[Crawler] Cache hit — {len(cached)} pages "
                          f"({age/3600:.1f}h old)")
                    return [
                        Document(
                            page_content=p["content"],
                            metadata=p.get("metadata", {"source": p.get("url", "")}),
                        )
                        for p in cached
                    ]
        except Exception as e:
            print(f"[Crawler] Cache read error: {e}")

    # ── Collect URLs ──────────────────────────────────────────
    urls = _collect_urls()
    print(f"[Crawler] Fetching {len(urls)} pages with {MAX_WORKERS} workers...")

    # ── Parallel fetch (original was sequential → much faster now) ─
    documents: list[Document] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(_fetch_page, url): url for url in urls}
        for i, future in enumerate(as_completed(future_to_url), 1):
            url = future_to_url[future]
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                    if i % 10 == 0:
                        print(f"[Crawler] Progress: {i}/{len(urls)} "
                              f"({len(documents)} ok, {failed} failed)")
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"[Crawler] Error {url}: {e}")

    print(f"[Crawler] Fetched {len(documents)} pages ({failed} failed)")

    # ── Add QA dataset (key to original's performance) ────────
    qa_docs = _fetch_qa_dataset()
    documents.extend(qa_docs)

    # ── Save cache ────────────────────────────────────────────
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache_data = [
            {
                "content":  doc.page_content,
                "metadata": doc.metadata,
                "url":      doc.metadata.get("source", ""),
            }
            for doc in documents
        ]
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"[Crawler] Cache saved ({len(documents)} docs → {CACHE_FILE})")
    except Exception as e:
        print(f"[Crawler] Cache save failed: {e}")

    return documents


if __name__ == "__main__":
    docs = crawl_katz(force_refresh=True)
    print(f"\nTotal documents: {len(docs)}")
    print(f"Sample: {docs[0].page_content[:200]}")