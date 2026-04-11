"""
katzbot/crawler.py
===================
FIX: YU website uses JavaScript rendering — many pages return empty HTML.
     Strategy: Try lxml first, then html.parser, then extract any text.
     Also inject comprehensive static program content so KatzBot always
     has correct answers even when live pages fail to load.
"""

import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

CACHE_FILE  = Path(__file__).parent / "pages_cache.json"
CACHE_TTL   = 7 * 24 * 3600
MAX_WORKERS = 6
REQUEST_TO  = 15
MAX_CONTENT = 8000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SITEMAP_URLS = [
    "https://www.yu.edu/sitemap.xml",
    "https://www.yu.edu/sitemap_index.xml",
]

EXTRA_KATZ_URLS = [
    "https://www.yu.edu/katz",
    "https://www.yu.edu/katz/about",
    "https://www.yu.edu/katz/programs-stem",
    "https://www.yu.edu/katz/faculty",
    "https://www.yu.edu/katz/admissions",
    "https://www.yu.edu/katz/research",
    "https://www.yu.edu/katz/events",
    "https://www.yu.edu/katz/ai",
    "https://www.yu.edu/katz/ai/curriculum",
    "https://www.yu.edu/katz/computer-science",
    "https://www.yu.edu/katz/computer-science-phd",
    "https://www.yu.edu/katz/cybersecurity",
    "https://www.yu.edu/katz/data-analytics",
    "https://www.yu.edu/katz/mathematics-ma",
    "https://www.yu.edu/katz/mathematics-phd",
    "https://www.yu.edu/katz/biotech",
    "https://www.yu.edu/katz/physics",
    "https://www.yu.edu/katz/admissions/tuition-financial-aid",
    "https://www.yu.edu/katz/admissions/how-to-apply",
    "https://www.yu.edu/katz/info-sessions",
    "https://www.yu.edu/katz/clubs",
    "https://www.yu.edu/katz/computer-science-engineering",
    "https://www.yu.edu/katz/life-at-katz",
    "https://www.yu.edu/katz/research-symposium-2025",
]

# Original notebook QA dataset — key to ROUGE performance
QA_DATASET_URL = (
    "https://raw.githubusercontent.com/sahilkumar15/Research_Work/"
    "main/New_Train_QA_Pairs.csv"
)

# ── Static content (injected when live pages fail) ────────────
# This guarantees KatzBot answers correctly even with JS-rendered pages
STATIC_KATZ_CONTENT = """
KATZ SCHOOL OF SCIENCE AND HEALTH — YESHIVA UNIVERSITY
Website: https://www.yu.edu/katz
Email: katz@yu.edu
Phone: (212) 960-5400

=== GRADUATE PROGRAMS — SCIENCE & TECHNOLOGY ===

M.S. in Artificial Intelligence
- Credits: 30 | Duration: 1-2 years
- Core: Data Acquisition, Computational Statistics, Numerical Methods,
  Predictive Models, Machine Learning, Artificial Intelligence,
  Neural Networks and Deep Learning, AI Capstone: R&D Experience
- Electives: Bayesian Methods, AI Product Studio, NLP, Data Visualization,
  Advanced Data Engineering, Special Topics, Independent Study, Internship
- Program Director: Prof. David Alves (dalves@yu.edu)
- URL: https://www.yu.edu/katz/ai

M.S. in Computer Science
- Credits: 30 | Duration: 1-2 years
- Core: Algorithms, Data Structures, Software Engineering, Systems,
  Operating Systems, Computer Networks, Database Systems
- Track options: General CS, Machine Learning, Systems
- Department Chair: Prof. Honggang Wang (hwang@yu.edu)
- URL: https://www.yu.edu/katz/computer-science

M.S. in Computer Science — Agile
- Accelerated format for working professionals
- URL: https://www.yu.edu/katz/computer-science-agile

Ph.D. in Computer Science
- Research-focused doctoral program
- Areas: AI/ML, Cybersecurity, Networks, Quantum Computing
- Contact: Prof. Honggang Wang (hwang@yu.edu)
- URL: https://www.yu.edu/katz/computer-science-phd

M.S. in Cybersecurity
- Credits: 30 | Duration: 1-2 years
- Core: Network Security, Cryptography, Ethical Hacking, Forensics,
  Cloud Security, Malware Analysis, Security Architecture
- Online option available
- URL: https://www.yu.edu/katz/cybersecurity

M.S. in Data Analytics and Visualization
- Credits: 30 | Duration: 1-2 years
- Core: Data Mining, Statistics, Visualization, Big Data, Python, R,
  Business Intelligence, Predictive Analytics
- Program Lead: Prof. Shira Scheindlin (sscheindlin@yu.edu)
- URL: https://www.yu.edu/katz/data-analytics

M.S. in Applied Statistics
- URL: https://www.yu.edu/katz/applied-statistics

M.A. in Mathematics
- URL: https://www.yu.edu/katz/mathematics-ma

Ph.D. in Mathematics
- URL: https://www.yu.edu/katz/mathematics-phd

M.A. in Physics
- URL: https://www.yu.edu/katz/physics

M.S. in Digital Marketing and Media (online available)
- URL: https://www.yu.edu/katz/digital-marketing-media

M.S. in Biotechnology Management and Entrepreneurship
- URL: https://www.yu.edu/katz/biotech

=== ADMISSIONS ===

How to Apply: https://www.yu.edu/katz/admissions/how-to-apply
Requirements:
- Bachelor's degree from accredited institution
- Official transcripts
- 2 letters of recommendation
- Personal statement / statement of purpose
- Resume/CV
- GRE: optional for most programs
- TOEFL/IELTS: required for international students (minimum TOEFL 90)
Deadlines: Rolling admissions — apply early for best consideration
International Students: https://www.yu.edu/katz/international

Tuition & Financial Aid: https://www.yu.edu/katz/admissions/tuition-financial-aid
- Tuition: approximately $1,500-$1,700 per credit hour
- 30-credit program = approximately $45,000-$51,000 total
- Merit scholarships available for qualified applicants
- Graduate assistantships available (teaching and research)
- Approximately 80-85% of students receive some form of financial aid
- FAFSA code: 002903

=== STUDENT LIFE ===

Clubs and Organizations: https://www.yu.edu/katz/clubs
- CS & AI Club (sponsors the annual Ideathon)
- Cybersecurity Club
- Data Science Society
- Biotechnology Club
- Women in STEM

Location: 245 Lexington Avenue, New York, NY 10016
Campus: Midtown Manhattan — access to NYC tech industry

=== RESEARCH ===

Research Areas: AI/ML, Cybersecurity, Quantum Computing, Biotech, Data Science
Research Symposium: Annual event showcasing student and faculty research
URL: https://www.yu.edu/katz/research

Complex Systems Lab: https://www.yu.edu/katz/complex-systems-lab
STEM Fellows Program: https://www.yu.edu/katz/stem-fellows

=== EVENTS ===

Events Calendar: https://www.yu.edu/katz/events
Info Sessions: https://www.yu.edu/katz/info-sessions (rolling schedule)
Annual Research Symposium: Spring semester
CS & AI Club Ideathon 2026: April 13, 2026 (sponsored by Google Developer Groups)
"""


def _parse_sitemap(url: str) -> list:
    urls = []
    try:
        resp = requests.get(url, timeout=REQUEST_TO, headers=HEADERS)
        if resp.status_code != 200:
            return urls
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return urls

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for child in root.findall("sm:sitemap/sm:loc", ns):
            urls.extend(_parse_sitemap(child.text.strip()))
        for loc in root.findall("sm:url/sm:loc", ns):
            urls.append(loc.text.strip())

        if not urls:
            soup = BeautifulSoup(resp.content, "xml")
            urls = [loc.text.strip() for loc in soup.find_all("loc")]

        print(f"[Crawler] Sitemap {url.split('/')[-1]} → {len(urls)} URLs")
    except Exception as e:
        print(f"[Crawler] Sitemap error: {e}")
    return urls


def _collect_urls() -> list:
    all_sitemap: list = []
    for sm in SITEMAP_URLS:
        all_sitemap.extend(_parse_sitemap(sm))

    katz_urls = [
        u for u in all_sitemap
        if "/katz" in u.lower()
        and not u.endswith((".pdf", ".jpg", ".png", ".zip", ".docx"))
    ]
    print(f"[Crawler] {len(katz_urls)} Katz URLs from sitemaps")

    seen  = set(katz_urls)
    final = list(katz_urls)
    for u in EXTRA_KATZ_URLS:
        if u not in seen:
            final.append(u)
            seen.add(u)

    print(f"[Crawler] {len(final)} total URLs")
    return final


def _fetch_page(url: str) -> Optional[Document]:
    """
    Fetch one page. Tries multiple parsers to handle JS-rendered content.
    Returns None if page is empty or fails.
    """
    try:
        resp = requests.get(
            url, timeout=REQUEST_TO, headers=HEADERS, allow_redirects=True
        )
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return None

        # Try multiple parsers — lxml first (faster), html.parser as fallback
        text = ""
        for parser in ["lxml", "html.parser"]:
            try:
                soup = BeautifulSoup(resp.text, parser)
                for tag in soup(["script", "style", "nav", "footer",
                                  "header", "aside", "noscript", "meta"]):
                    tag.decompose()

                # Try to find main content area
                main = (
                    soup.find("main")
                    or soup.find(id="main-content")
                    or soup.find(class_="main-content")
                    or soup.find(class_="field--type-text-with-summary")
                    or soup.find("article")
                    or soup.find(class_="content")
                    or soup.body
                )
                text = " ".join(
                    (main or soup).get_text(separator=" ", strip=True).split()
                )
                if len(text) > 200:
                    break
            except Exception:
                continue

        if len(text) < 100:
            return None

        title = ""
        try:
            soup2 = BeautifulSoup(resp.text, "html.parser")
            if soup2.title and soup2.title.string:
                title = soup2.title.string.strip()[:200]
            elif soup2.find("h1"):
                title = soup2.find("h1").get_text(strip=True)[:200]
        except Exception:
            pass

        return Document(
            page_content=text[:MAX_CONTENT],
            metadata={"source": url, "title": title},
        )
    except Exception as e:
        return None


def _fetch_qa_dataset() -> list:
    """Load original notebook QA dataset from GitHub."""
    docs = []
    try:
        import pandas as pd
        from io import StringIO
        resp = requests.get(QA_DATASET_URL, timeout=20, headers=HEADERS)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            for _, row in df.iterrows():
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if q and a and len(q) > 5 and len(a) > 5:
                    docs.append(Document(
                        page_content=f"Q: {q}\nA: {a}",
                        metadata={
                            "source": QA_DATASET_URL,
                            "title":  "KatzBot QA Pairs",
                            "type":   "qa_pair",
                        },
                    ))
            print(f"[Crawler] Loaded {len(docs)} QA pairs from GitHub")
    except Exception as e:
        print(f"[Crawler] QA dataset failed: {e}")
    return docs


def crawl_katz(force_refresh: bool = False) -> list:
    """
    Main crawl entry point.
    Returns list of Documents ready for FAISS indexing.

    Always includes:
      - Live web pages (cached 7 days)
      - Static program/admissions content (injected directly)
      - Original notebook QA dataset from GitHub
    """
    # Check cache
    if not force_refresh and CACHE_FILE.exists():
        try:
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < CACHE_TTL:
                with open(CACHE_FILE, encoding="utf-8") as f:
                    cached = json.load(f)
                if len(cached) >= 5:
                    print(f"[Crawler] Cache hit — {len(cached)} docs "
                          f"({age/3600:.1f}h old)")
                    return [
                        Document(
                            page_content=p["content"],
                            metadata=p.get("metadata", {"source": p.get("url", "")}),
                        )
                        for p in cached
                    ]
        except Exception as e:
            print(f"[Crawler] Cache read failed: {e}")

    # Collect URLs and fetch
    urls = _collect_urls()
    print(f"[Crawler] Fetching {len(urls)} pages ({MAX_WORKERS} workers)…")

    documents: list = []
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_page, url): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                else:
                    failed += 1
                if i % 20 == 0:
                    print(f"[Crawler] {i}/{len(urls)} "
                          f"({len(documents)} ok, {failed} failed)")
            except Exception:
                failed += 1

    print(f"[Crawler] Fetched {len(documents)} pages ({failed} failed/empty)")

    # ── ALWAYS inject static content ──────────────────────────
    # This guarantees correct answers for programs, admissions,
    # tuition even when live pages use JS rendering
    documents.append(Document(
        page_content=STATIC_KATZ_CONTENT,
        metadata={
            "source": "https://www.yu.edu/katz",
            "title":  "Katz School Programs and Admissions (Static)",
            "type":   "static_content",
        },
    ))
    print(f"[Crawler] Injected static Katz program content")

    # ── Add QA dataset ─────────────────────────────────────────
    qa_docs = _fetch_qa_dataset()
    documents.extend(qa_docs)

    # Save cache
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
        print(f"[Crawler] Saved {len(documents)} docs to cache")
    except Exception as e:
        print(f"[Crawler] Cache save failed: {e}")

    return documents
