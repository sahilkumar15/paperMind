"""
katzbot/crawler.py
===================
Crawler for KatzBot / PaperMind.

Features:
- Parses YU sitemap(s)
- Can crawl only Katz pages or the full YU sitemap
- Handles JS-heavy pages with BeautifulSoup fallback parsing
- Injects static Katz content
- Optional QA dataset
- Saves crawled docs to a scope-specific cache
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

CACHE_TTL = 7 * 24 * 3600
MAX_WORKERS = 6
REQUEST_TO = 15
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

QA_DATASET_URL = (
    "https://raw.githubusercontent.com/sahilkumar15/Research_Work/"
    "main/New_Train_QA_Pairs.csv"
)

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

Ph.D. in Computer Science
- Research-focused doctoral program
- Areas: AI/ML, Cybersecurity, Networks, Quantum Computing
- Contact: Prof. Honggang Wang (hwang@yu.edu)
- URL: https://www.yu.edu/katz/computer-science-phd

M.S. in Cybersecurity
- Credits: 30 | Duration: 1-2 years
- Online option available
- URL: https://www.yu.edu/katz/cybersecurity

M.S. in Data Analytics and Visualization
- Credits: 30 | Duration: 1-2 years
- Program Lead: Prof. Shira Scheindlin (sscheindlin@yu.edu)
- URL: https://www.yu.edu/katz/data-analytics

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

Tuition & Financial Aid: https://www.yu.edu/katz/admissions/tuition-financial-aid
- Tuition: approximately $1,500-$1,700 per credit hour
- 30-credit program = approximately $45,000-$51,000 total
- Merit scholarships available
- Graduate assistantships available
- Approximately 80-85% of students receive some form of financial aid
- FAFSA code: 002903
"""


def _crawl_scope() -> str:
    return os.getenv("KATZBOT_CRAWL_SCOPE", "all").strip().lower()


def _cache_file() -> Path:
    scope = _crawl_scope()
    return Path(__file__).parent / f"pages_cache_{scope}.json"


def _max_urls() -> int:
    raw = os.getenv("KATZBOT_MAX_URLS", "0").strip()
    try:
        return int(raw)
    except Exception:
        return 0


def _want_qa_dataset() -> bool:
    return os.getenv("KATZBOT_INCLUDE_QA_DATASET", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _normalize_urls(urls: list[str]) -> list[str]:
    seen = set()
    out = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        if u.endswith("/"):
            u = u[:-1]
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _parse_sitemap(url: str) -> list[str]:
    urls = []
    try:
        resp = requests.get(url, timeout=REQUEST_TO, headers=HEADERS)
        if resp.status_code != 200:
            return urls

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            root = None

        if root is not None:
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            for child in root.findall("sm:sitemap/sm:loc", ns):
                child_url = (child.text or "").strip()
                if child_url:
                    urls.extend(_parse_sitemap(child_url))

            for loc in root.findall("sm:url/sm:loc", ns):
                loc_url = (loc.text or "").strip()
                if loc_url:
                    urls.append(loc_url)

        if not urls:
            soup = BeautifulSoup(resp.content, "xml")
            for loc in soup.find_all("loc"):
                txt = loc.get_text(strip=True)
                if txt:
                    urls.append(txt)

        urls = _normalize_urls(urls)
        print(f"[Crawler] Sitemap {url.split('/')[-1]} → {len(urls)} URLs")
    except Exception as e:
        print(f"[Crawler] Sitemap error ({url}): {e}")

    return urls


def _collect_urls() -> list[str]:
    all_sitemap = []
    for sm in SITEMAP_URLS:
        all_sitemap.extend(_parse_sitemap(sm))

    all_sitemap = _normalize_urls(all_sitemap)

    filtered = [
        u for u in all_sitemap
        if not u.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".zip", ".docx", ".xlsx", ".pptx"))
    ]

    scope = _crawl_scope()
    if scope == "all":
        final = filtered
        print(f"[Crawler] Using ALL sitemap URLs → {len(final)} URLs")
    else:
        katz_urls = [u for u in filtered if "/katz" in u.lower()]
        print(f"[Crawler] {len(katz_urls)} Katz URLs from sitemaps")
        final = katz_urls

        seen = set(final)
        for u in EXTRA_KATZ_URLS:
            if u not in seen:
                final.append(u)
                seen.add(u)

    limit = _max_urls()
    if limit > 0:
        final = final[:limit]
        print(f"[Crawler] URL cap applied → {len(final)} URLs")

    print(f"[Crawler] {len(final)} total URLs")
    return final


def _fetch_page(url: str) -> Optional[Document]:
    try:
        resp = requests.get(url, timeout=REQUEST_TO, headers=HEADERS, allow_redirects=True)
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return None

        text = ""
        for parser in ["lxml", "html.parser"]:
            try:
                soup = BeautifulSoup(resp.text, parser)
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "meta"]):
                    tag.decompose()

                main = (
                    soup.find("main")
                    or soup.find(id="main-content")
                    or soup.find(class_="main-content")
                    or soup.find(class_="field--type-text-with-summary")
                    or soup.find("article")
                    or soup.find(class_="content")
                    or soup.body
                )

                text = " ".join((main or soup).get_text(separator=" ", strip=True).split())
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
            metadata={
                "source": url,
                "title": title,
                "type": "web_page",
            },
        )
    except Exception:
        return None


def _fetch_qa_dataset() -> list:
    if not _want_qa_dataset():
        print("[Crawler] Notebook QA dataset disabled")
        return []

    docs = []
    try:
        import pandas as pd
        from io import StringIO

        resp = requests.get(QA_DATASET_URL, timeout=20, headers=HEADERS)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            limit = int(os.getenv("KATZBOT_QA_LIMIT", "300"))

            columns = {c.lower(): c for c in df.columns}
            q_col = columns.get("question") or columns.get("q")
            a_col = columns.get("answer") or columns.get("a")

            if not q_col or not a_col:
                print("[Crawler] QA dataset missing expected question/answer columns")
                return []

            for _, row in df.head(limit).iterrows():
                q = str(row.get(q_col, "")).strip()
                a = str(row.get(a_col, "")).strip()
                if q and a and len(q) > 5 and len(a) > 5:
                    docs.append(Document(
                        page_content=f"Q: {q}\nA: {a}",
                        metadata={
                            "source": QA_DATASET_URL,
                            "title": "KatzBot QA Pairs",
                            "type": "qa_pair",
                        },
                    ))

            print(f"[Crawler] Loaded {len(docs)} QA pairs from GitHub")
    except Exception as e:
        print(f"[Crawler] QA dataset failed: {e}")

    return docs


def crawl_katz(force_refresh: bool = False) -> list:
    cache_file = _cache_file()

    if not force_refresh and cache_file.exists():
        try:
            age = time.time() - cache_file.stat().st_mtime
            if age < CACHE_TTL:
                with open(cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                if len(cached) >= 5:
                    print(f"[Crawler] Cache hit — {len(cached)} docs ({age/3600:.1f}h old)")
                    return [
                        Document(
                            page_content=p["content"],
                            metadata=p.get("metadata", {"source": p.get("url", "")}),
                        )
                        for p in cached
                    ]
        except Exception as e:
            print(f"[Crawler] Cache read failed: {e}")

    urls = _collect_urls()
    print(f"[Crawler] Fetching {len(urls)} pages ({MAX_WORKERS} workers)…")

    documents = []
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
                    print(f"[Crawler] {i}/{len(urls)} ({len(documents)} ok, {failed} failed)")
            except Exception:
                failed += 1

    print(f"[Crawler] Fetched {len(documents)} pages ({failed} failed/empty)")

    documents.append(Document(
        page_content=STATIC_KATZ_CONTENT,
        metadata={
            "source": "https://www.yu.edu/katz",
            "title": "Katz School Programs and Admissions (Static)",
            "type": "static_content",
        },
    ))
    print("[Crawler] Injected static Katz program content")

    qa_docs = _fetch_qa_dataset()
    documents.extend(qa_docs)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "url": doc.metadata.get("source", ""),
            }
            for doc in documents
        ]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"[Crawler] Saved {len(documents)} docs to cache → {cache_file}")
    except Exception as e:
        print(f"[Crawler] Cache save failed: {e}")

    return documents