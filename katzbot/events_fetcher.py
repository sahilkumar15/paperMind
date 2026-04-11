"""
katzbot/events_fetcher.py
==========================
Fetches Katz School + YU events from live pages.
Caches for 6 hours. Falls back to static known events.
"""

import re
import json
import time
import requests
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

EVENTS_CACHE = Path(__file__).parent / "events_cache.json"
CACHE_TTL    = 6 * 3600  # 6 hours

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ScholarMind/1.0)"}

EVENT_URLS = [
    "https://www.yu.edu/katz/events",
    "https://www.yu.edu/events",
    "https://www.yu.edu/graduate/events",
    "https://www.yu.edu/katz/info-sessions",
    "https://www.yu.edu/katz/research-symposium-2025",
]

# Static events — always available as fallback
STATIC_EVENTS = [
    {
        "title":       "Katz School CS & AI Club Ideathon 2026",
        "date":        "April 13, 2026",
        "time":        "11:45 AM EDT",
        "location":    "Online (Devpost)",
        "description": "Innovation challenge for Katz School students and faculty. "
                       "Sponsored by Google Developers Group. Judge: Prof. Honggang Wang.",
        "url":         "https://katz-ideathon-2026.devpost.com",
        "category":    "CS & AI Club",
        "type":        "competition",
    },
    {
        "title":       "Katz School Research Symposium 2025",
        "date":        "Spring 2025",
        "time":        "TBD",
        "location":    "Yeshiva University, New York",
        "description": "Annual symposium showcasing research across AI, CS, "
                       "Cybersecurity, Data Analytics, and Health Sciences.",
        "url":         "https://www.yu.edu/katz/research-symposium-2025",
        "category":    "Research",
        "type":        "symposium",
    },
    {
        "title":       "Katz School Admissions Information Sessions",
        "date":        "Rolling — check website",
        "time":        "Various",
        "location":    "Online and In-Person",
        "description": "Meet faculty and current students for all Katz programs. "
                       "Register at yu.edu/katz/info-sessions.",
        "url":         "https://www.yu.edu/katz/info-sessions",
        "category":    "Admissions",
        "type":        "info_session",
    },
    {
        "title":       "YU Graduate School Networking Events",
        "date":        "See yu.edu/graduate/events",
        "time":        "Various",
        "location":    "Yeshiva University",
        "description": "Networking nights, career fairs, research presentations, "
                       "and faculty talks for graduate students.",
        "url":         "https://www.yu.edu/graduate/events",
        "category":    "Graduate School",
        "type":        "networking",
    },
    {
        "title":       "Katz School Clubs & Organizations Meetings",
        "date":        "Weekly during semester",
        "time":        "Various",
        "location":    "Katz School Campus",
        "description": "CS & AI Club, Cybersecurity Club, Data Science Society, "
                       "and other student organizations. See yu.edu/katz/clubs.",
        "url":         "https://www.yu.edu/katz/clubs",
        "category":    "Student Life",
        "type":        "club",
    },
]


def _parse_html_events(html: str, source_url: str) -> list:
    """Extract events from HTML using multiple strategies."""
    from bs4 import BeautifulSoup
    events = []
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Strategy 1: JSON-LD structured data
        import json as _json
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data  = _json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("Event", "EducationEvent"):
                        loc = item.get("location", {})
                        events.append({
                            "title":       item.get("name", ""),
                            "date":        str(item.get("startDate", "")),
                            "time":        "",
                            "location":    loc.get("name", "") if isinstance(loc, dict)
                                          else str(loc),
                            "description": str(item.get("description", ""))[:300],
                            "url":         item.get("url", source_url),
                            "category":    "Katz School",
                            "type":        "event",
                        })
            except Exception:
                pass

        # Strategy 2: CSS selectors
        for sel in ["article.event", ".event-item", ".views-row",
                    "[class*='event']", ".card"]:
            for card in soup.select(sel)[:10]:
                text = card.get_text(separator=" ", strip=True)
                if len(text) < 20:
                    continue
                h = card.find(["h2","h3","h4"])
                title = h.get_text(strip=True) if h else text[:60]
                te = card.find("time") or card.find(class_=re.compile(r"date|time",re.I))
                date = (te.get("datetime","") or te.get_text(strip=True)) if te else ""
                a = card.find("a", href=True)
                href = a["href"] if a else ""
                url = href if href.startswith("http") else f"https://www.yu.edu{href}"
                if title not in [e["title"] for e in events]:
                    events.append({
                        "title": title[:120], "date": date[:50], "time": "",
                        "location": "Yeshiva University",
                        "description": text[:300],
                        "url": url or source_url,
                        "category": "Katz School", "type": "event",
                    })
    except Exception as e:
        print(f"[Events] Parse error: {e}")
    return events


def fetch_events(force_refresh: bool = False) -> list:
    """Fetch all Katz/YU events. Cached 6 hours. Falls back to static."""
    if not force_refresh and EVENTS_CACHE.exists():
        try:
            age = time.time() - EVENTS_CACHE.stat().st_mtime
            if age < CACHE_TTL:
                with open(EVENTS_CACHE) as f:
                    cached = json.load(f)
                if cached:
                    print(f"[Events] Cache hit — {len(cached)} events")
                    return cached
        except Exception:
            pass

    all_events = []
    seen       = set()

    for url in EVENT_URLS:
        try:
            resp = requests.get(url, timeout=12, headers=HEADERS)
            if resp.status_code == 200:
                evs = _parse_html_events(resp.text, url)
                for ev in evs:
                    t = ev.get("title","").lower().strip()
                    if t and t not in seen and len(t) > 5:
                        seen.add(t)
                        all_events.append(ev)
            time.sleep(0.5)
        except Exception as e:
            print(f"[Events] Failed {url}: {e}")

    # Always add static events
    for ev in STATIC_EVENTS:
        if ev["title"].lower() not in seen:
            seen.add(ev["title"].lower())
            all_events.append(ev)

    print(f"[Events] Total: {len(all_events)} events")

    try:
        EVENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_CACHE, "w") as f:
            json.dump(all_events, f, indent=2)
    except Exception:
        pass

    return all_events


def get_events_documents(events: list = None) -> list:
    """Convert events to Documents for FAISS indexing."""
    if events is None:
        events = fetch_events()

    docs = []
    for ev in events:
        text = (
            f"EVENT: {ev.get('title','')}\n"
            f"Date: {ev.get('date','TBD')}\n"
            f"Time: {ev.get('time','')}\n"
            f"Location: {ev.get('location','')}\n"
            f"Category: {ev.get('category','')}\n"
            f"Description: {ev.get('description','')}\n"
            f"URL: {ev.get('url','')}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source":   ev.get("url", "https://www.yu.edu/katz/events"),
                "title":    ev.get("title",""),
                "type":     "event",
                "category": ev.get("category",""),
            }
        ))

    summary = "KATZ SCHOOL UPCOMING EVENTS:\n\n"
    for ev in events:
        summary += f"• {ev.get('title','')} — {ev.get('date','TBD')} — {ev.get('url','')}\n"
    docs.append(Document(
        page_content=summary,
        metadata={"source": "https://www.yu.edu/katz/events",
                  "title":  "Katz Events Summary", "type": "events_summary"}
    ))
    return docs


def match_events_to_topic(events: list, topic: str, top_k: int = 3) -> list:
    """Find events relevant to a research topic."""
    topic_lower = topic.lower()
    keywords    = [w for w in topic_lower.split() if len(w) > 3]
    boost       = ["ai", "artificial intelligence", "computer science", "cybersecurity",
                   "data", "research", "seminar", "workshop", "symposium"]
    scored = []
    for ev in events:
        score    = 0
        ev_text  = f"{ev.get('title','')} {ev.get('description','')} {ev.get('category','')}".lower()
        for kw in keywords:
            if kw in ev_text:
                score += 2
        for bw in boost:
            if bw in ev_text:
                score += 1
        if score > 0:
            scored.append((score, ev))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ev for _, ev in scored[:top_k]]
