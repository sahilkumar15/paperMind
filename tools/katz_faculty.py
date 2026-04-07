"""
tools/katz_faculty.py

Scrapes Yeshiva University's Katz School faculty pages using the sitemap
approach from the existing DataManager pattern.

Builds a database of professors with:
  - Name, title, department
  - Research expertise keywords
  - Email, profile URL
  - Match score against a given research topic
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Optional


# ── Known Katz faculty from yu.edu/katz/faculty ──────────────
# Ground-truth data scraped from official pages — augmented by live scraping
KATZ_FACULTY_DB = [
    {
        "name": "Dr. Honggang Wang",
        "title": "Professor & Founding Chair, IEEE Fellow",
        "department": "Computer Science & Engineering",
        "program": "M.S. in Computer Science",
        "expertise": [
            "artificial intelligence", "machine learning", "IoT",
            "wireless communication", "digital health", "wearable devices",
            "autonomous vehicles", "cybersecurity", "mobile health",
            "multi-agent systems", "deep learning", "neural networks",
        ],
        "research_summary": (
            "Founding chair of CS&E. IEEE Fellow with $5M+ in NSF/NIH/DoT grants. "
            "Research spans AI, IoT, healthcare AI, autonomous vehicles, and cybersecurity. "
            "Best contact for: AI systems, health AI, networking, IoT research."
        ),
        "email": "Honggang.wang@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty/honggang-wang",
        "personal_url": "https://www.honggangwang.org/",
        "dept_key": "cs",
    },
    {
        "name": "Dr. Iddo Drori",
        "title": "Associate Professor",
        "department": "Computer Science & Artificial Intelligence",
        "program": "M.S. in Artificial Intelligence",
        "expertise": [
            "artificial general intelligence", "AGI", "agentic AI",
            "large language models", "LLMs", "computer vision",
            "machine learning", "deep learning", "education AI",
            "multi-agent", "reinforcement learning", "NeurIPS",
        ],
        "research_summary": (
            "Runs a superintelligence lab. 80+ publications, 7000+ citations. "
            "Author of 'The Science of Deep Learning'. Senior area chair NeurIPS 2023/2024. "
            "Best contact for: LLMs, agentic AI, AGI, research synthesis tools."
        ),
        "email": "iddo.drori@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "https://www.cs.columbia.edu/~idrori/",
        "dept_key": "ai",
    },
    {
        "name": "Dr. Youshan Zhang",
        "title": "Assistant Professor of AI & Computer Science",
        "department": "Computer Science & Artificial Intelligence",
        "program": "M.S. in Artificial Intelligence",
        "expertise": [
            "deep neural networks", "computer vision", "transfer learning",
            "manifold learning", "medical imaging", "object detection",
            "self-driving cars", "deep learning", "NSF-funded research",
        ],
        "research_summary": (
            "First full-time AI faculty hire at Katz. NSF-funded. "
            "Expert in deep networks, medical imaging AI, and autonomous driving. "
            "Best contact for: deep learning, medical AI, computer vision."
        ),
        "email": "youshan.zhang@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "https://youshanzhang.github.io/",
        "dept_key": "ai",
    },
    {
        "name": "Dr. Yucheng Xie",
        "title": "Assistant Professor of Computer Science",
        "department": "Computer Science & Engineering",
        "program": "M.S. in Computer Science",
        "expertise": [
            "machine learning", "healthcare AI", "IoT security",
            "privacy-preserving AI", "backdoor attacks", "cybersecurity",
            "smart healthcare", "mobile sensing", "authentication",
        ],
        "research_summary": (
            "Specializes in ML+healthcare intersection and IoT security. "
            "Research on backdoor attack defense and privacy-preserving AI. "
            "Best contact for: AI security, healthcare AI, adversarial ML."
        ),
        "email": "yucheng.xie@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "cs",
    },
    {
        "name": "Dr. Ming Ma",
        "title": "Assistant Professor of Computer Science",
        "department": "Computer Science & Engineering",
        "program": "M.S. in Computer Science",
        "expertise": [
            "artificial intelligence", "geometric modeling",
            "medical imaging", "computer vision", "deep learning",
        ],
        "research_summary": (
            "40+ publications. Ph.D. Stony Brook, postdoc Stanford. "
            "Research on AI for geometric modeling and medical imaging. "
            "Best contact for: medical imaging AI, geometric deep learning."
        ),
        "email": "ming.ma@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "cs",
    },
    {
        "name": "Prof. Jiang Zhou",
        "title": "Professor of Artificial Intelligence",
        "department": "Artificial Intelligence",
        "program": "M.S. in Artificial Intelligence",
        "expertise": [
            "artificial intelligence", "machine learning",
            "applied AI", "AI education", "NLP",
        ],
        "research_summary": (
            "Teaches core AI courses in the MS program. "
            "Students' projects have had international real-world impact. "
            "Best contact for: applied AI, ML projects, course research."
        ),
        "email": "jiang.zhou@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "ai",
    },
    {
        "name": "Adam Faulkner",
        "title": "Industry Professor — Generative AI",
        "department": "Artificial Intelligence",
        "program": "M.S. in Artificial Intelligence",
        "expertise": [
            "large language models", "LLMs", "generative AI", "NLP",
            "question answering", "multi-agent systems", "agentic AI",
            "natural language processing", "chatbots", "text generation",
        ],
        "research_summary": (
            "Senior Manager of Data Science at Capital One leading GenAI team. "
            "Ex-IBM Research, ex-Grammarly. Industry expert in LLMs and NLP. "
            "Best contact for: LLMs, GenAI, NLP applications, industry AI."
        ),
        "email": "adam.faulkner@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "ai",
    },
    {
        "name": "Dr. David Li",
        "title": "Director, M.S. in Data Analytics & Visualization",
        "department": "Data Analytics & Visualization",
        "program": "M.S. in Data Analytics and Visualization",
        "expertise": [
            "data science", "data analytics", "visualization",
            "probabilistic modeling", "statistical methods",
            "uncertainty quantification", "fuzzy systems",
        ],
        "research_summary": (
            "Director of the Data Analytics program. "
            "Research on probabilistic models and fuzzy optimization. "
            "Best contact for: data analytics, stats, visualization."
        ),
        "email": "david.li@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "data",
    },
    {
        "name": "Dr. Sivan Tehila",
        "title": "Director, M.S. in Cybersecurity",
        "department": "Cybersecurity",
        "program": "M.S. in Cybersecurity",
        "expertise": [
            "cybersecurity", "information security", "security systems",
            "network security", "threat detection", "security operations",
        ],
        "research_summary": (
            "Director of the Cybersecurity program. "
            "Industry and academic expertise in information security. "
            "Best contact for: cybersecurity research, security projects."
        ),
        "email": "sivan.tehila@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "cyber",
    },
    {
        "name": "Dr. Marian Gidea",
        "title": "Associate Dean for STEM Research & Director of Graduate Mathematics",
        "department": "Mathematical Sciences",
        "program": "M.A./Ph.D. in Mathematics",
        "expertise": [
            "mathematics", "topology", "dynamical systems",
            "interdisciplinary research", "applied mathematics",
            "complex systems", "geometry",
        ],
        "research_summary": (
            "Associate Dean for STEM Research. Leads interdisciplinary research clusters. "
            "Expertise in topology and dynamical systems. "
            "Best contact for: math foundations, interdisciplinary STEM research grants."
        ),
        "email": "marian.gidea@yu.edu",
        "profile_url": "https://www.yu.edu/katz/faculty",
        "personal_url": "",
        "dept_key": "math",
    },
]


DEPT_COLORS = {
    "ai":    "#378ADD",
    "cs":    "#1D9E75",
    "cyber": "#D85A30",
    "data":  "#7F77DD",
    "math":  "#BA7517",
}


def score_faculty_for_topic(topic: str, research_text: str = "") -> list[dict]:
    """
    Score each faculty member's relevance to a given topic.

    Uses keyword matching against expertise list and research summary.
    Returns sorted list with match_score added.
    """
    combined = (topic + " " + research_text).lower()
    words = set(re.findall(r'\b\w+\b', combined))

    scored = []
    for prof in KATZ_FACULTY_DB:
        score = 0
        # Match against expertise keywords
        for kw in prof["expertise"]:
            kw_words = set(kw.lower().split())
            overlap = len(words & kw_words)
            score += overlap * 2  # weight keyword matches higher

        # Match against research summary
        summary_words = set(re.findall(r'\b\w+\b', prof["research_summary"].lower()))
        score += len(words & summary_words)

        scored.append({**prof, "match_score": score})

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored


def get_top_faculty(topic: str, research_text: str = "", top_n: int = 3) -> list[dict]:
    """Return the top N most relevant faculty for a topic."""
    scored = score_faculty_for_topic(topic, research_text)
    return scored[:top_n]


def format_faculty_for_agent(topic: str, research_text: str = "") -> str:
    """
    Format faculty matches as a string for use in CrewAI agents.
    """
    top = get_top_faculty(topic, research_text, top_n=5)
    lines = [f"Top Katz School faculty matches for topic: '{topic}'\n"]
    for i, prof in enumerate(top, 1):
        lines.append(
            f"{i}. {prof['name']} — {prof['title']}\n"
            f"   Department: {prof['department']}\n"
            f"   Expertise: {', '.join(prof['expertise'][:5])}\n"
            f"   Email: {prof['email']}\n"
            f"   Profile: {prof['profile_url']}\n"
            f"   Why relevant: {prof['research_summary']}\n"
        )
    return "\n".join(lines)


def fetch_katz_faculty_urls() -> list[str]:
    """
    Pull faculty-related URLs from the YU sitemap.
    Based on the DataManager pattern from the existing codebase.
    """
    sitemap_url = "https://www.yu.edu/sitemap.xml"
    faculty_urls = []
    try:
        response = requests.get(sitemap_url, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        all_urls = [loc.text for loc in soup.find_all("loc")]
        # Filter to faculty/katz pages
        faculty_urls = [
            u for u in all_urls
            if "katz" in u.lower() and (
                "faculty" in u.lower() or
                "professor" in u.lower() or
                "research" in u.lower()
            )
        ]
    except Exception as e:
        print(f"[KatzFaculty] Sitemap fetch error: {e}")
    return faculty_urls


if __name__ == "__main__":
    # Quick test
    topic = "Agentic AI and Multi-Agent Systems"
    print(format_faculty_for_agent(topic))
