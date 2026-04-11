"""
katzbot/faculty.py
===================
Katz School faculty database + matching utilities.
Faculty are injected as Documents into the FAISS index so KatzBot
can answer "who is X" questions even when the website doesn't load.

Also used by the Faculty Match tab in app.py.
"""

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# ── Katz Faculty Database ──────────────────────────────────────
# Source: yu.edu/katz/faculty (curated April 2026)
KATZ_FACULTY = [
    {
        "name":      "Prof. Honggang Wang",
        "title":     "Professor & Chair, Computer Science",
        "dept":      "Computer Science & Engineering",
        "email":     "hwang@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/honggang-wang",
        "expertise": ["computer networks", "wireless communications", "IoT",
                      "multimedia systems", "cybersecurity", "machine learning",
                      "5G", "network security"],
        "note":      "Department Chair of CS — contact for program structure and research.",
        "courses":   ["Computer Networks", "Wireless Systems", "IoT Security"],
    },
    {
        "name":      "Prof. David Leidner",
        "title":     "Dean, Katz School of Science and Health",
        "dept":      "Administration",
        "email":     "dleidner@yu.edu",
        "profile":   "https://www.yu.edu/katz/about/dean",
        "expertise": ["information systems", "knowledge management", "AI ethics",
                      "technology leadership", "digital transformation"],
        "note":      "Dean of the Katz School — oversees all programs and strategic direction.",
        "courses":   [],
    },
    {
        "name":      "Prof. Barry Burd",
        "title":     "Professor, Computer Science",
        "dept":      "Computer Science & Engineering",
        "email":     "bburd@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/barry-burd",
        "expertise": ["software engineering", "Java", "Android development",
                      "programming languages", "computer science education",
                      "mobile development"],
        "note":      "Author of 'Java For Dummies' and Android textbooks. Great mentor for software projects.",
        "courses":   ["Software Engineering", "Mobile Development", "Java Programming"],
    },
    {
        "name":      "Prof. Mark Hillery",
        "title":     "Professor, Physics & Quantum Computing",
        "dept":      "Computer Science & Engineering",
        "email":     "mhillery@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/mark-hillery",
        "expertise": ["quantum computing", "quantum information", "quantum cryptography",
                      "theoretical physics", "quantum algorithms", "entanglement"],
        "note":      "World-leading quantum computing researcher with 200+ publications.",
        "courses":   ["Quantum Computing", "Quantum Information Theory"],
    },
    {
        "name":      "Prof. Mordechai Guri",
        "title":     "Professor, Cybersecurity",
        "dept":      "Cybersecurity",
        "email":     "mguri@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/mordechai-guri",
        "expertise": ["cybersecurity", "air-gap attacks", "hardware security",
                      "side-channel attacks", "malware", "covert channels",
                      "electromagnetic attacks", "acoustic attacks"],
        "note":      "Known globally for air-gap attack research. Multiple viral security demos.",
        "courses":   ["Cybersecurity", "Hardware Security", "Malware Analysis"],
    },
    {
        "name":      "Prof. Reza Curtmola",
        "title":     "Professor, Cybersecurity",
        "dept":      "Cybersecurity",
        "email":     "rcurtmola@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/reza-curtmola",
        "expertise": ["cybersecurity", "cryptography", "network security",
                      "cloud security", "privacy", "applied cryptography",
                      "searchable encryption", "provable security"],
        "note":      "Expert in applied cryptography and cloud data security.",
        "courses":   ["Applied Cryptography", "Network Security", "Cloud Security"],
    },
    {
        "name":      "Prof. Charles Ying",
        "title":     "Associate Professor, Computer Science",
        "dept":      "Computer Science & Engineering",
        "email":     "cying@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/charles-ying",
        "expertise": ["natural language processing", "text mining", "information retrieval",
                      "machine learning", "large language models", "AI",
                      "computational linguistics"],
        "note":      "NLP and LLM specialist — excellent supervisor for text/language AI projects.",
        "courses":   ["Natural Language Processing", "Machine Learning", "AI"],
    },
    {
        "name":      "Prof. Shira Scheindlin",
        "title":     "Associate Professor, Data Analytics",
        "dept":      "Data Analytics & Visualization",
        "email":     "sscheindlin@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/shira-scheindlin",
        "expertise": ["data analytics", "data visualization", "statistics",
                      "business intelligence", "data science", "R", "Tableau",
                      "predictive analytics"],
        "note":      "Program lead for Data Analytics — strong industry connections.",
        "courses":   ["Data Visualization", "Predictive Analytics", "Business Intelligence"],
    },
    {
        "name":      "Prof. Nelly Levin",
        "title":     "Assistant Professor, Mathematical Sciences",
        "dept":      "Mathematical Sciences",
        "email":     "nlevin@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/nelly-levin",
        "expertise": ["mathematics", "statistics", "probability",
                      "mathematical modeling", "optimization", "analysis",
                      "differential equations"],
        "note":      "Specialises in mathematical modelling and statistical methods.",
        "courses":   ["Mathematical Modeling", "Statistics", "Optimization"],
    },
    {
        "name":      "Prof. David Alves",
        "title":     "Associate Professor & Program Director, M.S. in AI",
        "dept":      "M.S. in Artificial Intelligence",
        "email":     "dalves@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/david-alves",
        "expertise": ["artificial intelligence", "machine learning", "neural networks",
                      "deep learning", "computer vision", "reinforcement learning",
                      "AI applications"],
        "note":      "Program director for the M.S. in AI. First contact for AI admissions.",
        "courses":   ["Machine Learning", "Deep Learning", "AI Capstone"],
    },
]


def match_faculty(topic: str, top_k: int = 3) -> list:
    """
    Match faculty to a topic using keyword scoring.
    Also does direct name matching (e.g. 'who is Honggang Wang').
    """
    topic_lower = topic.lower()
    scored = []

    for f in KATZ_FACULTY:
        score = 0

        # Direct name match (high priority)
        last_name = f["name"].lower().split()[-1]
        full_name = f["name"].lower().replace("prof. ", "")
        if last_name in topic_lower or full_name in topic_lower:
            score += 20

        # Expertise keyword match
        for kw in f["expertise"]:
            if kw.lower() in topic_lower:
                score += 3
            elif any(word in kw.lower() for word in topic_lower.split() if len(word) > 3):
                score += 1

        # Department match
        dept_words = f["dept"].lower().split()
        for word in dept_words:
            if len(word) > 4 and word in topic_lower:
                score += 2

        if score > 0:
            scored.append((score, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_k]]


def get_faculty_documents() -> list:
    """
    Convert faculty records to LangChain Documents for FAISS indexing.
    This ensures KatzBot can answer faculty questions even without website access.
    """
    docs = []
    for f in KATZ_FACULTY:
        # Rich text representation for good embedding recall
        text = (
            f"Faculty Profile:\n"
            f"Name: {f['name']}\n"
            f"Title: {f['title']}\n"
            f"Department: {f['dept']}\n"
            f"Email: {f['email']}\n"
            f"Profile URL: {f['profile']}\n"
            f"Research Expertise: {', '.join(f['expertise'])}\n"
            f"Courses Taught: {', '.join(f.get('courses', []))}\n"
            f"Research Focus: {f['note']}\n"
            f"\nTo contact {f['name'].replace('Prof. ', '')}, "
            f"email {f['email']} or visit {f['profile']}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source": f["profile"],
                "title":  f["name"],
                "type":   "faculty",
                "dept":   f["dept"],
                "email":  f["email"],
            },
        ))

    # Also add a summary document listing all faculty
    summary = "Katz School Faculty List:\n\n"
    for f in KATZ_FACULTY:
        summary += (
            f"• {f['name']} ({f['dept']}) — "
            f"{f['email']} — {', '.join(f['expertise'][:3])}\n"
        )
    docs.append(Document(
        page_content=summary,
        metadata={"source": "https://www.yu.edu/katz/faculty", "title": "Faculty Directory"},
    ))

    print(f"[Faculty] Generated {len(docs)} faculty documents")
    return docs