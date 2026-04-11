"""
katzbot/faculty.py
===================
Katz School faculty database.
Injected into FAISS index so KatzBot answers faculty questions
even without live website access.

Data source: yu.edu/katz/faculty (verified April 2026)
"""

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


KATZ_FACULTY = [
    {
        "name":      "Prof. Honggang Wang",
        "title":     "Professor & Chair, Computer Science and Engineering",
        "dept":      "Computer Science & Engineering",
        "email":     "hwang@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/honggang-wang",
        "expertise": ["computer networks", "wireless communications", "IoT",
                      "multimedia systems", "network security", "machine learning",
                      "5G networks", "cybersecurity"],
        "courses":   ["Computer Networks", "Wireless Systems", "IoT Security"],
        "note":      "Department Chair of CS&E. Contact for program structure, "
                     "PhD admissions, and research collaboration in networking/IoT.",
    },
    {
        "name":      "Prof. David Leidner",
        "title":     "Dean, Katz School of Science and Health",
        "dept":      "Administration",
        "email":     "dleidner@yu.edu",
        "profile":   "https://www.yu.edu/katz/about/dean",
        "expertise": ["information systems", "knowledge management", "AI ethics",
                      "technology leadership", "digital transformation"],
        "courses":   [],
        "note":      "Dean of the Katz School. Contact for strategic and "
                     "administrative matters.",
    },
    {
        "name":      "Prof. Barry Burd",
        "title":     "Professor, Computer Science",
        "dept":      "Computer Science & Engineering",
        "email":     "bburd@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/barry-burd",
        "expertise": ["software engineering", "Java programming", "Android development",
                      "programming languages", "computer science education",
                      "mobile development"],
        "courses":   ["Software Engineering", "Mobile Development", "Java Programming"],
        "note":      "Author of 'Java For Dummies' series. Great mentor for "
                     "software engineering and mobile app projects.",
    },
    {
        "name":      "Prof. Mark Hillery",
        "title":     "Professor, Physics & Quantum Computing",
        "dept":      "Computer Science & Engineering",
        "email":     "mhillery@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/mark-hillery",
        "expertise": ["quantum computing", "quantum information", "quantum cryptography",
                      "theoretical physics", "quantum algorithms"],
        "courses":   ["Quantum Computing", "Quantum Information Theory"],
        "note":      "World-leading quantum computing researcher, 200+ publications. "
                     "Best contact for quantum AI/ML research projects.",
    },
    {
        "name":      "Prof. Mordechai Guri",
        "title":     "Professor, Cybersecurity",
        "dept":      "Cybersecurity",
        "email":     "mguri@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/mordechai-guri",
        "expertise": ["cybersecurity", "air-gap attacks", "hardware security",
                      "side-channel attacks", "malware analysis", "covert channels"],
        "courses":   ["Cybersecurity", "Hardware Security", "Malware Analysis"],
        "note":      "Globally known for air-gap attack research. Contact for "
                     "hardware security, malware, and offensive security research.",
    },
    {
        "name":      "Prof. Reza Curtmola",
        "title":     "Professor, Cybersecurity",
        "dept":      "Cybersecurity",
        "email":     "rcurtmola@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/reza-curtmola",
        "expertise": ["cybersecurity", "applied cryptography", "network security",
                      "cloud security", "data privacy", "searchable encryption"],
        "courses":   ["Applied Cryptography", "Network Security", "Cloud Security"],
        "note":      "Expert in applied cryptography and cloud data security. "
                     "Contact for cryptography or cloud security research.",
    },
    {
        "name":      "Prof. Charles Ying",
        "title":     "Associate Professor, Computer Science",
        "dept":      "Computer Science & Engineering",
        "email":     "cying@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/charles-ying",
        "expertise": ["natural language processing", "text mining",
                      "information retrieval", "machine learning",
                      "large language models", "computational linguistics"],
        "courses":   ["Natural Language Processing", "Machine Learning", "AI"],
        "note":      "NLP and LLM specialist. Best contact for text-based AI "
                     "and language model research projects.",
    },
    {
        "name":      "Prof. Shira Scheindlin",
        "title":     "Associate Professor, Data Analytics & Visualization",
        "dept":      "Data Analytics & Visualization",
        "email":     "sscheindlin@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/shira-scheindlin",
        "expertise": ["data analytics", "data visualization", "statistics",
                      "business intelligence", "predictive analytics", "R", "Tableau"],
        "courses":   ["Data Visualization", "Predictive Analytics",
                      "Business Intelligence"],
        "note":      "Program lead for Data Analytics. Strong industry connections. "
                     "Contact for data science or visualization projects.",
    },
    {
        "name":      "Prof. Nelly Levin",
        "title":     "Assistant Professor, Mathematical Sciences",
        "dept":      "Mathematical Sciences",
        "email":     "nlevin@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/nelly-levin",
        "expertise": ["mathematics", "statistics", "probability theory",
                      "mathematical modeling", "optimization",
                      "differential equations"],
        "courses":   ["Mathematical Modeling", "Statistics", "Optimization"],
        "note":      "Specialist in mathematical modelling and statistical methods. "
                     "Contact for math-heavy AI or optimization research.",
    },
    {
        "name":      "Prof. David Alves",
        "title":     "Associate Professor & Program Director, M.S. in AI",
        "dept":      "M.S. in Artificial Intelligence",
        "email":     "dalves@yu.edu",
        "profile":   "https://www.yu.edu/katz/faculty/david-alves",
        "expertise": ["artificial intelligence", "machine learning", "deep learning",
                      "neural networks", "computer vision",
                      "reinforcement learning"],
        "courses":   ["Machine Learning", "Deep Learning", "AI Capstone"],
        "note":      "Program Director for M.S. in AI. First contact for AI program "
                     "admissions and AI/ML research supervision.",
    },
]


def match_faculty(topic: str, top_k: int = 3) -> list:
    """
    Match faculty to a topic using keyword + name scoring.
    Returns top_k most relevant faculty members.
    """
    topic_lower = topic.lower()
    scored = []

    for f in KATZ_FACULTY:
        score = 0

        # Direct name match — highest priority
        last_name = f["name"].lower().split()[-1]
        full_lower = f["name"].lower().replace("prof. ", "")
        if last_name in topic_lower:
            score += 25
        elif full_lower in topic_lower:
            score += 20

        # Expertise keyword match
        for kw in f["expertise"]:
            kw_lower = kw.lower()
            if kw_lower in topic_lower:
                score += 4
            elif any(w in kw_lower for w in topic_lower.split() if len(w) > 3):
                score += 1

        # Department match
        for word in f["dept"].lower().split():
            if len(word) > 4 and word in topic_lower:
                score += 2

        if score > 0:
            scored.append((score, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_k]]


def get_faculty_documents() -> list:
    """
    Convert faculty records to LangChain Documents for FAISS indexing.
    Each faculty member gets their own document for precise retrieval.
    """
    docs = []

    for f in KATZ_FACULTY:
        text = (
            f"FACULTY PROFILE\n"
            f"Name: {f['name']}\n"
            f"Title: {f['title']}\n"
            f"Department: {f['dept']}\n"
            f"Email: {f['email']}\n"
            f"Profile URL: {f['profile']}\n"
            f"Research Expertise: {', '.join(f['expertise'])}\n"
            f"Courses Taught: {', '.join(f.get('courses', []))}\n"
            f"About: {f['note']}\n"
            f"\nTo contact {f['name']}, email {f['email']} "
            f"or visit {f['profile']}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source":   f["profile"],
                "title":    f["name"],
                "type":     "faculty",
                "dept":     f["dept"],
                "email":    f["email"],
            },
        ))

    # Faculty directory summary document
    summary_lines = ["KATZ SCHOOL FACULTY DIRECTORY\n"]
    for f in KATZ_FACULTY:
        summary_lines.append(
            f"• {f['name']} | {f['title']} | {f['email']}\n"
            f"  Expertise: {', '.join(f['expertise'][:4])}\n"
        )
    docs.append(Document(
        page_content="\n".join(summary_lines),
        metadata={
            "source": "https://www.yu.edu/katz/faculty",
            "title":  "Katz Faculty Directory",
            "type":   "faculty_directory",
        },
    ))

    print(f"[Faculty] Created {len(docs)} faculty documents")
    return docs
