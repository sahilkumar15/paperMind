"""
chatbot.py — PaperBot conversational assistant
================================================
Works with both Groq (free) and OpenAI.
Provider is controlled by LLM_PROVIDER in .env
"""

import os
from dotenv import load_dotenv
from llm_config import get_openai_client, get_model_name

load_dotenv()

PAPERBOT_SYSTEM = """You are PaperBot, an expert AI research assistant built into PaperMind 
at Yeshiva University's Katz School of Science and Health.

Your dual role:
1. RESEARCH ADVISOR — Help students understand papers, navigate literature,
   identify research opportunities, and prepare strong academic arguments.
2. ACADEMIC ADVISOR — Help students connect with the right Katz professors for
   supervision, collaboration, or feedback on their research ideas.

Your personality:
- Direct and intellectually sharp — clear, well-reasoned answers
- Encouraging but honest — tell students when an idea is weak or already done
- Research-focused — connect answers to the academic literature
- Practical — help students take the next concrete step

When answering:
1. Be direct — lead with the answer
2. Reference specific papers from the context when relevant
3. When recommending professors, explain WHY they are a good match
4. Always offer the next concrete step

Context from the student's research will be injected below when available."""


def chat(messages: list, context: str = "", topic: str = "") -> str:
    """
    Send conversation to PaperBot and return the reply.
    Uses whichever provider is set in LLM_PROVIDER (.env).
    """
    client = get_openai_client()
    model  = get_model_name()

    system = PAPERBOT_SYSTEM

    if context:
        system += (
            "\n\n--- STUDENT'S RESEARCH CONTEXT ---\n"
            f"{context[:4000]}\n"
            "--- END RESEARCH CONTEXT ---\n"
        )

    if topic:
        try:
            from katzbot.rag_engine import match_faculty
            top_profs = match_faculty(topic)
            faculty_ctx = "\n--- RELEVANT KATZ FACULTY ---\n"
            for p in top_profs:
                faculty_ctx += (
                    f"* {p['name']} ({p['title']})\n"
                    f"  Dept: {p['dept']} | Email: {p['email']}\n"
                    f"  Expertise: {', '.join(p['expertise'][:4])}\n\n"
                )
            system += faculty_ctx + "---\n"
        except Exception:
            pass

    full_messages = [{"role": "system", "content": system}] + messages

    resp = client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=1200,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def draft_contact_email(
    student_name: str,
    prof_name: str,
    prof_email: str,
    topic: str,
    project_summary: str,
) -> str:
    """Draft a professional email from a student to a Katz professor."""
    client = get_openai_client()
    model  = get_model_name()

    prompt = (
        f"Draft a professional email from a Katz School graduate student "
        f"to a professor requesting feedback or supervision.\n\n"
        f"Student name: {student_name}\n"
        f"Professor: {prof_name} ({prof_email})\n"
        f"Research topic: {topic}\n"
        f"Project summary: {project_summary}\n\n"
        f"Requirements: Include subject line. Professional but warm. Under 200 words. "
        f"Specific about the project. Clear ask (15-min meeting or email feedback). "
        f"Mention the Katz School Ideathon 2026 context."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=450,
        temperature=0.6,
    )
    return resp.choices[0].message.content


QUICK_PROMPTS = [
    "Which Katz professor should I contact about my research?",
    "What are the most important papers I must read first?",
    "Explain the biggest debate in this field",
    "Is my research idea genuinely novel?",
    "What methodology should I use for my research?",
    "Help me write a 2-sentence positioning statement",
    "Which research gap is easiest to address in 6 months?",
    "Draft an email to a professor about my project",
    "What will reviewers criticize about work in this area?",
    "Compare the top 2 competing approaches",
]
