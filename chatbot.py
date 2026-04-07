"""
chatbot.py — PaperBot

Conversational AI assistant that answers research questions
using the generated literature review, gap analysis, and paper list as context.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PAPERBOT_SYSTEM = """You are PaperBot, an expert AI research assistant built into PaperMind 
at Yeshiva University's Katz School of Science and Health.

Your role: Help graduate students deeply understand research papers, navigate literature, 
identify research opportunities, and prepare strong academic arguments.

Your personality:
- Direct and intellectually sharp — you give clear, well-reasoned answers
- Encouraging but honest — you tell students when an idea is weak or already done
- Research-focused — you always connect answers to the academic literature
- Practical — you help students take the next concrete step

What you can do:
- Explain any paper, concept, or methodology in the literature
- Compare competing approaches and explain who is right and why
- Help a student position their research idea within the field
- Suggest which papers to read next based on their goals
- Help draft arguments for a literature review
- Identify if a proposed research idea is truly novel
- Prepare students for Q&A about their research

When answering:
1. Be direct — lead with the answer, not with "great question"
2. Reference specific papers from the context when relevant
3. Point out nuances and edge cases
4. End with a question or suggestion to push the student's thinking forward

Context from the student's research will be injected below when available."""


def chat(messages: list, context: str = "") -> str:
    """
    Send conversation to PaperBot and return the reply.

    messages: list of {"role": "user"/"assistant", "content": "..."}
    context:  the generated research outputs to use as knowledge base
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model  = os.getenv("MODEL_NAME", "gpt-4o-mini")

    system = PAPERBOT_SYSTEM
    if context:
        system += (
            "\n\n--- STUDENT'S RESEARCH CONTEXT ---\n"
            f"{context[:5000]}\n"
            "--- END CONTEXT ---\n"
            "Use this context to give specific, grounded answers."
        )

    full_messages = [{"role": "system", "content": system}] + messages

    resp = client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=1200,
        temperature=0.7,
    )
    return resp.choices[0].message.content


QUICK_PROMPTS = [
    "What are the most important papers I must read first?",
    "Explain the biggest debate in this field",
    "Is my research idea genuinely novel?",
    "What methodology should I use for my research?",
    "Help me write a 2-sentence positioning statement",
    "Which research gap is easiest to address in 6 months?",
    "What will reviewers criticize about work in this area?",
    "Compare the top 2 competing approaches",
]
