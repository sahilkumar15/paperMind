"""
chatbot.py — research-aware PaperBot with backwards-compatible chat()

Fixes:
- Provides answer_paperbot(), which app.py expects.
- Keeps chat() for older code paths.
- Grounds answers in the current retrieved paper set.
- Uses deterministic latest-paper retrieval via Semantic Scholar + arXiv.
"""

from __future__ import annotations

from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


def _safe_topic(current_topic: str, user_prompt: str) -> str:
    current_topic = (current_topic or "").strip()
    prompt = (user_prompt or "").strip()
    if current_topic:
        return current_topic
    return prompt[:120]


def _wants_latest(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in [
            "latest paper",
            "latest one",
            "most recent",
            "recent paper",
            "newest paper",
            "current paper",
            "latest work",
        ]
    )


def _wants_novelty(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in [
            "novelty",
            "novel",
            "research gap",
            "what can i work on",
            "future work",
            "new idea",
            "position my idea",
        ]
    )


def _format_latest_answer(topic: str, latest: List[Dict], gaps_text: str = "") -> str:
    if not latest:
        return f"I could not find recent papers for **{topic}** right now from Semantic Scholar/arXiv."

    top = latest[0]
    authors = ", ".join(top.get("authors", [])[:3]) or "Unknown authors"
    bullets = []
    for p in latest[:5]:
        auth = ", ".join(p.get("authors", [])[:2]) or "Unknown authors"
        if len(p.get("authors", [])) > 2:
            auth += " et al."
        bullets.append(
            f"- **{p.get('title','Untitled')}** ({p.get('year','N/A')}) — {auth} — {p.get('source','Unknown')} — {p.get('url','')}"
        )

    novelty = ""
    if gaps_text:
        novelty = (
            "\n\n**Good novelty directions from your current synthesis**\n"
            + gaps_text[:900]
        )

    return (
        f"**Latest paper I found for {topic}:**\n"
        f"**{top.get('title','Untitled')}** ({top.get('year','N/A')})\n"
        f"Authors: {authors}\n"
        f"Source: {top.get('source','Unknown')}\n"
        f"URL: {top.get('url','')}\n"
        "Why start here: it is the most recent result in the retrieved set and gives you an up-to-date entry point before reading older highly cited work."
        f"\n\n**Other recent papers to read**\n" + "\n".join(bullets[1:5]) + novelty
    )


def answer_paperbot(
    prompt: str,
    chat_history: List[Dict],
    current_topic: str = "",
    results: Dict | None = None,
) -> str:
    topic = _safe_topic(current_topic, prompt)
    results = results or {}

    if _wants_latest(prompt):
        from tools.paper_search import search_latest_papers

        latest = search_latest_papers(topic, max_results=6)
        return _format_latest_answer(topic, latest, gaps_text=results.get("gaps", ""))

    evidence_parts = []
    paper_items = results.get("paper_items") or []
    if paper_items:
        snippets = []
        for i, p in enumerate(paper_items[:8], 1):
            snippets.append(
                f"[P{i}] {p.get('title','Untitled')} ({p.get('year','N/A')}) — {p.get('summary','')}"
            )
        evidence_parts.append("Retrieved papers:\n" + "\n".join(snippets))
    elif results.get("papers"):
        evidence_parts.append("Retrieved papers:\n" + str(results.get("papers", ""))[:2200])

    if results.get("map"):
        evidence_parts.append("Relationship map:\n" + results["map"][:1600])
    if results.get("gaps"):
        evidence_parts.append("Research gaps:\n" + results["gaps"][:1400])
    if results.get("lit_review"):
        evidence_parts.append("Literature review draft:\n" + results["lit_review"][:1800])

    system = (
        "You are PaperBot, a grounded AI research assistant inside KatzScholarMind. "
        "Answer using the provided evidence first. Do not claim lack of internet access; instead use the retrieved evidence. "
        "If the user asks for novelty, connect it to concrete gaps and explain why it is promising. "
        "If the evidence is insufficient, say exactly what is missing. "
        f"Current topic: {topic}.\n\n" + "\n\n".join(evidence_parts)
    )

    try:
        from llm_config import get_openai_client, get_model_name

        client = get_openai_client()
        model = get_model_name()
        messages = (
            [{"role": "system", "content": system}]
            + chat_history
            + [{"role": "user", "content": prompt}]
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
        )
        return resp.choices[0].message.content or "No response generated."
    except Exception as e:
        if _wants_novelty(prompt) and results.get("gaps"):
            return (
                f"I could not use the chat model right now, but based on the current grounded gaps for **{topic}**, "
                f"the best novelty directions are:\n\n{results['gaps'][:1200]}"
            )
        if results.get("papers"):
            return (
                f"I could not use the chat model right now. Here are the grounded papers already retrieved for **{topic}**:\n\n"
                f"{results['papers'][:1800]}"
            )
        return f"PaperBot error: {e}"


def chat(messages: list, system_prompt: str = "") -> str:
    """Backwards-compatible direct chat helper for older code paths."""
    try:
        from llm_config import get_openai_client, get_model_name

        client = get_openai_client()
        model = get_model_name()

        system = system_prompt or (
            "You are PaperBot, an AI research assistant for KatzScholarMind. "
            "Help with research questions, academic writing, literature reviews, "
            "citation formatting, and research methodology. "
            "Be concise, accurate, and academic in tone."
        )

        full_messages = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return resp.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"PaperBot error: {e}"
