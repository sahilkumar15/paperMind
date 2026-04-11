"""
chatbot.py — PaperBot direct chat (no agents, instant response)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def chat(messages: list, system_prompt: str = "") -> str:
    """
    Direct chat using Groq/OpenAI API (OpenAI-compatible).
    messages: list of {"role": "user"/"assistant", "content": "..."}
    """
    try:
        from llm_config import get_openai_client, get_model_name

        client = get_openai_client()
        model  = get_model_name()

        system = system_prompt or (
            "You are PaperBot, an AI research assistant for ScholarMind. "
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
