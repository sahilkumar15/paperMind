"""
crew.py — PaperMind orchestrator
Works with Groq (free) or OpenAI — set LLM_PROVIDER in .env
"""

import os
from dotenv import load_dotenv
from crewai import Crew, Process
from agents.tasks import build_research_tasks

load_dotenv()


def run_papermind(
    topic: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> dict:
    agents, tasks = build_research_tasks(
        topic=topic,
        research_question=research_question,
        days=days,
        hours_per_day=hours_per_day,
        include_planner=include_planner,
    )

    from llm_config import get_provider, get_model_name
    provider = get_provider()
    model    = get_model_name()
    print(f"\n{'='*60}")
    print(f"  PaperMind | Provider: {provider.upper()} | Model: {model}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    # memory=False avoids the embeddings 403 error on Groq/basic OpenAI accounts
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    result = crew.kickoff()

    keys = ["papers", "extractions", "map", "gaps", "lit_review"]
    if include_planner:
        keys.append("study_plan")

    outputs = {}
    for i, task in enumerate(tasks):
        key = keys[i] if i < len(keys) else f"task_{i}"
        try:
            outputs[key] = task.output.raw if (task.output and task.output.raw) else ""
        except Exception:
            outputs[key] = ""

    outputs["raw"] = str(result) if result else ""
    return outputs


if __name__ == "__main__":
    result = run_papermind(
        topic="Agentic AI and Multi-Agent Systems",
        research_question="What are the key architectures and unsolved problems?",
        days=7,
        hours_per_day=3,
    )
    print("\n=== LIT REVIEW DRAFT ===")
    print(result["lit_review"])
