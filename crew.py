"""
crew.py — KatzScholarMind orchestrator
======================================

Key fix:
- Search is deterministic Python now, not LLM tool-calling.
- This avoids Groq function-call failures such as hallucinated brave_search.
- Semantic Scholar is primary; arXiv is the free fallback.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def run_papermind(
    topic: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> dict:
    from crewai import Crew, Process
    from agents.tasks import build_research_tasks
    from llm_config import get_provider, get_model_name, get_crewai_llm_string
    from tools.paper_search import search_papers, format_papers_for_display, format_papers_for_prompt

    provider = get_provider()
    model = get_model_name()
    llm_str = get_crewai_llm_string()

    if provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    print(f"\n{'='*60}")
    print(f"  KatzScholarMind | Provider: {provider.upper()} | Model: {model}")
    print(f"  LiteLLM string: {llm_str}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    # Deterministic retrieval first.
    papers = search_papers(topic, max_results=12)
    papers_display = format_papers_for_display(papers)
    papers_prompt = format_papers_for_prompt(papers)

    if not papers:
        return {
            "papers": papers_display,
            "extractions": "No papers found.",
            "map": "No papers found.",
            "gaps": "No papers found.",
            "lit_review": "No papers found.",
            "study_plan": "No papers found." if include_planner else "",
            "raw": "No papers found from Semantic Scholar or arXiv.",
        }

    agents, tasks = build_research_tasks(
        topic=topic,
        papers_context=papers_prompt,
        research_question=research_question,
        days=days,
        hours_per_day=hours_per_day,
        include_planner=include_planner,
    )

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
        embedder=None,
    )

    result = crew.kickoff()

    keys = ["papers", "extractions", "map", "gaps", "lit_review"]
    if include_planner:
        keys.append("study_plan")

    outputs = {"papers": papers_display}
    for i, task in enumerate(tasks[1:], start=1):
        key = keys[i] if i < len(keys) else f"task_{i}"
        try:
            raw = ""
            if task.output:
                if hasattr(task.output, "raw") and task.output.raw:
                    raw = task.output.raw
                elif hasattr(task.output, "result") and task.output.result:
                    raw = str(task.output.result)
                else:
                    raw = str(task.output)
            outputs[key] = raw
        except Exception as e:
            outputs[key] = f"[Error extracting output: {e}]"

    # Curated paper list is still the authoritative papers output.
    outputs.setdefault("extractions", "")
    outputs.setdefault("map", "")
    outputs.setdefault("gaps", "")
    outputs.setdefault("lit_review", "")
    if include_planner:
        outputs.setdefault("study_plan", "")

    outputs["raw"] = str(result) if result else ""
    return outputs


if __name__ == "__main__":
    result = run_papermind(
        topic="Agentic AI and Multi-Agent Systems",
        research_question="What are the key architectures and unsolved problems?",
        days=7,
        hours_per_day=3,
    )
    print("\n=== PAPERS ===")
    print(result.get("papers", "No output"))
    print("\n=== LIT REVIEW DRAFT ===")
    print(result.get("lit_review", "No output"))
