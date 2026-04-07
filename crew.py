"""
crew.py — PaperMind orchestrator

Runs the full multi-agent research pipeline.
Call run_papermind() to start.
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
    """
    Run the full PaperMind research synthesis pipeline.

    Returns dict with keys:
        papers      — crawled papers list
        extractions — per-paper structured extraction
        map         — relationship map
        gaps        — research gap analysis
        lit_review  — full literature review draft
        study_plan  — personalized reading plan (if include_planner)
    """

    agents, tasks = build_research_tasks(
        topic=topic,
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
    )

    # crew = Crew(
    #     agents=agents,
    #     tasks=tasks,
    #     process=Process.sequential,
    #     verbose=True,
    #     memory=True,
    #     embedder={
    #         "provider": "openai",
    #         "config": {"model": "text-embedding-3-small"},
    #     },
    # )

    print(f"\n{'='*60}")
    print(f"  PaperMind starting for: {topic}")
    print(f"{'='*60}\n")

    result = crew.kickoff()

    keys = ["papers", "extractions", "map", "gaps", "lit_review"]
    if include_planner:
        keys.append("study_plan")

    outputs = {}
    for i, task in enumerate(tasks):
        key = keys[i] if i < len(keys) else f"task_{i}"
        outputs[key] = task.output.raw if task.output else ""

    outputs["raw"] = str(result)
    return outputs


if __name__ == "__main__":
    result = run_papermind(
        topic="Agentic AI and Multi-Agent Systems",
        research_question="What are the key architectures and unsolved problems in deploying agentic AI?",
        days=7,
        hours_per_day=3,
    )
    print("\n=== LIT REVIEW DRAFT ===")
    print(result["lit_review"])
    print("\n=== RESEARCH GAPS ===")
    print(result["gaps"])
