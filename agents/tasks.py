"""
agents/tasks.py
===============
Tasks for KatzScholarMind.
Search is already completed in Python before these tasks run.
"""

from crewai import Task
from agents.research_agents import make_agents


def build_research_tasks(
    topic: str,
    papers_context: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> tuple:
    agents = make_agents(include_planner=include_planner)
    curator = agents[0]
    reader = agents[1]
    mapper = agents[2]
    gap_finder = agents[3]
    writer = agents[4]
    planner = agents[5] if include_planner and len(agents) > 5 else None

    rq = f" Research question: {research_question}." if research_question else ""

    task_curate = Task(
        description=(
            f"Topic: '{topic}'.{rq}\n"
            f"The candidate papers below were already retrieved from Semantic Scholar and/or arXiv.\n"
            f"Do NOT invent or browse for new papers.\n\n"
            f"Candidate papers:\n{papers_context}\n\n"
            f"Select the 10-12 most relevant papers and rewrite them as a numbered list in this format:\n"
            f"[N]. Title (Year) — Authors — Citations: X — Source: Semantic Scholar/arXiv\n"
            f"Summary: one sentence. URL: ...\n"
            f"Stay under 650 words."
        ),
        expected_output=(
            "A clean numbered list of the best 10-12 papers with title, year, authors, citations, source, summary, and URL."
        ),
        agent=curator,
    )

    task_read = Task(
        description=(
            f"Using only the curated paper list for '{topic}', write one short paragraph per paper.\n"
            f"Format each as: Claim: ... Method: ... Finding: ...\n"
            f"Be faithful to the available summaries and do not invent extra details.\n"
            f"Total output under 700 words."
        ),
        expected_output=(
            "Short claim/method/finding extraction for each paper, grounded in the curated list."
        ),
        agent=reader,
        context=[task_curate],
    )

    task_map = Task(
        description=(
            f"Based on the extractions for '{topic}', write three short sections:\n"
            f"## Agreements\n"
            f"## Contradictions\n"
            f"## Methodological Trends\n"
            f"Keep the whole answer under 300 words."
        ),
        expected_output=(
            "Three short sections: Agreements, Contradictions, Methodological Trends."
        ),
        agent=mapper,
        context=[task_read],
    )

    task_gaps = Task(
        description=(
            f"Based on the papers about '{topic}', identify exactly 3 research gaps.{rq}\n"
            f"Format each as:\n"
            f"**Gap [N]: [Name]**\n"
            f"Missing: ... Matters: ... Approach: ...\n"
            f"Keep the whole answer under 280 words."
        ),
        expected_output=(
            "Exactly 3 research gaps with what is missing, why it matters, and a suggested approach."
        ),
        agent=gap_finder,
        context=[task_curate, task_map],
    )

    task_litrev = Task(
        description=(
            f"Write a literature review on '{topic}'.{rq}\n"
            f"Length: 400-500 words. Academic prose only, no bullet points.\n"
            f"Use citations like (Author et al., Year) when possible from the curated list.\n"
            f"Structure: introduction -> themes -> synthesis -> gap positioning."
        ),
        expected_output=(
            "A 400-500 word literature review grounded in the curated papers and ending with gap positioning."
        ),
        agent=writer,
        context=[task_curate, task_map, task_gaps],
    )

    tasks = [task_curate, task_read, task_map, task_gaps, task_litrev]

    if include_planner and planner:
        task_plan = Task(
            description=(
                f"Create a {days}-day research reading plan for '{topic}'.\n"
                f"Time budget: {hours_per_day} hours/day.{rq}\n"
                f"Use only papers from the curated list.\n"
                f"Format: **Day X** — Focus — Papers to read.\n"
                f"Keep under 350 words."
            ),
            expected_output=(
                f"A day-by-day {days}-day reading plan naming specific papers from the curated list."
            ),
            agent=planner,
            context=[task_curate, task_gaps],
        )
        tasks.append(task_plan)

    return agents, tasks
