"""
agents/tasks.py
================
Token-efficient tasks for Groq free tier.
Each task output is capped to stay under TPM limits.
"""

from crewai import Task
from agents.research_agents import make_agents


def build_research_tasks(
    topic: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> tuple:
    agents     = make_agents(include_planner=include_planner)
    crawler    = agents[0]
    reader     = agents[1]
    mapper     = agents[2]
    gap_finder = agents[3]
    writer     = agents[4]
    planner    = agents[5] if include_planner and len(agents) > 5 else None

    rq = f" Research question: {research_question}." if research_question else ""

    task_crawl = Task(
        description=(
            f"Search Semantic Scholar for: '{topic}'.{rq}\n"
            f"Use the semantic_scholar_search tool.\n"
            f"Return a numbered list (10-15 papers):\n"
            f"[N]. Title (Year) — Authors — Citations: X\n"
            f"Summary: one sentence. URL: ...\n"
            f"Total output: under 500 words."
        ),
        expected_output=(
            "Numbered list of 10-15 papers with title, year, authors, "
            "citations, 1-sentence summary, URL. Under 500 words."
        ),
        agent=crawler,
    )

    task_read = Task(
        description=(
            f"Papers on '{topic}' are listed above.\n"
            f"For each paper write ONE paragraph (2-3 sentences):\n"
            f"Claim: ... Method: ... Finding: ...\n"
            f"Total output: under 500 words."
        ),
        expected_output=(
            "One short paragraph per paper with claim, method, finding. Under 500 words."
        ),
        agent=reader,
        context=[task_crawl],
    )

    task_map = Task(
        description=(
            f"Based on extractions for '{topic}', write:\n"
            f"## Agreements\n(2-3 sentences)\n"
            f"## Contradictions\n(2-3 sentences)\n"
            f"## Methodological Trends\n(2-3 sentences)\n"
            f"Total: under 250 words."
        ),
        expected_output=(
            "Three sections (Agreements, Contradictions, Trends), under 250 words."
        ),
        agent=mapper,
        context=[task_read],
    )

    task_gaps = Task(
        description=(
            f"Based on papers about '{topic}', list 3 research gaps.{rq}\n"
            f"Format each:\n"
            f"**Gap [N]: [Name]**\n"
            f"Missing: ... Matters: ... Approach: ...\n"
            f"Total: under 250 words."
        ),
        expected_output=(
            "3 research gaps with name, what is missing, why it matters, "
            "and suggested approach. Under 250 words."
        ),
        agent=gap_finder,
        context=[task_map],
    )

    task_litrev = Task(
        description=(
            f"Write a literature review on '{topic}'.{rq}\n"
            f"400-500 words, academic prose, no bullets.\n"
            f"Cite as (Author et al., Year).\n"
            f"Structure: intro → themes → synthesis → gap positioning."
        ),
        expected_output=(
            "400-500 word literature review in academic prose, ending with gaps."
        ),
        agent=writer,
        context=[task_crawl, task_map, task_gaps],
    )

    tasks = [task_crawl, task_read, task_map, task_gaps, task_litrev]

    if include_planner and planner:
        task_plan = Task(
            description=(
                f"Create a {days}-day study plan for '{topic}'.\n"
                f"Time: {hours_per_day}h/day.{rq}\n"
                f"Format: **Day X** — Focus — Papers to read.\n"
                f"Name actual papers. Under 300 words."
            ),
            expected_output=(
                f"Day-by-day plan for {days} days naming specific papers. Under 300 words."
            ),
            agent=planner,
            context=[task_crawl, task_gaps],
        )
        tasks.append(task_plan)

    return agents, tasks
