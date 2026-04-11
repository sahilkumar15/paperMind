"""
agents/tasks.py
================
TOKEN-EFFICIENT tasks designed for Groq free tier (250K TPM with llama-3.1-8b-instant).

Key rules:
  - Each task output is SHORT (300-600 words max)
  - Context passed between tasks is TRIMMED (not full previous output)
  - Task descriptions are concise to minimize input tokens
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

    # ── Task 1: Crawl ─────────────────────────────────────────
    task_crawl = Task(
        description=(
            f"Search Semantic Scholar for papers on: '{topic}'.{rq}\n"
            f"Call the semantic_scholar_search tool with query: '{topic}'.\n"
            f"Format results as a numbered list (10-15 papers):\n"
            f"[N]. Title (Year) — Authors — Citations: X\n"
            f"Summary: one sentence.\nURL: ...\n"
            f"Total output: under 500 words."
        ),
        expected_output=(
            "Numbered list of 10-15 papers. Each entry: title, year, "
            "authors, citation count, 1-sentence summary, URL. Under 500 words."
        ),
        agent=crawler,
    )

    # ── Task 2: Read ──────────────────────────────────────────
    task_read = Task(
        description=(
            f"Papers on '{topic}' are listed above.\n"
            f"For each paper write ONE paragraph (2-3 sentences):\n"
            f"Claim: ... Method: ... Finding: ...\n"
            f"Total output: under 500 words."
        ),
        expected_output=(
            "One short paragraph per paper with claim, method, finding. "
            "Under 500 words total."
        ),
        agent=reader,
        context=[task_crawl],
    )

    # ── Task 3: Map ───────────────────────────────────────────
    task_map = Task(
        description=(
            f"Based on the paper extractions for '{topic}', write 3 short sections:\n"
            f"## Agreements\n(2-3 sentences)\n"
            f"## Contradictions\n(2-3 sentences)\n"
            f"## Methodological Trends\n(2-3 sentences)\n"
            f"Total: under 250 words."
        ),
        expected_output=(
            "Three sections (Agreements, Contradictions, Methodological Trends), "
            "each 2-3 sentences. Under 250 words."
        ),
        agent=mapper,
        context=[task_read],
    )

    # ── Task 4: Gaps ──────────────────────────────────────────
    task_gaps = Task(
        description=(
            f"Based on papers about '{topic}', list 3 research gaps.{rq}\n"
            f"Format each gap as:\n"
            f"**Gap [N]: [Name]**\n"
            f"Missing: ... Matters: ... Approach: ...\n"
            f"Total: under 250 words."
        ),
        expected_output=(
            "3 research gaps, each with name, what is missing, why it matters, "
            "and suggested approach. Under 250 words."
        ),
        agent=gap_finder,
        context=[task_map],
    )

    # ── Task 5: Literature Review ─────────────────────────────
    task_litrev = Task(
        description=(
            f"Write a literature review on '{topic}'.{rq}\n"
            f"Rules: 400-500 words, academic prose, no bullets.\n"
            f"Cite as (Author et al., Year).\n"
            f"Structure: intro → themes → synthesis → gap positioning."
        ),
        expected_output=(
            "400-500 word literature review in academic prose, citing papers, "
            "ending with gap positioning."
        ),
        agent=writer,
        context=[task_crawl, task_map, task_gaps],
    )

    tasks = [task_crawl, task_read, task_map, task_gaps, task_litrev]

    # ── Task 6: Study Plan ────────────────────────────────────
    if include_planner and planner:
        task_plan = Task(
            description=(
                f"Create a {days}-day study plan for '{topic}'.\n"
                f"Time budget: {hours_per_day}h/day.{rq}\n"
                f"Format: **Day X** — Focus — Papers to read\n"
                f"Name actual papers from the list. Under 300 words."
            ),
            expected_output=(
                f"Day-by-day plan for {days} days naming specific papers. Under 300 words."
            ),
            agent=planner,
            context=[task_crawl, task_gaps],
        )
        tasks.append(task_plan)

    return agents, tasks