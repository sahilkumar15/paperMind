"""
agents/research_agents.py
=========================
Text-only research agents for KatzScholarMind.

Important design change:
- No agent gets a search tool.
- Paper retrieval is done deterministically in Python before CrewAI starts.
- This avoids Groq tool-calling failures like hallucinated brave_search calls.
"""

from crewai import Agent
from llm_config import get_crewai_llm_string


def make_agents(include_planner: bool = True) -> list:
    llm = get_crewai_llm_string()

    curator = Agent(
        role="Academic Paper Curator",
        goal=(
            "Given a pre-fetched candidate paper list, choose the most relevant 10-12 papers "
            "and rewrite them as a clean, concise research reading list."
        ),
        backstory=(
            "You are an expert academic librarian. You do not browse the web here. "
            "You only organize and refine the papers already provided in context."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    reader = Agent(
        role="Research Paper Analyst",
        goal=(
            "For each paper in the provided list, write a short extraction with claim, "
            "method, and finding."
        ),
        backstory="You are a PhD researcher who rapidly extracts core insights from paper summaries.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    mapper = Agent(
        role="Research Relationship Mapper",
        goal=(
            "Based on the paper extractions, summarize agreements, contradictions, and "
            "methodological trends."
        ),
        backstory="You are an expert in research synthesis and meta-analysis.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    gap_finder = Agent(
        role="Research Gap Identifier",
        goal=(
            "Identify exactly 3 unexplored research opportunities grounded in the provided papers."
        ),
        backstory="You identify white spaces in academic literature.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    writer = Agent(
        role="Academic Literature Review Writer",
        goal=(
            "Write a 400-500 word academic literature review in flowing prose using only the "
            "provided paper set and downstream synthesis."
        ),
        backstory="You write polished, evidence-grounded literature review drafts.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    agents = [curator, reader, mapper, gap_finder, writer]

    if include_planner:
        planner = Agent(
            role="Research Planner",
            goal=(
                "Create a day-by-day research reading plan using the provided paper list and gaps."
            ),
            backstory="You create realistic, structured academic reading plans.",
            tools=[],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=1,
        )
        agents.append(planner)

    return agents
