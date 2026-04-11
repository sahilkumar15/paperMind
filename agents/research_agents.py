"""
agents/research_agents.py
===========================
6 research agents for ScholarMind.

Groq free-tier strategy:
  - Default: llama-3.1-8b-instant (250K TPM — safe for 6 agents)
  - Only Crawler has tools (prevents function-call schema errors on Groq)
  - max_iter=5 on Crawler (needs: think → call tool → read → format)
  - max_iter=3 on all others (text-in text-out)
  - allow_delegation=False everywhere
  - memory=False in Crew (Groq has no embeddings endpoint)
"""

from crewai import Agent
from llm_config import get_crewai_llm_string


def make_agents(include_planner: bool = True) -> list:
    llm = get_crewai_llm_string()

    from tools.semantic_scholar import semantic_scholar_search

    crawler = Agent(
        role="Academic Paper Crawler",
        goal=(
            "Search Semantic Scholar to find 10-15 relevant academic papers "
            "on the given research topic. Return a numbered list with title, "
            "authors (first 2), year, citation count, 1-sentence abstract, URL."
        ),
        backstory=(
            "You are an expert academic librarian specialising in finding "
            "high-impact papers. You ALWAYS use the semantic_scholar_search "
            "tool to find real papers."
        ),
        tools=[semantic_scholar_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        max_retry_limit=2,
    )

    reader = Agent(
        role="Research Paper Analyst",
        goal=(
            "For each paper in the provided list, write a 3-sentence extraction: "
            "main claim, methodology used, and key finding."
        ),
        backstory="You are a PhD researcher who rapidly extracts core insights.",
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
            "Based on the paper extractions, write 3 short sections: "
            "Agreements, Contradictions, Methodological Trends. "
            "Under 300 words total."
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
            "Identify exactly 3 unexplored research opportunities. "
            "For each: gap name, what is missing, why it matters, "
            "suggested approach. Under 250 words total."
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
            "Write a 400-500 word academic literature review in flowing prose. "
            "Cite as (Author et al., Year). No bullet points. "
            "End with gap positioning."
        ),
        backstory="You write publication-ready academic literature reviews.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    agents = [crawler, reader, mapper, gap_finder, writer]

    if include_planner:
        planner = Agent(
            role="Study Planner",
            goal=(
                "Create a day-by-day study plan naming specific papers each day. "
                "Under 300 words."
            ),
            backstory="You create realistic, structured academic study plans.",
            tools=[],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=1,
        )
        agents.append(planner)

    return agents
