"""
agents/research_agents.py
===========================
GROQ FREE TIER FIX:
  - Only llama-3.1-8b-instant works reliably on free tier:
      250,000 TPM  (vs 6,000 for qwen, 12,000 for llama-70b)
      14,400 req/day
  - max_iter raised to 4 so the crawler actually gets to call its tool
  - allow_delegation=False on all agents
  - tools=[] on all non-crawler agents (no schema errors)
"""

import os
from crewai import Agent
from llm_config import get_safe_crewai_llm_string, get_crewai_llm_string, get_provider, get_model_name



def make_agents(include_planner: bool = True) -> list:
    llm = get_safe_crewai_llm_string()

    from tools.semantic_scholar import semantic_scholar_search

    # ── 1. Crawler — ONLY agent with a tool ──────────────────
    crawler = Agent(
        role="Academic Paper Crawler",
        goal=(
            "Search Semantic Scholar to find 10-15 relevant academic papers "
            "on the given research topic. Return a numbered list with title, "
            "authors (first 2), year, citation count, 1-sentence abstract, and URL."
        ),
        backstory=(
            "You are an expert academic librarian. You use the "
            "semantic_scholar_search tool to find papers, then format the results."
        ),
        tools=[semantic_scholar_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,           # needs iterations: think → tool call → format output
        max_retry_limit=2,
    )

    # ── 2. Reader — no tools needed ──────────────────────────
    reader = Agent(
        role="Research Paper Analyst",
        goal=(
            "For each paper in the provided list, write a 3-sentence extraction: "
            "main claim, methodology, and key finding."
        ),
        backstory="You are a PhD researcher who rapidly extracts core insights from papers.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    # ── 3. Mapper ─────────────────────────────────────────────
    mapper = Agent(
        role="Research Relationship Mapper",
        goal=(
            "Based on the paper extractions, identify: "
            "(1) agreement clusters, (2) contradictions, "
            "(3) methodological trends. Keep under 300 words."
        ),
        backstory="You are an expert in research synthesis and meta-analysis.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    # ── 4. Gap Finder ─────────────────────────────────────────
    gap_finder = Agent(
        role="Research Gap Identifier",
        goal=(
            "Identify exactly 3 unexplored research opportunities. "
            "For each: gap name, what is missing, why it matters, suggested approach. "
            "Keep each gap under 60 words. Total under 250 words."
        ),
        backstory="You are a research strategist who identifies white spaces in literature.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    # ── 5. Writer ─────────────────────────────────────────────
    writer = Agent(
        role="Academic Literature Review Writer",
        goal=(
            "Write a 400-500 word academic literature review in flowing prose. "
            "Cite papers as (Author et al., Year). No bullet points. "
            "End with a paragraph on the top research gap."
        ),
        backstory="You are an experienced academic writer for peer-reviewed journals.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
    )

    agents = [crawler, reader, mapper, gap_finder, writer]

    # ── 6. Study Planner (optional) ───────────────────────────
    if include_planner:
        planner = Agent(
            role="Study Planner",
            goal=(
                "Create a day-by-day study plan naming specific papers each day. "
                "Keep total under 300 words."
            ),
            backstory="You are an academic coach who creates realistic study plans.",
            tools=[],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=1,
        )
        agents.append(planner)

    return agents