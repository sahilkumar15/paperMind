"""
agents/research_agents.py

PaperMind — 6 specialized research agents.
Provider-agnostic: works with Groq (free) or OpenAI.
"""

import os
from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tools.semantic_scholar import search_and_format, get_recommendations, format_paper
from llm_config import get_crewai_llm_string


# ── Custom CrewAI tools wrapping Semantic Scholar ─────────────
class SearchInput(BaseModel):
    query: str = Field(description="Academic search query string")
    limit: int = Field(default=15, description="Max number of papers to return")


class SemanticSearchTool(BaseTool):
    name: str = "semantic_scholar_search"
    description: str = (
        "Search Semantic Scholar's database of 200M+ academic papers. "
        "Returns titles, authors, abstracts, citation counts, and paper IDs."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, limit: int = 15) -> str:
        return search_and_format(query, limit)


class RecommendInput(BaseModel):
    paper_id: str = Field(description="Semantic Scholar paper ID")


class RecommendTool(BaseTool):
    name: str = "get_paper_recommendations"
    description: str = (
        "Given a Semantic Scholar paper ID, returns similar recommended papers."
    )
    args_schema: type[BaseModel] = RecommendInput

    def _run(self, paper_id: str) -> str:
        papers = get_recommendations(paper_id, limit=10)
        if not papers:
            return "No recommendations found for this paper ID."
        return "\n\n".join(format_paper(p) for p in papers)


search_tool    = SemanticSearchTool()
recommend_tool = RecommendTool()


# ── Agent 1: Crawler ─────────────────────────────────────────
def create_crawler_agent() -> Agent:
    return Agent(
        role="Academic Paper Crawler",
        goal=(
            "Find the most relevant, high-impact academic papers on the given "
            "research topic using Semantic Scholar. Run at least 3 different search "
            "queries. Find at least 15 papers. Prioritize highly cited recent work."
        ),
        backstory=(
            "You are an expert academic librarian who has spent 20 years helping "
            "researchers at MIT and Stanford find the exact papers they need. "
            "You know how to craft precise search queries and identify seminal works."
        ),
        tools=[search_tool, recommend_tool],
        llm=get_crewai_llm_string(),
        verbose=True,
        max_iter=6,
    )


# ── Agent 2: Reader ──────────────────────────────────────────
def create_reader_agent() -> Agent:
    return Agent(
        role="Deep Paper Analyst",
        goal=(
            "For each paper, extract: core claim, methodology, key findings, "
            "dataset used, limitations, and contribution type."
        ),
        backstory=(
            "You are a PhD researcher who has reviewed thousands of papers for "
            "NeurIPS, ICML, and ACL. You produce clean structured extractions."
        ),
        llm=get_crewai_llm_string(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 3: Mapper ──────────────────────────────────────────
def create_mapper_agent() -> Agent:
    return Agent(
        role="Research Relationship Mapper",
        goal=(
            "Analyze all papers and map: agreement clusters, contradictions, "
            "citation lineage, methodological landscape, and consensus view."
        ),
        backstory=(
            "You are a meta-researcher who specializes in understanding how fields "
            "evolve. You spot debates, consensus views, outliers, and intellectual lineages."
        ),
        llm=get_crewai_llm_string(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 4: Gap Finder ──────────────────────────────────────
def create_gap_finder_agent() -> Agent:
    return Agent(
        role="Research Gap Analyst",
        goal=(
            "Identify genuine research gaps: unexplored combinations, understudied "
            "domains, unresolved contradictions, limitation opportunities, and "
            "temporal gaps. Rank top 3 gaps by importance and feasibility."
        ),
        backstory=(
            "You are a grant reviewer and PhD advisor with a sharp eye for what's "
            "missing in a field. Your gap analyses have led to top-venue publications."
        ),
        llm=get_crewai_llm_string(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 5: Writer ──────────────────────────────────────────
def create_writer_agent() -> Agent:
    return Agent(
        role="Academic Literature Review Writer",
        goal=(
            "Write a publication-quality literature review (600-900 words). "
            "Organize thematically, synthesize (don't just summarize), highlight "
            "debates, and position the research within the field."
        ),
        backstory=(
            "You are a prolific academic writer with 100+ published papers. "
            "Your literature reviews are cited as models of clarity."
        ),
        llm=get_crewai_llm_string(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 6: Study Planner ───────────────────────────────────
def create_planner_agent() -> Agent:
    return Agent(
        role="Research Study Planner",
        goal=(
            "Create a realistic day-by-day study and research plan. Prioritize "
            "the most cited papers first. Include daily milestones and reading strategies."
        ),
        backstory=(
            "You are an academic success coach who helps grad students manage "
            "overwhelming reading lists using spaced repetition and deep work scheduling."
        ),
        llm=get_crewai_llm_string(),
        verbose=True,
        max_iter=3,
    )
