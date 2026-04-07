"""
agents/research_agents.py

PaperMind — 7 specialized agents:

RESEARCH PIPELINE (5 agents):
  1. Crawler        — finds relevant papers via Semantic Scholar
  2. Reader         — extracts structured info from each paper
  3. Mapper         — finds agreements, contradictions, relationships
  4. Gap Finder     — identifies what has NOT been studied yet
  5. Writer         — drafts the literature review section

SUPPORT AGENTS (2 agents):
  6. Study Planner  — builds a personalized research/study schedule
  7. PaperBot       — conversational assistant using paper context
"""

import os
from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tools.semantic_scholar import search_and_format, get_recommendations, format_paper, search_papers


# ── Custom CrewAI tool wrapping Semantic Scholar ─────────────
class SearchInput(BaseModel):
    query: str = Field(description="Academic search query string")
    limit: int = Field(default=15, description="Max number of papers to return")


class SemanticSearchTool(BaseTool):
    name: str = "semantic_scholar_search"
    description: str = (
        "Search Semantic Scholar's database of 200M+ academic papers. "
        "Returns titles, authors, abstracts, citation counts, and paper IDs. "
        "Use this to find papers on any research topic."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, limit: int = 15) -> str:
        return search_and_format(query, limit)


class RecommendInput(BaseModel):
    paper_id: str = Field(description="Semantic Scholar paper ID")


class RecommendTool(BaseTool):
    name: str = "get_paper_recommendations"
    description: str = (
        "Given a Semantic Scholar paper ID, returns similar recommended papers. "
        "Use this to find papers related to a specific key paper you already found."
    )
    args_schema: type[BaseModel] = RecommendInput

    def _run(self, paper_id: str) -> str:
        papers = get_recommendations(paper_id, limit=10)
        if not papers:
            return "No recommendations found for this paper ID."
        return "\n\n".join(format_paper(p) for p in papers)


# Instantiate tools
search_tool = SemanticSearchTool()
recommend_tool = RecommendTool()


def _llm():
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return f"openai/{model}"


# ── Agent 1: Crawler ─────────────────────────────────────────
def create_crawler_agent() -> Agent:
    return Agent(
        role="Academic Paper Crawler",
        goal=(
            "Find the most relevant, high-impact academic papers on the given "
            "research topic using Semantic Scholar. Prioritize papers with high "
            "citation counts and recent publication dates. Find at least 15 papers."
        ),
        backstory=(
            "You are an expert academic librarian who has spent 20 years helping "
            "researchers at MIT and Stanford find the exact papers they need. "
            "You know how to craft precise search queries, identify seminal works, "
            "and find both foundational papers and cutting-edge recent work. "
            "You never miss an important paper in your searches."
        ),
        tools=[search_tool, recommend_tool],
        llm=_llm(),
        verbose=True,
        max_iter=6,
    )


# ── Agent 2: Reader ──────────────────────────────────────────
def create_reader_agent() -> Agent:
    return Agent(
        role="Deep Paper Analyst",
        goal=(
            "For each paper found by the Crawler, extract structured information: "
            "main research claim, methodology used, key findings, datasets, "
            "limitations acknowledged, and what problem it solves."
        ),
        backstory=(
            "You are a PhD researcher who has reviewed thousands of papers for "
            "top-tier conferences like NeurIPS, ICML, and ACL. You can read an "
            "abstract and immediately understand the core contribution, the method, "
            "and the limitations. You produce clean, structured extractions that "
            "other researchers can use to compare papers at a glance."
        ),
        llm=_llm(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 3: Mapper ──────────────────────────────────────────
def create_mapper_agent() -> Agent:
    return Agent(
        role="Research Relationship Mapper",
        goal=(
            "Analyze all papers and map their relationships: which papers agree, "
            "which contradict each other, which build upon others, which use the "
            "same datasets/methods, and which represent competing schools of thought. "
            "Identify clusters of related work."
        ),
        backstory=(
            "You are a meta-researcher and science philosopher who specializes in "
            "understanding how fields evolve. You see the narrative arc of research "
            "— you can look at 20 papers and immediately spot the debates, the "
            "consensus views, the outliers, and the intellectual lineages. Your "
            "relationship maps have helped dozens of PhD students position their "
            "work correctly within their field."
        ),
        llm=_llm(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 4: Gap Finder ──────────────────────────────────────
def create_gap_finder_agent() -> Agent:
    return Agent(
        role="Research Gap Analyst",
        goal=(
            "Based on the paper analysis and relationship map, identify genuine "
            "research gaps — questions that have NOT been answered, combinations "
            "that haven't been tried, populations/domains not studied, and "
            "limitations in existing work that create opportunities for new research. "
            "Rank gaps by importance and feasibility."
        ),
        backstory=(
            "You are a grant reviewer and PhD advisor who has guided 50+ students "
            "to publishable research. You have a sharp eye for what's missing in a "
            "field. You never suggest gaps that are already filled by papers the "
            "student might have missed — you verify. Your gap analyses have led to "
            "papers published in top venues."
        ),
        llm=_llm(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 5: Writer ──────────────────────────────────────────
def create_writer_agent() -> Agent:
    return Agent(
        role="Academic Literature Review Writer",
        goal=(
            "Write a comprehensive, publication-quality literature review section "
            "using the analysis from the other agents. Organize thematically, "
            "synthesize (not just summarize), highlight debates, acknowledge gaps, "
            "and position the student's research within the field. Use proper "
            "academic citation format."
        ),
        backstory=(
            "You are a prolific academic writer with 100+ published papers. You "
            "know exactly how a literature review should be structured: thematic "
            "organization, smooth transitions, synthesis of ideas rather than "
            "mere summary, proper attribution, and a clear narrative that builds "
            "toward the gap your research fills. Your reviews are cited as models "
            "of clarity in academic writing workshops."
        ),
        llm=_llm(),
        verbose=True,
        max_iter=4,
    )


# ── Agent 6: Study Planner ───────────────────────────────────
def create_planner_agent() -> Agent:
    return Agent(
        role="Research Study Planner",
        goal=(
            "Create a realistic, day-by-day study and research plan based on the "
            "papers found and the student's timeline. Prioritize reading the most "
            "cited and most relevant papers first. Schedule time for note-taking, "
            "synthesis, and writing. Include daily milestones."
        ),
        backstory=(
            "You are an academic success coach who specializes in helping grad "
            "students manage overwhelming reading lists. You understand spaced "
            "repetition, deep work scheduling, and how to build from foundational "
            "papers to cutting-edge work. You've helped hundreds of students "
            "go from 'I don't know where to start' to a complete lit review "
            "draft in record time."
        ),
        llm=_llm(),
        verbose=True,
        max_iter=3,
    )
