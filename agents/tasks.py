"""
agents/tasks.py

Defines all tasks for PaperMind's research pipeline + study planner.
Each task feeds into the next — outputs are shared as context.
"""

from crewai import Task
from agents.research_agents import (
    create_crawler_agent,
    create_reader_agent,
    create_mapper_agent,
    create_gap_finder_agent,
    create_writer_agent,
    create_planner_agent,
)


def build_research_tasks(
    topic: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
):
    """
    Build all agents and tasks for the full PaperMind pipeline.

    Returns: (agents_list, tasks_list)
    """

    crawler    = create_crawler_agent()
    reader     = create_reader_agent()
    mapper     = create_mapper_agent()
    gap_finder = create_gap_finder_agent()
    writer     = create_writer_agent()
    planner    = create_planner_agent()

    rq = research_question or f"What are the key developments, debates, and open problems in {topic}?"

    # ── Task 1: Crawl ─────────────────────────────────────────
    crawl_task = Task(
        description=(
            f"Search Semantic Scholar for papers on: **{topic}**\n\n"
            f"Research question to guide your search: {rq}\n\n"
            "Run AT LEAST 3 different search queries to cover the topic from different angles.\n"
            "For example, for 'Agentic AI': search 'agentic AI systems', 'autonomous AI agents', "
            "'multi-agent LLM frameworks'.\n\n"
            "For each search, collect the top results. Then use the recommendation tool on the "
            "2-3 most highly cited papers to find additional related work.\n\n"
            "TARGET: Identify at least 15 relevant papers total.\n\n"
            "Output format for each paper:\n"
            "- Title\n"
            "- Authors + Year\n"
            "- Citation count\n"
            "- Semantic Scholar ID\n"
            "- 1-sentence relevance note\n"
            "- Abstract (truncated to 150 words)"
        ),
        expected_output=(
            "A structured list of 15–25 relevant academic papers with title, authors, "
            "year, citation count, paper ID, and abstract for each. Organized by relevance."
        ),
        agent=crawler,
    )

    # ── Task 2: Read + Extract ───────────────────────────────
    read_task = Task(
        description=(
            f"You have a list of papers on **{topic}** from the Crawler agent.\n\n"
            "For EACH paper, extract the following in a structured format:\n\n"
            "1. **Core Claim** — What is the single main argument/contribution? (1-2 sentences)\n"
            "2. **Methodology** — What approach/technique do they use? (e.g., transformer-based, "
            "   reinforcement learning, survey, empirical study)\n"
            "3. **Key Findings** — What did they prove/show/discover? (2-3 bullet points)\n"
            "4. **Dataset/Benchmark** — What data or evaluation benchmark did they use?\n"
            "5. **Limitations** — What do the authors admit their work doesn't cover?\n"
            "6. **Contribution Type** — Is this: theory / system / empirical / survey / position paper?\n\n"
            "Focus on the abstracts and conclusions provided. Be precise and objective.\n"
            "Do NOT include papers that are clearly irrelevant to the research question."
        ),
        expected_output=(
            "A structured extraction for each paper with: core claim, methodology, "
            "key findings (bullets), dataset used, limitations, and contribution type."
        ),
        agent=reader,
        context=[crawl_task],
    )

    # ── Task 3: Map Relationships ────────────────────────────
    map_task = Task(
        description=(
            f"You have structured extractions of papers on **{topic}**.\n\n"
            "Analyze all papers together and produce a relationship map:\n\n"
            "**A. Agreement Clusters**\n"
            "Group papers that share the same core argument or finding. "
            "Name each cluster (e.g., 'RAG-based approaches', 'Role-based agent frameworks').\n\n"
            "**B. Contradictions & Debates**\n"
            "Identify pairs or groups of papers that contradict each other or represent "
            "competing approaches. Explain what they disagree on specifically.\n\n"
            "**C. Citation Lineage**\n"
            "Identify foundational/seminal papers that most other papers build upon. "
            "Identify the most recent papers pushing the frontier.\n\n"
            "**D. Methodological Landscape**\n"
            "Map which methods/approaches are most common, which are emerging, which are declining.\n\n"
            "**E. Consensus View**\n"
            "What does the field broadly agree on? What is still contested?\n\n"
            "Format as clearly labeled sections with paper titles referenced by [Author, Year]."
        ),
        expected_output=(
            "A relationship map with 5 sections: agreement clusters, contradictions/debates, "
            "citation lineage, methodological landscape, and field consensus summary."
        ),
        agent=mapper,
        context=[crawl_task, read_task],
    )

    # ── Task 4: Find Gaps ────────────────────────────────────
    gap_task = Task(
        description=(
            f"You have a complete analysis of the literature on **{topic}**.\n"
            f"The student's research question is: {rq}\n\n"
            "Identify genuine, actionable research gaps:\n\n"
            "**A. Unexplored Combinations**\n"
            "What combinations of methods/ideas have NOT been tried together?\n\n"
            "**B. Understudied Populations/Domains**\n"
            "What contexts, domains, languages, or populations have existing work ignored?\n\n"
            "**C. Unresolved Contradictions**\n"
            "Where papers disagree — what experiment would settle the debate?\n\n"
            "**D. Limitation Opportunities**\n"
            "What limitations do authors admit? Each limitation = a potential research opportunity.\n\n"
            "**E. Temporal Gaps**\n"
            "What was studied 5+ years ago but hasn't been revisited with modern models/data?\n\n"
            "**F. Top 3 Recommended Gaps**\n"
            "Rank the top 3 most promising gaps for a grad student to pursue. "
            "For each: explain the gap, why it matters, and what a paper addressing it might look like.\n\n"
            "Be specific — cite actual paper titles that make these gaps evident."
        ),
        expected_output=(
            "A gap analysis with 5 gap categories and a ranked top-3 list of the most "
            "promising research opportunities, each with a brief research proposal sketch."
        ),
        agent=gap_finder,
        context=[crawl_task, read_task, map_task],
    )

    # ── Task 5: Write Literature Review ──────────────────────
    write_task = Task(
        description=(
            f"Write a publication-quality **Literature Review** section for a research paper on:\n"
            f"**{topic}**\n\n"
            "Use ALL the analysis from previous agents as your source material.\n\n"
            "Structure:\n"
            "1. **Introduction paragraph** — Motivate the topic, state the scope of this review\n"
            "2. **Thematic sections** (2-4 sections based on the clusters from the Mapper)\n"
            "   - Each section synthesizes a cluster of related work\n"
            "   - Compare and contrast — don't just list papers\n"
            "   - Highlight debates and unresolved questions\n"
            "3. **Methodological Overview** — Brief summary of dominant methods used in the field\n"
            "4. **Research Gaps** — Transition from what exists to what's missing\n"
            "5. **Positioning paragraph** — 'This work addresses the gap by...'\n\n"
            "Citation format: [Author et al., Year] inline.\n"
            "Target length: 600–900 words.\n"
            "Tone: academic but clear. Synthesize, don't just describe.\n\n"
            "IMPORTANT: This must read like a real literature review, not a list of summaries."
        ),
        expected_output=(
            "A complete, well-structured literature review section (600-900 words) organized "
            "thematically with inline citations, synthesis of debates, and a clear gap statement."
        ),
        agent=writer,
        context=[crawl_task, read_task, map_task, gap_task],
    )

    agents = [crawler, reader, mapper, gap_finder, writer]
    tasks  = [crawl_task, read_task, map_task, gap_task, write_task]

    # ── Task 6: Study Plan (optional) ────────────────────────
    if include_planner:
        plan_task = Task(
            description=(
                f"Create a {days}-day reading and research plan for the student studying **{topic}**.\n\n"
                f"Constraints: {days} days available, {hours_per_day} hours/day.\n\n"
                "Use the paper list from the Crawler to build the plan.\n\n"
                "Structure:\n"
                "**Priority 1 (Days 1-2): Foundational Papers**\n"
                "- List the 3-4 most-cited seminal papers — must read first\n"
                "- For each: estimated read time, what to take notes on\n\n"
                "**Priority 2 (Days 3-5): Core Literature**\n"
                "- The main body of relevant papers, grouped by theme\n"
                "- Reading strategy: skim vs deep read for each\n\n"
                "**Priority 3 (Days 6-7): Synthesis & Writing**\n"
                "- Day 6: Organize notes, build argument structure\n"
                "- Day 7: Draft the literature review using the generated draft as a starting point\n\n"
                "**Daily Schedule Template:**\n"
                "Morning (X hrs): [specific activity]\n"
                "Afternoon (X hrs): [specific activity]\n"
                "End of day: [milestone checkpoint]\n\n"
                "Also include: recommended note-taking method, "
                "tools to use (Zotero, Obsidian, etc.), and weekly milestone."
            ),
            expected_output=(
                "A complete day-by-day reading and research plan with prioritized paper list, "
                "daily schedule template, reading strategies, and milestone checkpoints."
            ),
            agent=planner,
            context=[crawl_task, gap_task],
        )
        agents.append(planner)
        tasks.append(plan_task)

    return agents, tasks
