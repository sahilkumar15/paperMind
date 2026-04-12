"""
crew.py — KatzScholarMind orchestrator
======================================

Key updates:
- Search is deterministic Python now, not LLM tool-calling.
- This avoids Groq function-call failures such as hallucinated search tools.
- Semantic Scholar is primary; arXiv is the free fallback.
- For selected Groq models, paper context is processed in chunks first to reduce
  prompt size / TPM spikes / long-context failures.

Groq chunking flow:
1. Retrieve papers deterministically in Python
2. Split papers into chunks
3. Run only the curator task on each chunk
4. Merge the curated outputs
5. Run the full pipeline once on the merged shortlist
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODELS = {
    "llama-3.3-70b-versatile": "⭐ Best quality · 300K TPM · recommended",
    "llama-3.1-8b-instant": "⚡ Fastest · 250K TPM · 14,400 req/day",
    "openai/gpt-oss-120b": "🔥 GPT-class 120B · 250K TPM",
    "openai/gpt-oss-20b": "🚀 GPT-class fast · 1000 t/s",
    "meta-llama/llama-4-scout-17b-16e-instruct": "🦙 Llama 4 Scout · 300K TPM",
    "qwen/qwen3-32b": "🧠 Qwen3 32B · strong reasoning",
}

DEFAULT_MAX_RESULTS = 12
GROQ_CHUNK_SIZE = 4
GROQ_FINAL_MAX_SECTIONS = 12


def _safe_task_output(task) -> str:
    """Extract CrewAI task output safely."""
    try:
        if task.output:
            if hasattr(task.output, "raw") and task.output.raw:
                return task.output.raw
            if hasattr(task.output, "result") and task.output.result:
                return str(task.output.result)
            return str(task.output)
    except Exception as e:
        return f"[Error extracting output: {e}]"
    return ""


def _chunk_list(items, chunk_size):
    """Split a list into fixed-size chunks."""
    if chunk_size <= 0:
        chunk_size = GROQ_CHUNK_SIZE
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _should_chunk_for_groq(provider: str, model: str, papers_count: int) -> bool:
    """Enable chunking only for known Groq models and sufficiently large paper sets."""
    return provider == "groq" and model in GROQ_MODELS and papers_count > GROQ_CHUNK_SIZE


def _truncate_merged_curations(merged_text: str, max_sections: int = GROQ_FINAL_MAX_SECTIONS) -> str:
    """
    Keep merged curator output reasonably small before the final full pipeline.
    We conservatively split on blank lines and keep the earliest sections.
    """
    if not merged_text.strip():
        return ""

    sections = [s.strip() for s in merged_text.split("\n\n") if s.strip()]
    if len(sections) <= max_sections:
        return merged_text

    return "\n\n".join(sections[:max_sections])


import time

import time
from litellm import RateLimitError

def safe_kickoff(crew):
    for attempt in range(3):
        try:
            return crew.kickoff()
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait = 5 + attempt * 2
                print(f"[Retry] Rate limit hit. Sleeping {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError("Max retries exceeded")

def _run_single_task_crew(agent, task):
    from crewai import Crew, Process

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,
        embedder=None,
    )

    result = safe_kickoff(crew)

    # 🔥 ADD THIS
    time.sleep(4)   # safe buffer (Groq suggested ~3.3s)

    return result


def _run_chunked_curation(
    topic: str,
    paper_chunks: list,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> str:
    """
    First-pass chunk curation:
    - For each chunk, build normal tasks
    - Run only task 0 (curation)
    - Merge curated text across chunks
    """
    from agents.tasks import build_research_tasks
    from tools.paper_search import format_papers_for_prompt

    merged_outputs = []

    print(f"[Groq chunk mode] Running first-pass curation over {len(paper_chunks)} chunk(s)")

    for idx, chunk in enumerate(paper_chunks, start=1):
        chunk_prompt = format_papers_for_prompt(chunk)

        agents, tasks = build_research_tasks(
            topic=topic,
            papers_context=chunk_prompt,
            research_question=research_question,
            days=days,
            hours_per_day=hours_per_day,
            include_planner=include_planner,
        )

        curator = agents[0]
        curate_task = tasks[0]

        print(
            f"[Groq chunk mode] Curating chunk {idx}/{len(paper_chunks)} "
            f"with {len(chunk)} paper(s)"
        )

        _run_single_task_crew(curator, curate_task)

        curated_text = _safe_task_output(curate_task).strip()
        if curated_text:
            merged_outputs.append(f"### Curated chunk {idx}\n{curated_text}")

    return "\n\n".join(merged_outputs).strip()


def _run_full_pipeline(
    topic: str,
    papers_context: str,
    papers_display: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> dict:
    """Run the normal full sequential CrewAI pipeline."""
    from crewai import Crew, Process
    from agents.tasks import build_research_tasks

    agents, tasks = build_research_tasks(
        topic=topic,
        papers_context=papers_context,
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
        embedder=None,
    )

    result = crew.kickoff()

    outputs = {
        "papers": papers_display,
        "curated": _safe_task_output(tasks[0]) if len(tasks) > 0 else "",
        "extractions": _safe_task_output(tasks[1]) if len(tasks) > 1 else "",
        "map": _safe_task_output(tasks[2]) if len(tasks) > 2 else "",
        "gaps": _safe_task_output(tasks[3]) if len(tasks) > 3 else "",
        "lit_review": _safe_task_output(tasks[4]) if len(tasks) > 4 else "",
        "raw": str(result) if result else "",
    }

    if include_planner:
        outputs["study_plan"] = _safe_task_output(tasks[5]) if len(tasks) > 5 else ""
    else:
        outputs["study_plan"] = ""

    return outputs


def run_papermind(
    topic: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> dict:
    from llm_config import get_provider, get_model_name, get_crewai_llm_string
    from tools.paper_search import search_papers, format_papers_for_display, format_papers_for_prompt

    provider = get_provider()
    model = get_model_name()
    llm_str = get_crewai_llm_string()

    if provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    print(f"\n{'='*60}")
    print(f"  KatzScholarMind | Provider: {provider.upper()} | Model: {model}")
    print(f"  LiteLLM string: {llm_str}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    # Deterministic retrieval first.
    papers = search_papers(topic, max_results=DEFAULT_MAX_RESULTS)
    papers_display = format_papers_for_display(papers)
    papers_prompt = format_papers_for_prompt(papers)

    if not papers:
        return {
            "papers": papers_display,
            "curated": "No papers found.",
            "extractions": "No papers found.",
            "map": "No papers found.",
            "gaps": "No papers found.",
            "lit_review": "No papers found.",
            "study_plan": "No papers found." if include_planner else "",
            "raw": "No papers found from Semantic Scholar or arXiv.",
            "chunk_mode": False,
            "chunked_context": "",
        }

    # Groq chunked pathway.
    if _should_chunk_for_groq(provider, model, len(papers)):
        print("[Groq chunk mode] Enabled for this model.")

        paper_chunks = _chunk_list(papers, GROQ_CHUNK_SIZE)

        merged_curated = _run_chunked_curation(
            topic=topic,
            paper_chunks=paper_chunks,
            research_question=research_question,
            days=days,
            hours_per_day=hours_per_day,
            include_planner=include_planner,
        )

        if not merged_curated.strip():
            print("[Groq chunk mode] First-pass curation returned empty output. Falling back to raw paper prompt.")
            final_context = papers_prompt
        else:
            final_context = _truncate_merged_curations(
                merged_curated,
                max_sections=GROQ_FINAL_MAX_SECTIONS,
            )

        outputs = _run_full_pipeline(
            topic=topic,
            papers_context=final_context,
            papers_display=papers_display,
            research_question=research_question,
            days=days,
            hours_per_day=hours_per_day,
            include_planner=include_planner,
        )
        outputs["chunk_mode"] = True
        outputs["chunked_context"] = final_context
        return outputs

    # Default non-chunked pathway.
    outputs = _run_full_pipeline(
        topic=topic,
        papers_context=papers_prompt,
        papers_display=papers_display,
        research_question=research_question,
        days=days,
        hours_per_day=hours_per_day,
        include_planner=include_planner,
    )
    outputs["chunk_mode"] = False
    outputs["chunked_context"] = ""
    return outputs


if __name__ == "__main__":
    result = run_papermind(
        topic="Agentic AI and Multi-Agent Systems",
        research_question="What are the key architectures and unsolved problems?",
        days=7,
        hours_per_day=3,
    )

    print("\n=== PAPERS ===")
    print(result.get("papers", "No output"))

    print("\n=== CURATED ===")
    print(result.get("curated", "No output"))

    print("\n=== LIT REVIEW DRAFT ===")
    print(result.get("lit_review", "No output"))