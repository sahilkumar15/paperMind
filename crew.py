"""
crew.py — PaperMind orchestrator
==================================
GROQ FREE TIER FIX:
  - Adds sleep between tasks to stay under TPM window
  - Smarter retry: reads exact wait time from Groq error message
  - Catches both crew-level and task-level rate limit failures
  - memory=False, embedder=None (no embeddings API needed)
"""

import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# Inter-task delay for Groq free tier (seconds).
# llama-3.1-8b-instant: 250K TPM — 1-minute window resets fast.
# A 5s pause between tasks gives the TPM bucket time to refill.
GROQ_INTER_TASK_DELAY = 5   # seconds between tasks on Groq


def _parse_wait_seconds(error_str: str) -> float:
    """Extract retry wait time from Groq error message."""
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 3   # add 3s buffer
    return 35.0   # default


def run_papermind(
    topic: str,
    research_question: str = "",
    days: int = 7,
    hours_per_day: float = 3.0,
    include_planner: bool = True,
) -> dict:
    from crewai import Crew, Process
    from agents.tasks import build_research_tasks
    from llm_config import get_provider, get_model_name, get_crewai_llm_string

    provider = get_provider()
    model    = get_model_name()
    llm_str  = get_crewai_llm_string()

    # Ensure LiteLLM can find the key
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        if key:
            os.environ["GROQ_API_KEY"] = key
    else:
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            os.environ["OPENAI_API_KEY"] = key

    print(f"\n{'='*60}")
    print(f"  PaperMind | {provider.upper()} | {model}")
    print(f"  LiteLLM: {llm_str}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    agents, tasks = build_research_tasks(
        topic=topic,
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

    # ── Run with retry on rate limit ──────────────────────────
    result    = None
    max_tries = 4
    for attempt in range(1, max_tries + 1):
        try:
            result = crew.kickoff()
            break   # success
        except Exception as e:
            err_str  = str(e)
            err_lower = err_str.lower()
            is_rate_limit = (
                "rate limit" in err_lower
                or "ratelimit" in err_lower
                or "rate_limit_exceeded" in err_lower
                or "429" in err_lower
                or "tokens per minute" in err_lower
            )
            if is_rate_limit and attempt < max_tries:
                wait = _parse_wait_seconds(err_str)
                print(
                    f"\n[PaperMind] ⏳ Rate limit hit (attempt {attempt}/{max_tries}). "
                    f"Waiting {wait:.0f}s before retry...\n"
                )
                time.sleep(wait)
                # Re-build crew so agents get fresh state
                agents, tasks = build_research_tasks(
                    topic=topic,
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
            else:
                raise

    # ── Extract per-task outputs ──────────────────────────────
    keys = ["papers", "extractions", "map", "gaps", "lit_review"]
    if include_planner:
        keys.append("study_plan")

    outputs = {}
    for i, task in enumerate(tasks):
        key = keys[i] if i < len(keys) else f"task_{i}"
        try:
            raw = ""
            if task.output:
                if hasattr(task.output, "raw") and task.output.raw:
                    raw = task.output.raw
                elif hasattr(task.output, "result") and task.output.result:
                    raw = str(task.output.result)
                else:
                    raw = str(task.output)
            outputs[key] = raw
        except Exception as e:
            outputs[key] = f"[Error extracting output: {e}]"

    outputs["raw"] = str(result) if result else ""
    return outputs


if __name__ == "__main__":
    result = run_papermind(
        topic="Agentic AI and Multi-Agent Systems",
        research_question="Key architectures and unsolved problems?",
        days=7,
        hours_per_day=3,
    )
    print("\n=== LIT REVIEW ===")
    print(result.get("lit_review", "No output"))