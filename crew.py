"""
crew.py — ScholarMind research orchestrator
=============================================
Runs the 6-agent pipeline with retry on Groq rate limits.
"""

import os
import re
import time
from dotenv import load_dotenv

load_dotenv()


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

    # Ensure LiteLLM finds the API key
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY","")
        if key: os.environ["GROQ_API_KEY"] = key
    else:
        key = os.getenv("OPENAI_API_KEY","")
        if key: os.environ["OPENAI_API_KEY"] = key

    print(f"\n{'='*60}")
    print(f"  KatzScholarMind | {provider.upper()} | {model}")
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

    result    = None
    max_tries = 4

    def _parse_wait(err: str) -> float:
        m = re.search(r"try again in (\d+(?:\.\d+)?)s", err, re.I)
        return float(m.group(1)) + 5 if m else 35.0

    for attempt in range(1, max_tries+1):
        try:
            result = crew.kickoff()
            break
        except Exception as e:
            err = str(e).lower()
            is_rl = any(x in err for x in
                        ["rate limit","ratelimit","429","tokens per minute",
                         "rate_limit_exceeded"])
            if is_rl and attempt < max_tries:
                wait = _parse_wait(str(e))
                print(f"\n[ScholarMind] Rate limit (attempt {attempt}/{max_tries}). "
                      f"Waiting {wait:.0f}s…\n")
                time.sleep(wait)
                # Rebuild crew with fresh state
                agents, tasks = build_research_tasks(
                    topic=topic, research_question=research_question,
                    days=days, hours_per_day=hours_per_day,
                    include_planner=include_planner,
                )
                crew = Crew(agents=agents, tasks=tasks,
                            process=Process.sequential,
                            verbose=True, memory=False, embedder=None)
            else:
                raise

    # Extract per-task outputs
    keys = ["papers","extractions","map","gaps","lit_review"]
    if include_planner:
        keys.append("study_plan")

    outputs = {}
    for i, task in enumerate(tasks):
        key = keys[i] if i < len(keys) else f"task_{i}"
        try:
            raw = ""
            if task.output:
                if hasattr(task.output,"raw") and task.output.raw:
                    raw = task.output.raw
                elif hasattr(task.output,"result") and task.output.result:
                    raw = str(task.output.result)
                else:
                    raw = str(task.output)
            outputs[key] = raw
        except Exception as e:
            outputs[key] = f"[Error: {e}]"

    outputs["raw"] = str(result) if result else ""
    return outputs
