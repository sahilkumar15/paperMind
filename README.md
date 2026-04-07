# 🔬 PaperMind — Agentic Research Synthesis AI

> **The real problem:** A CS grad student spends 15–30 hours on a single literature review.
> **PaperMind cuts that to under 2 hours** using 6 specialized AI agents that autonomously read,
> map, analyze, and synthesize academic papers.

Built for **Katz School of Science and Health Ideathon 2026** × Google Developer Groups.

---

## Why This Wins

| What judges ask | Your answer |
|---|---|
| Is the problem real? | Every grad student spends weeks on lit reviews. It's universally painful. |
| Is it innovative? | No tool autonomously maps *contradictions between papers* — that's the fresh angle. |
| Is it agentic AI? | 6 specialized agents, each with a distinct role. True multi-agent orchestration. |
| Can it be built? | Live demo in 7 days. Semantic Scholar API is free. |
| Who benefits? | Every researcher on earth. Scales from Katz to every university globally. |

---

## Architecture — 6 Agents

```
User Input (topic + research question)
             │
             ▼
    ┌─────────────────┐
    │  Crawler Agent  │  ← Searches Semantic Scholar (200M+ papers, FREE API)
    └────────┬────────┘
             │ 15–25 papers
             ▼
    ┌─────────────────┐
    │  Reader Agent   │  ← Extracts: claim, method, findings, limitations
    └────────┬────────┘
             │ structured extractions
             ▼
    ┌─────────────────┐
    │  Mapper Agent   │  ← Finds agreements, contradictions, citation lineage
    └────────┬────────┘
             │ relationship map
             ▼
    ┌──────────────────┐
    │ Gap Finder Agent │  ← Identifies unexplored research opportunities
    └────────┬─────────┘
             │ ranked gap list
             ▼
    ┌─────────────────┐
    │  Writer Agent   │  ← Drafts the literature review section
    └────────┬────────┘
             │
    ┌─────────────────┐
    │ Planner Agent   │  ← Builds day-by-day reading schedule
    └─────────────────┘
             │
    ┌─────────────────────────────────┐
    │  PaperBot Chatbot               │  ← Q&A using all outputs as context
    └─────────────────────────────────┘
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Set up keys
```bash
cp .env.example .env
# Add your OpenAI key — Semantic Scholar needs NO key
```

### 3. Run
```bash
streamlit run app.py
```

---

## What It Costs

| Component | Cost per full run |
|---|---|
| Crawler Agent (gpt-4o-mini) | ~$0.003 |
| Reader Agent | ~$0.005 |
| Mapper Agent | ~$0.004 |
| Gap Finder | ~$0.004 |
| Writer Agent | ~$0.005 |
| Planner Agent | ~$0.003 |
| **Total** | **~$0.025 (2.5 cents)** |

Semantic Scholar API: **completely free**, 200M+ papers, no key needed.

---

## Project Structure

```
papermind/
├── app.py                    ← Streamlit UI (run this)
├── crew.py                   ← CrewAI orchestrator
├── chatbot.py                ← PaperBot conversational AI
├── requirements.txt
├── .env.example
├── agents/
│   ├── research_agents.py    ← All 6 agent definitions
│   └── tasks.py              ← All task definitions with detailed prompts
└── tools/
    └── semantic_scholar.py   ← Free paper search API wrapper
```

---

## Your 10-Minute Pitch Script

**[0:00–1:30] The Problem**
"Show of hands — how many of you have spent more than 10 hours writing a literature review?
That pain is universal. The average CS grad student spends 15–30 hours per lit review.
They're manually reading papers, copy-pasting quotes, losing track of which paper said what,
and missing important related work. There's no tool that *thinks across* papers."

**[1:30–3:30] The Insight**
"The key insight is this: what makes a great literature review isn't reading papers —
it's understanding the *relationships between* papers. Which ones agree? Which contradict?
What hasn't been tried? That's the hard part. And it's exactly what AI agents are built for."

**[3:30–6:30] Live Demo** ← Run PaperMind live on stage
"Watch what happens when I enter a research topic…
[run agents or show pre-generated results]
The Crawler found 20 papers. The Reader extracted structured information from each.
The Mapper found 3 major debates in this field. The Gap Finder identified 5 unexplored areas.
The Writer produced a 700-word literature review draft — in 4 minutes."

**[6:30–8:30] Impact & Feasibility**
"Who benefits: every grad student, researcher, and PhD candidate on earth.
The technology: CrewAI for agent orchestration, Semantic Scholar's free API for papers,
OpenAI for reasoning. Total cost: 2.5 cents per run.
Scalability: deploy as a university service, charge $5/month per student."

**[8:30–10:00] Q&A Buffer**
Likely questions:
- "How is this different from ChatGPT?" → ChatGPT answers questions. PaperMind *acts* — it autonomously searches, reads, and synthesizes without being told what to do at each step.
- "What about hallucinations?" → We ground every claim in real Semantic Scholar data. The lit review cites real papers with real IDs.
- "Can you scale this?" → Yes. Swap gpt-4o-mini for a local model to reduce cost to near zero.

---

## The One Line That Wins the Room

> *"ChatGPT tells you about papers. PaperMind reads them for you, finds what's missing,
> and hands you a literature review draft — so you can spend your time doing research,
> not reading about it."*
