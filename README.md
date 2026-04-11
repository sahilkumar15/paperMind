# 🔬 ScholarMind: Research Intelligence Platform

> **Yeshiva University · Katz School of Science and Health**  
> Katz School CS & AI Club Ideathon 2026 · Sponsored by Google Developer Groups

---

## What is ScholarMind?

ScholarMind is a complete AI-powered academic research platform that takes a research topic and delivers:

- **Verified literature review** — cited, academic prose, ready to submit
- **Exact BibTeX citations** — from Semantic Scholar API, paste into Overleaf
- **Research gap analysis** — 3 ranked gaps with feasibility scores
- **Faculty matching** — which Katz professor to email, with their actual email
- **Live campus events** — relevant seminars and workshops from yu.edu/katz
- **Smart Advisor** — connects gap → faculty → event → email template in one click
- **KatzBot RAG** — trained on the real Katz School website, persisted to disk

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a FREE Groq API key
Visit **https://console.groq.com** → Sign up → API Keys → Create Key  
Key starts with `gsk_`

### 3. Configure
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Pre-build KatzBot index (optional but recommended)
```bash
python katzbot/build_index.py --refresh --test
```
This takes ~5 minutes on first run. Subsequent starts load from disk in ~2 seconds.

### 5. Run
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Project Structure

```
scholarmind/
├── app.py                      ← Main Streamlit UI (12 tabs)
├── crew.py                     ← 6-agent pipeline orchestrator
├── chatbot.py                  ← PaperBot direct chat
├── llm_config.py               ← Groq/OpenAI configuration
├── requirements.txt
├── .env.example                ← Copy to .env
│
├── agents/
│   ├── research_agents.py      ← 6 CrewAI agents
│   └── tasks.py                ← Task definitions
│
├── tools/
│   ├── semantic_scholar.py     ← Paper search tool
│   └── citation_fetcher.py     ← BibTeX generation
│
└── katzbot/
    ├── __init__.py
    ├── build_index.py           ← CLI: build FAISS index
    ├── crawler.py               ← Web crawler (sitemap + parallel)
    ├── indexer.py               ← FAISS builder + persistence
    ├── chain.py                 ← RAG chain (version-safe)
    ├── rag_engine.py            ← Main engine (auto-build)
    ├── faculty.py               ← Faculty database (10 professors)
    ├── events_fetcher.py        ← Live events from yu.edu/katz
    └── smart_advisor.py         ← Gap → Faculty → Event → Email
```

---

## KatzBot Persistence

On first run, KatzBot:
1. Parses `yu.edu/sitemap.xml` (original notebook found 1,262 URLs)
2. Filters and fetches Katz pages in parallel (8 workers)
3. Loads QA dataset from GitHub (key to original ROUGE-1 F1=0.367)
4. Injects all 10 faculty as structured documents
5. Builds FAISS index → saved to `katzbot/faiss_index/`
6. Saves retriever → `katzbot/retriever_store.pkl`

**Subsequent runs**: loads from disk in ~2 seconds. No rebuild needed.

To force rebuild:
```bash
python katzbot/build_index.py --refresh
```

---

## Groq Free Tier — Model Recommendations

| Model | TPM | Recommended? |
|-------|-----|-------------|
| `llama-3.1-8b-instant` | 250,000 | ✅ **Default — always works** |
| `llama-3.3-70b-versatile` | 12,000 | ⚠️ May rate-limit |
| `qwen/qwen3-32b` | 6,000 | ❌ Avoid with 6 agents |

---

## Why ScholarMind Beats ChatGPT, Elicit, and Perplexity

| Feature | ChatGPT | Elicit | ScholarMind |
|---------|---------|--------|-------------|
| Verified citations (no hallucinations) | ❌ | ✅ partial | ✅ Semantic Scholar API |
| 6-agent specialised pipeline | ❌ | ❌ | ✅ |
| Institution-specific RAG | ❌ | ❌ | ✅ Katz School |
| Faculty matching with emails | ❌ | ❌ | ✅ |
| Live campus events integration | ❌ | ❌ | ✅ |
| Complete workflow (paper→gap→email) | ❌ | ❌ | ✅ |
| Cost | Paid | Paid | **$0** |

---

## Project Title Origin

**ScholarMind** reflects the three pillars:
- **Scholar** — academic research, literature, citations
- **Mind** — AI intelligence, agent reasoning, knowledge synthesis
- The combination → an AI that thinks like a scholar

*Alternative names considered: AcademIQ, ResearchNexus, KatzScholar, MindMap Research*
