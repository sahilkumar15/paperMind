# 🔬 PaperMind — Katz School AI Research Platform

> Multi-agent AI that reads 20+ papers, maps relationships, finds gaps,
> writes your literature review, and connects you with the right Katz professor.
> Powered by **Groq (FREE)** or OpenAI.

Built for **Katz School Ideathon 2026** × Google Developer Groups.

---

## 🐛 Bugs Fixed in This Version

### Fix 1 — `GroqException: Failed to call a function`
**Root cause:** CrewAI passes tool/function-call JSON schemas to all agents. Groq's 
Llama models reject malformed schemas. LiteLLM was also not receiving the `GROQ_API_KEY` 
reliably before the crew started.

**Fixes applied:**
- Only the **Crawler agent** gets tools (the only one that actually calls Semantic Scholar)
- All other 5 agents have `tools=[]` — they read context passed from previous tasks
- `allow_delegation=False` on all agents prevents extra tool-calling attempts
- `crew.py` explicitly sets `os.environ["GROQ_API_KEY"]` before `crew.kickoff()`
- `memory=False` and `embedder=None` in Crew config prevents Groq embeddings API calls
- `max_iter=3` caps retry loops when Groq rate-limits

### Fix 2 — `No module named 'langchain_chroma'`
**Root cause:** `langchain-chroma` is a newer standalone package that may not be 
installed despite being in requirements.

**Fix applied:**
```python
# katzbot/rag_engine.py now does:
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback
```
This works with any langchain version.

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your FREE Groq API key
Go to **https://console.groq.com** → Sign up → API Keys → Create Key  
Key starts with `gsk_`

### 3. Create .env file
```bash
cp .env.example .env
```
Edit `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
MODEL_NAME=llama-3.3-70b-versatile
OPENAI_API_KEY=          # optional — leave blank to use free embeddings
```

### 4. Run
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 🗂️ Project Structure

```
papermind/
├── app.py                  ← Run this (Streamlit UI)
├── crew.py                 ← Agent orchestrator (FIXED)
├── chatbot.py              ← PaperBot chat
├── llm_config.py           ← Groq/OpenAI switcher (FIXED)
├── .env.example            ← Copy to .env
├── requirements.txt        ← Updated dependencies
├── agents/
│   ├── __init__.py
│   ├── research_agents.py  ← Agent definitions (FIXED — tools only on Crawler)
│   └── tasks.py            ← Task definitions (FIXED — context chaining)
├── tools/
│   ├── __init__.py
│   └── semantic_scholar.py ← Search tool (FIXED — simple string signature)
└── katzbot/
    ├── __init__.py
    ├── rag_engine.py       ← RAG pipeline (FIXED — langchain_chroma fallback)
    └── chroma_index/       ← Auto-created on first KatzBot use
```

---

## 🤖 Groq Model Recommendations

| Model | Speed | Best For |
|-------|-------|----------|
| `llama-3.3-70b-versatile` | 280 t/s | **Recommended** — best quality |
| `llama-3.1-8b-instant` | 560 t/s | If hitting rate limits — fastest |
| `openai/gpt-oss-120b` | 500 t/s | GPT-quality output on Groq |
| `qwen/qwen3-32b` | 400 t/s | Strong reasoning tasks |

**Rate limits on free Groq tier:**
- llama-3.3-70b: 300K tokens/min, 1K req/min
- llama-3.1-8b: 250K tokens/min, 14,400 req/day

If you get a rate limit error → switch to `llama-3.1-8b-instant` in the sidebar.

---

## 🎓 KatzBot First-Time Setup

1. Click **🎓 KatzBot** tab
2. Click **"Build / Refresh Index"**
3. Wait ~5 minutes (crawls ~60 Katz pages, saved to disk)
4. All future runs load instantly from the cached index

---

## 🚨 Troubleshooting

| Error | Fix |
|-------|-----|
| `GroqException: Failed to call a function` | Fixed in this version — update your files |
| `No module named 'langchain_chroma'` | Fixed in this version — auto-fallback to community |
| `GROQ_API_KEY not found` | Check `.env` file — key must start with `gsk_` |
| `Rate limit exceeded` | Switch to `llama-3.1-8b-instant` in sidebar |
| `Module not found: langchain_groq` | `pip install langchain-groq --upgrade` |
| `KatzBot build fails` | `pip install beautifulsoup4 lxml chromadb --upgrade` |
| Agents produce empty output | Try `llama-3.1-8b-instant` (more daily quota) |