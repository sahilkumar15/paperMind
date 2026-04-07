# 🔬 PaperMind — Katz School AI Research Platform

> Multi-agent AI that reads 20+ papers, maps relationships, finds gaps,
> writes your literature review, and connects you with the right Katz professor.
> Powered by **Groq (FREE)** or OpenAI.

Built for **Katz School Ideathon 2026** × Google Developer Groups.

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

## 🤖 Groq vs OpenAI

| | Groq (Free) | OpenAI (Paid) |
|--|--|--|
| Cost | **$0** | ~$0.025/run |
| Model | Llama 3.3 70B | GPT-4o-mini |
| Rate limit | 500 req/day | No limit |
| KatzBot embeddings | HuggingFace (free) | text-embedding-3-small |

**Use Groq** — it's free, fast, and same quality as GPT-4o-mini.

---

## 🗂️ Tabs

| Tab | What it does |
|-----|-------------|
| ⚡ Run Agents | 6 agents run automatically |
| 📄 Papers Found | 15–25 real papers from Semantic Scholar |
| 🗺️ Relationship Map | Agreements, contradictions, citation lineage |
| 🎯 Research Gaps | Top 3 unexplored opportunities |
| ✍️ Lit Review Draft | 700-word publication-ready literature review |
| 📅 Study Plan | Day-by-day reading schedule |
| 💬 PaperBot | Chat with full paper context |
| 🎓 KatzBot | RAG chatbot over yu.edu/katz |
| 👨‍🏫 Faculty Match | Auto-match Katz professors to your topic |

---

## 🏗️ Project Structure

```
papermind/
├── app.py               ← Run this
├── crew.py              ← Agent orchestrator
├── chatbot.py           ← PaperBot chat
├── llm_config.py        ← Groq/OpenAI switcher (NEW)
├── .env.example         ← Copy to .env
├── requirements.txt
├── agents/
│   ├── research_agents.py
│   └── tasks.py
├── tools/
│   └── semantic_scholar.py
└── katzbot/
    ├── rag_engine.py    ← RAG pipeline
    └── chroma_index/    ← Auto-created
```

---

## 🎓 KatzBot First-Time Setup

1. Click **🎓 KatzBot** tab
2. Click **"Build / Refresh Index"**
3. Wait ~5 minutes (crawls 60 Katz pages once, saved to disk)
4. Done — all future runs are instant

---

## 🚨 Troubleshooting

**"GROQ_API_KEY not found"** → Check .env file exists and key starts with `gsk_`

**"Rate limit exceeded" on Groq** → Switch model to `llama-3.1-8b-instant` (14,400 req/day)

**"Module not found: langchain_groq"** → `pip install langchain-groq --upgrade`

**Agents produce empty output** → Check terminal, try switching to `gemma2-9b-it`

**KatzBot build fails** → `pip install unstructured lxml beautifulsoup4 --upgrade`
