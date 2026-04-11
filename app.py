"""
app.py — PaperMind + KatzBot  |  Yeshiva University Katz School
=================================================================
Tabs:
  1. ⚡ Run Agents       — 6-agent research pipeline
  2. 📄 Papers Found     — Semantic Scholar results
  3. 🗺️ Relationship Map — agreements / contradictions
  4. 🎯 Research Gaps    — unexplored opportunities
  5. ✍️ Lit Review Draft — ready-to-paste literature review
  6. 📅 Study Plan       — day-by-day reading schedule
  7. 💬 PaperBot         — research Q&A (paper context)
  8. 🎓 KatzBot          — RAG chatbot over yu.edu/katz
  9. 👨‍🏫 Faculty Match    — find the right Katz professor

Run:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PaperMind — Katz School AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
:root{--ink:#0D1117;--ink2:#161B22;--border:#30363D;--muted:#8B949E;--text:#E6EDF3;--accent:#58A6FF;--gold:#D29922;--green:#3FB950;--red:#F85149;--purple:#BC8CFF}
.stApp{background:#0D1117;color:#E6EDF3}
section[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D}
section[data-testid="stSidebar"] *{color:#E6EDF3!important}
.pm-header{padding:2.5rem 0 2rem;border-bottom:1px solid #30363D;margin-bottom:2rem}
.pm-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:#58A6FF;margin-bottom:.6rem}
.pm-title{font-family:'DM Serif Display',serif;font-size:2.8rem;color:#E6EDF3;line-height:1.1;margin:0 0 .5rem}
.pm-title em{color:#58A6FF;font-style:italic}
.pm-sub{font-size:1rem;color:#8B949E;font-weight:300;max-width:600px;line-height:1.6}
.agent-pipeline{display:flex;flex-direction:column;gap:6px;margin:1rem 0}
.agent-row{display:flex;align-items:center;gap:12px;background:#161B22;border:1px solid #30363D;border-radius:8px;padding:10px 14px;transition:border-color .2s}
.agent-row.active{border-color:#D29922;background:#1C1A10}
.agent-row.done{border-color:#3FB950;background:#0D1F12}
.agent-row.waiting{opacity:.55}
.agent-num{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#8B949E;min-width:20px}
.agent-name{font-size:.88rem;font-weight:500;color:#E6EDF3}
.agent-desc{font-size:.78rem;color:#8B949E;margin-left:auto}
.dot{width:8px;height:8px;border-radius:50%;background:#30363D;flex-shrink:0}
.dot.active{background:#D29922;animation:pulse 1s infinite}
.dot.done{background:#3FB950}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.stats-row{display:flex;gap:8px;flex-wrap:wrap;margin:1.2rem 0}
.stat-chip{background:#161B22;border:1px solid #30363D;border-radius:20px;padding:4px 14px;font-size:.82rem;color:#8B949E;font-family:'JetBrains Mono',monospace}
.stat-chip b{color:#E6EDF3}
.stTabs [data-baseweb="tab-list"]{background:#161B22;border-bottom:1px solid #30363D;gap:0;padding:0}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif;font-size:.82rem;font-weight:500;color:#8B949E;padding:.7rem 1.1rem;border-bottom:2px solid transparent;border-radius:0}
.stTabs [aria-selected="true"]{color:#58A6FF!important;border-bottom:2px solid #58A6FF!important;background:transparent!important}
.content-card{background:#161B22;border:1px solid #30363D;border-radius:10px;padding:1.4rem 1.6rem;margin:.8rem 0}
.card-label{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#8B949E;margin-bottom:.5rem}
.stButton>button{background:#1F6FEB!important;color:white!important;border:none!important;border-radius:6px!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important;font-size:.88rem!important;padding:.55rem 1.8rem!important;transition:background .2s!important}
.stButton>button:hover{background:#388BFD!important}
.stButton>button:disabled{background:#21262D!important;color:#484F58!important}
.bubble-user{background:#1F6FEB;color:white;border-radius:14px 14px 3px 14px;padding:.65rem 1rem;margin:.4rem 0 .4rem 3.5rem;font-size:.9rem;line-height:1.55}
.bubble-bot{background:#1C2128;border:1px solid #30363D;color:#E6EDF3;border-radius:14px 14px 14px 3px;padding:.65rem 1rem;margin:.4rem 3.5rem .4rem 0;font-size:.9rem;line-height:1.55}
.chat-who{font-size:.7rem;font-family:'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:#8B949E;margin:.8rem 0 .2rem}
.stTextInput input,.stTextArea textarea{background:#161B22!important;border:1px solid #30363D!important;color:#E6EDF3!important;border-radius:6px!important}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:#58A6FF!important;box-shadow:none!important}
label{color:#8B949E!important;font-size:.82rem!important}
/* Faculty cards */
.fac-card{background:#161B22;border:1px solid #30363D;border-left:4px solid #58A6FF;border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin-bottom:.8rem}
.fac-name{font-size:1rem;font-weight:600;color:#E6EDF3;margin-bottom:.2rem}
.fac-title{font-size:.78rem;color:#8B949E;margin-bottom:.4rem}
.fac-dept{font-size:.72rem;font-family:'JetBrains Mono',monospace;color:#58A6FF;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem}
.fac-expertise{font-size:.82rem;color:#C9D1D9;line-height:1.5;margin-bottom:.5rem}
.fac-contact a{font-size:.82rem;color:#58A6FF;text-decoration:none}
.fac-contact a:hover{text-decoration:underline}
.fac-note{font-size:.75rem;color:#8B949E;font-style:italic;margin-top:.3rem}
/* KatzBot source pills */
.source-pill{display:inline-block;background:#161B22;border:1px solid #30363D;border-radius:20px;padding:2px 10px;font-size:.72rem;color:#8B949E;margin:2px;font-family:'JetBrains Mono',monospace}
.stMarkdown h1,h2,h3{color:#E6EDF3;font-family:'DM Serif Display',serif}
.stMarkdown p{color:#C9D1D9;line-height:1.7}
.stMarkdown li{color:#C9D1D9}
.stMarkdown code{background:#1C2128;color:#79C0FF;border-radius:4px;padding:1px 5px}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
for k, v in {
    "results": None,
    "running": False,
    "chat": [],            # PaperBot chat
    "katz_chat": [],       # KatzBot chat
    "topic": "",
    "katzbot_ready": False,
    "citations": [],
    "bib_file": "",
    "bib_saved": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 PaperMind")
    st.markdown(
        "<span style='font-size:.75rem;color:#8B949E;font-family:monospace'>"
        "Katz School · Ideathon 2026</span>", unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── LLM Provider selector ─────────────────────────────────
    st.markdown("#### 🤖 AI Provider")
    provider_choice = st.radio(
        "provider",
        ["Groq (Free)", "OpenAI (Paid)"],
        index=0 if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" else 1,
        label_visibility="collapsed",
        horizontal=True,
    )
    is_groq = provider_choice == "Groq (Free)"

    from llm_config import get_provider_display, is_api_key_set
    pinfo = get_provider_display()

    # Show appropriate key status
    groq_key  = os.getenv("GROQ_API_KEY",  "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_ok   = bool(groq_key  and groq_key.startswith("gsk_"))
    openai_ok = bool(openai_key and openai_key.startswith("sk-"))

    if is_groq:
        if groq_ok:
            st.markdown(
                "<span style='color:#3FB950;font-size:.8rem'>● Groq API key loaded ✓</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#F85149;font-size:.8rem'>● Add GROQ_API_KEY to .env</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<span style='color:#8B949E;font-size:.75rem'>"
                "Free key → console.groq.com</span>",
                unsafe_allow_html=True,
            )
        # Show model choices for Groq (production models, April 2026)
        groq_model = st.selectbox(
            "Groq model",
            [
                "llama-3.3-70b-versatile",                  # best quality
                "llama-3.1-8b-instant",                     # fastest
                "openai/gpt-oss-120b",                      # GPT-class 120B
                "openai/gpt-oss-20b",                       # GPT-class fast
                "meta-llama/llama-4-scout-17b-16e-instruct", # Llama 4
                "qwen/qwen3-32b",                            # Qwen reasoning
            ],
            index=0,
            label_visibility="collapsed",
        )
        groq_descriptions = {
            "llama-3.3-70b-versatile":                   "⭐ Best quality · 300K TPM · recommended",
            "llama-3.1-8b-instant":                      "⚡ Fastest · 250K TPM · 14,400 req/day",
            "openai/gpt-oss-120b":                       "🔥 GPT-class 120B · 250K TPM",
            "openai/gpt-oss-20b":                        "🚀 GPT-class · 1000 t/s · fastest",
            "meta-llama/llama-4-scout-17b-16e-instruct": "🦙 Llama 4 Scout · vision support",
            "qwen/qwen3-32b":                            "🧠 Qwen3 32B · strong reasoning",
        }
        st.markdown(
            f"<span style='color:#8B949E;font-size:.75rem'>"
            f"{groq_descriptions.get(groq_model, groq_model)}</span>",
            unsafe_allow_html=True,
        )
    else:
        if openai_ok:
            st.markdown(
                "<span style='color:#3FB950;font-size:.8rem'>● OpenAI API key loaded ✓</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#F85149;font-size:.8rem'>● Add OPENAI_API_KEY to .env</span>",
                unsafe_allow_html=True,
            )
        openai_model = st.selectbox(
            "OpenAI model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            label_visibility="collapsed",
        )

    # Write runtime env vars so llm_config picks them up this session
    os.environ["LLM_PROVIDER"] = "groq" if is_groq else "openai"
    if is_groq:
        os.environ["MODEL_NAME"] = groq_model
    else:
        os.environ["MODEL_NAME"] = openai_model

    st.markdown("---")
    st.markdown("#### Research Topic")
    topic = st.text_input(
        "Topic", value="Agentic AI and Multi-Agent Systems",
        label_visibility="collapsed",
        placeholder="e.g. Transformer attention mechanisms",
    )
    st.markdown("#### Research Question *(optional)*")
    rq = st.text_area(
        "RQ", label_visibility="collapsed",
        placeholder="What specific question are you trying to answer?",
        height=80,
    )
    st.markdown("#### Study Timeline")
    days  = st.slider("Days available", 3, 30, 7)
    hours = st.slider("Hours per day", 1, 8, 3)
    st.markdown("---")
    incl_planner = st.toggle("Study planner", value=True)
    st.markdown("---")
    st.markdown(
        "<span style='color:#58A6FF;font-size:.8rem'>● Semantic Scholar: free</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='color:#D29922;font-size:.8rem'>● KatzBot: crawls yu.edu/katz</span>",
        unsafe_allow_html=True,
    )
    if not openai_ok:
        st.markdown(
            "<span style='color:#8B949E;font-size:.75rem'>"
            "KatzBot embeddings: HuggingFace (free)</span>",
            unsafe_allow_html=True,
        )


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="pm-header">
  <div class="pm-eyebrow">Yeshiva University · Katz School of Science and Health</div>
  <div class="pm-title">Paper<em>Mind</em></div>
  <div class="pm-sub">
    6 research agents + KatzBot RAG + Faculty Matcher — your complete AI research platform.
    Find papers, map relationships, identify gaps, draft your lit review, and connect with 
    the right Katz professor.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────
(tab_run, tab_papers, tab_map, tab_gaps,
 tab_litrev, tab_plan, tab_chat,
 tab_katzbot, tab_faculty, tab_citations) = st.tabs([
    "⚡ Run Agents",
    "📄 Papers Found",
    "🗺️ Relationship Map",
    "🎯 Research Gaps",
    "✍️ Lit Review Draft",
    "📅 Study Plan",
    "💬 PaperBot",
    "🎓 KatzBot",
    "👨‍🏫 Faculty Match",
    "📚 Citations (.bib)",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — RUN AGENTS
# ══════════════════════════════════════════════════════════════
with tab_run:
    col_agents, col_action = st.columns([3, 2], gap="large")

    with col_agents:
        st.markdown(
            "<div style='font-family:monospace;font-size:.75rem;text-transform:uppercase;"
            "letter-spacing:.12em;color:#8B949E;margin-bottom:.8rem'>Agent pipeline</div>",
            unsafe_allow_html=True,
        )
        agents_info = [
            ("01", "Crawler",      "Searches Semantic Scholar · finds 15–25 papers"),
            ("02", "Reader",       "Extracts claims, methods, findings per paper"),
            ("03", "Mapper",       "Maps agreements, contradictions, lineage"),
            ("04", "Gap Finder",   "Identifies unexplored research opportunities"),
            ("05", "Writer",       "Drafts the literature review section"),
        ]
        if incl_planner:
            agents_info.append(("06", "Study Planner", "Builds day-by-day reading schedule"))

        is_done = bool(st.session_state.results)
        is_run  = st.session_state.running

        st.markdown('<div class="agent-pipeline">', unsafe_allow_html=True)
        for num, name, desc in agents_info:
            state   = "done" if is_done else ("active" if is_run else "waiting")
            st.markdown(f"""
            <div class="agent-row {state}">
                <div class="dot {state}"></div>
                <div class="agent-num">{num}</div>
                <div class="agent-name">{name}</div>
                <div class="agent-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if is_done:
            st.markdown('<div class="stats-row">', unsafe_allow_html=True)
            n = len(agents_info)
            st.markdown(
                f'<div class="stat-chip"><b>{n}</b> agents</div>'
                f'<div class="stat-chip"><b>5</b> outputs</div>'
                f'<div class="stat-chip"><b>{days}d</b> plan</div>'
                f'<div class="stat-chip">Semantic Scholar ✓</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col_action:
        st.markdown(
            "<div style='font-family:monospace;font-size:.75rem;text-transform:uppercase;"
            "letter-spacing:.12em;color:#8B949E;margin-bottom:.8rem'>Configuration</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="content-card">
            <div class="card-label">Topic</div>
            <div style="color:#E6EDF3;font-size:.95rem;font-weight:500">{topic}</div>
            {"<div class='card-label' style='margin-top:.8rem'>Research question</div>"
             + "<div style='color:#8B949E;font-size:.82rem;line-height:1.5'>"
             + rq[:120] + "...</div>" if rq else ""}
            <div class="card-label" style="margin-top:.8rem">Timeline</div>
            <div style="color:#E6EDF3;font-size:.88rem">{days} days · {hours}h/day · {days*hours}h total</div>
        </div>
        """, unsafe_allow_html=True)

        active_key_ok = groq_ok if is_groq else openai_ok
        if not active_key_ok:
            provider_label = "GROQ_API_KEY (console.groq.com — free)" if is_groq else "OPENAI_API_KEY"
            st.markdown(
                f"<div style='background:#1C0A0A;border:1px solid #F85149;border-radius:8px;"
                f"padding:.8rem 1rem;font-size:.82rem;color:#F85149;margin:.8rem 0'>"
                f"⚠ Add {provider_label} to .env to run agents</div>",
                unsafe_allow_html=True,
            )

        run_btn = st.button(
            "⚡ Run PaperMind",
            use_container_width=True,
            disabled=st.session_state.running,
        )

        if run_btn:
            st.session_state.running = True
            st.session_state.results = None
            st.session_state.chat    = []
            st.session_state.topic   = topic

            with st.spinner("Agents working… 3–5 minutes"):
                try:
                    from crew import run_papermind
                    results = run_papermind(
                        topic=topic,
                        research_question=rq,
                        days=days,
                        hours_per_day=hours,
                        include_planner=incl_planner,
                    )
                    st.session_state.results = results
                    has_output = any(
                        results.get(k, "").strip()
                        for k in ["papers", "lit_review", "gaps", "map"]
                    )
                    if has_output:
                        st.success("✅ Done! Switch to the tabs above to see your results.")
                    else:
                        st.warning("Agents ran but output was empty. Check your API key.")
                except Exception as e:
                    err = str(e)
                    if st.session_state.results:
                        st.warning(f"Minor issue at finish (results saved): {err[:120]}")
                    else:
                        st.error(f"Error: {err}")
                finally:
                    st.session_state.running = False

        if st.session_state.results:
            r   = st.session_state.results
            full = "\n\n---\n\n".join([
                f"# Papers Found\n{r.get('papers','')}",
                f"# Paper Extractions\n{r.get('extractions','')}",
                f"# Relationship Map\n{r.get('map','')}",
                f"# Research Gaps\n{r.get('gaps','')}",
                f"# Literature Review Draft\n{r.get('lit_review','')}",
                f"# Study Plan\n{r.get('study_plan','')}",
            ])
            st.download_button(
                "⬇ Download all outputs (.md)",
                data=full,
                file_name=f"PaperMind_{topic[:30].replace(' ','_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════
# TAB 2 — PAPERS
# ══════════════════════════════════════════════════════════════
with tab_papers:
    if not st.session_state.results:
        st.markdown("<div style='color:#8B949E;margin-top:2rem'>Run the agents first.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("### Papers found by the Crawler agent")
        st.markdown(st.session_state.results.get("papers", "No output."))
        st.download_button("⬇ Download papers list",
                           data=st.session_state.results.get("papers", ""),
                           file_name="papers_found.md", mime="text/markdown")


# ══════════════════════════════════════════════════════════════
# TAB 3 — RELATIONSHIP MAP
# ══════════════════════════════════════════════════════════════
with tab_map:
    if not st.session_state.results:
        st.markdown("<div style='color:#8B949E;margin-top:2rem'>Run the agents first.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("### How the papers relate to each other")
        st.info("The Mapper agent identified agreement clusters, contradictions, "
                "citation lineage, and the methodological landscape.")
        st.markdown(st.session_state.results.get("map", "No output."))
        st.download_button("⬇ Download relationship map",
                           data=st.session_state.results.get("map", ""),
                           file_name="relationship_map.md", mime="text/markdown")


# ══════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH GAPS
# ══════════════════════════════════════════════════════════════
with tab_gaps:
    if not st.session_state.results:
        st.markdown("<div style='color:#8B949E;margin-top:2rem'>Run the agents first.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("### Research gaps identified by the Gap Finder agent")
        st.markdown(
            "<div style='background:#1C1A10;border:1px solid #D29922;border-radius:8px;"
            "padding:.8rem 1rem;font-size:.85rem;color:#D29922;margin-bottom:1rem'>"
            "🎯 These are genuine opportunities for your original research contribution"
            "</div>", unsafe_allow_html=True,
        )
        st.markdown(st.session_state.results.get("gaps", "No output."))
        st.download_button("⬇ Download gap analysis",
                           data=st.session_state.results.get("gaps", ""),
                           file_name="research_gaps.md", mime="text/markdown")


# ══════════════════════════════════════════════════════════════
# TAB 5 — LIT REVIEW
# ══════════════════════════════════════════════════════════════
with tab_litrev:
    if not st.session_state.results:
        st.markdown("<div style='color:#8B949E;margin-top:2rem'>Run the agents first.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("### Literature review draft — ready to paste into your paper")
        st.markdown(
            "<div style='background:#0D1F12;border:1px solid #3FB950;border-radius:8px;"
            "padding:.8rem 1rem;font-size:.85rem;color:#3FB950;margin-bottom:1rem'>"
            "✍ Starting draft. Review, edit, and add your own voice before submitting."
            "</div>", unsafe_allow_html=True,
        )
        lit = st.session_state.results.get("lit_review", "No output.")
        st.markdown(lit)
        with st.expander("📋 Per-paper extractions (Reader agent)"):
            st.markdown(st.session_state.results.get("extractions", "No output."))
        st.download_button("⬇ Download literature review draft",
                           data=lit, file_name="literature_review_draft.md",
                           mime="text/markdown")


# ══════════════════════════════════════════════════════════════
# TAB 6 — STUDY PLAN
# ══════════════════════════════════════════════════════════════
with tab_plan:
    if not st.session_state.results:
        st.markdown("<div style='color:#8B949E;margin-top:2rem'>Run the agents first.</div>",
                    unsafe_allow_html=True)
    elif not st.session_state.results.get("study_plan"):
        st.info("Study planner was not enabled. Re-run with the toggle on.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Study days", days)
        c2.metric("Hours/day", hours)
        c3.metric("Total hours", days * hours)
        st.markdown("### Your personalized reading & research plan")
        st.markdown(st.session_state.results.get("study_plan", ""))
        st.download_button("⬇ Download study plan",
                           data=st.session_state.results.get("study_plan", ""),
                           file_name="study_plan.md", mime="text/markdown")


# ══════════════════════════════════════════════════════════════
# TAB 7 — PAPERBOT (research context chatbot)
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### PaperBot — AI research discussion partner")

    has_ctx = bool(st.session_state.results)
    if has_ctx:
        st.markdown(
            "<div style='background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;"
            "padding:.6rem 1rem;font-size:.82rem;color:#58A6FF;margin-bottom:1rem'>"
            "● PaperBot has full context from your literature analysis"
            "</div>", unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#161B22;border:1px solid #30363D;border-radius:8px;"
            "padding:.6rem 1rem;font-size:.82rem;color:#8B949E;margin-bottom:1rem'>"
            "ℹ Run the agents first for context-aware answers"
            "</div>", unsafe_allow_html=True,
        )

    if not st.session_state.chat:
        st.markdown("""
        <div class="chat-who">PaperBot</div>
        <div class="bubble-bot">
        I'm PaperBot — your research discussion partner.<br><br>
        Ask me about the papers in your analysis, how to position your research,
        which methodology to choose, or how to make your literature review stronger.
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat:
            who = "You" if msg["role"] == "user" else "PaperBot"
            cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
            st.markdown(
                f'<div class="chat-who">{who}</div><div class="{cls}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    if not st.session_state.chat:
        st.markdown("<div style='font-size:.78rem;color:#8B949E;margin:.8rem 0 .4rem;"
                    "font-family:monospace;text-transform:uppercase;letter-spacing:.1em'>"
                    "Quick prompts</div>", unsafe_allow_html=True)
        from chatbot import QUICK_PROMPTS
        cols = st.columns(2)
        for i, p in enumerate(QUICK_PROMPTS):
            with cols[i % 2]:
                if st.button(p, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat.append({"role": "user", "content": p})
                    st.rerun()

    user_input = st.chat_input("Ask PaperBot about your research…")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.spinner("PaperBot thinking…"):
            try:
                from chatbot import chat
                ctx = ""
                if st.session_state.results:
                    r = st.session_state.results
                    ctx = "\n\n".join([
                        r.get("map", ""), r.get("gaps", ""),
                        r.get("lit_review", "")[:2000],
                    ])
                reply = chat(st.session_state.chat, context=ctx)
                st.session_state.chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"PaperBot error: {e}")
        st.rerun()

    if st.session_state.chat:
        if st.button("Clear PaperBot chat", type="secondary"):
            st.session_state.chat = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 8 — KATZBOT (RAG over yu.edu/katz)
# ══════════════════════════════════════════════════════════════
with tab_katzbot:
    st.markdown("### 🎓 KatzBot — Yeshiva University AI Assistant")
    st.markdown(
        "<div style='background:#161B22;border:1px solid #D29922;border-radius:8px;"
        "padding:.8rem 1rem;font-size:.85rem;color:#D29922;margin-bottom:1rem'>"
        "KatzBot is grounded in the real Katz School website (yu.edu/katz). "
        "Ask about programs, faculty, research, admissions, events — anything on the site."
        "</div>", unsafe_allow_html=True,
    )

    # Index status
    col_kb1, col_kb2 = st.columns([3, 1])
    with col_kb1:
        if not st.session_state.katzbot_ready:
            st.info(
                "KatzBot needs to index the Katz School website before answering. "
                "First run takes ~5 minutes (crawls ~60 pages). "
                "Subsequent runs load instantly from disk."
            )
    with col_kb2:
        build_btn = st.button(
            "🔨 Build / Refresh Index",
            key="build_katz_index",
            use_container_width=True,
        )

    if build_btn:
        with st.spinner("Crawling yu.edu/katz and building vector index… (~5 min first time)"):
            try:
                from katzbot.rag_engine import get_engine
                engine = get_engine()
                stats = engine.build(force_refresh=True)
                st.session_state.katzbot_ready = True
                st.session_state.katzbot_ready = True
                st.success("✅ KatzBot index ready!")
            except Exception as e:
                st.error(f"Index build failed: {e}")
                st.info("Tip: Set GROQ_API_KEY in .env.")

    # Chat display
    if not st.session_state.katz_chat:
        st.markdown("""
        <div class="chat-who">KatzBot</div>
        <div class="bubble-bot">
        Hi! I'm KatzBot, grounded in the Yeshiva University Katz School website.<br><br>
        Ask me about <b>programs</b> (AI, CS, Data Analytics, Cybersecurity),
        <b>faculty research</b>, <b>admissions</b>, <b>tuition</b>, <b>events</b>,
        or anything else on the Katz School site.<br><br>
        Click "Build / Refresh Index" above first if this is your first time.
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.katz_chat:
            who = "You" if msg["role"] == "user" else "KatzBot"
            cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
            st.markdown(
                f'<div class="chat-who">{who}</div><div class="{cls}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Quick prompts for KatzBot
    if not st.session_state.katz_chat:
        katz_prompts = [
            "What programs does Katz School offer?",
            "Who is the chair of Computer Science?",
            "Tell me about the M.S. in AI program",
            "What research is happening in machine learning?",
            "How much is tuition at Katz School?",
            "Who should I contact about cybersecurity research?",
        ]
        st.markdown("<div style='font-size:.78rem;color:#8B949E;margin:.8rem 0 .4rem;"
                    "font-family:monospace;text-transform:uppercase;letter-spacing:.1em'>"
                    "Quick prompts</div>", unsafe_allow_html=True)
        kc = st.columns(2)
        for i, p in enumerate(katz_prompts):
            with kc[i % 2]:
                if st.button(p, key=f"kqp_{i}", use_container_width=True):
                    st.session_state.katz_chat.append({"role": "user", "content": p})
                    st.rerun()

    katz_input = st.chat_input("Ask KatzBot about Yeshiva University / Katz School…")
    if katz_input:
        st.session_state.katz_chat.append({"role": "user", "content": katz_input})
        with st.spinner("KatzBot searching yu.edu/katz…"):
            try:
                from katzbot.rag_engine import get_engine
                engine = get_engine()

                # Pass research context if agents have run
                extra_ctx = ""
                if st.session_state.results:
                    extra_ctx = st.session_state.results.get("gaps", "")[:500]

                result = engine.ask(katz_input, history=st.session_state.katz_chat, extra_context=extra_ctx)
                answer  = result["answer"]
                sources = result["sources"]

                # Show faculty suggestions if relevant
                fac_html = ""
                if result["faculty_matches"]:
                    fac_html = "<br><br><b style='color:#D29922'>Relevant faculty:</b><br>"
                    for f in result["faculty_matches"][:2]:
                        fac_html += (
                            f"• <b>{f['name']}</b> — {f['title']}<br>"
                            f"  <a href='mailto:{f['email']}' style='color:#58A6FF'>"
                            f"{f['email']}</a><br>"
                        )

                sources_html = ""
                if sources:
                    pills = "".join(
                        f"<span class='source-pill'>{s.split('/')[-1] or s}</span>"
                        for s in sources[:4]
                    )
                    sources_html = f"<br><br><span style='font-size:.72rem;color:#8B949E'>Sources:</span><br>{pills}"

                full_reply = answer + fac_html + sources_html
                st.session_state.katz_chat.append({"role": "assistant", "content": full_reply})
                st.session_state.katzbot_ready = True

            except Exception as e:
                st.session_state.katz_chat.append({
                    "role": "assistant",
                    "content": f"⚠ KatzBot needs the index to be built first. "
                               f"Click 'Build / Refresh Index' above. (Error: {str(e)[:80]})"
                })
        st.rerun()

    if st.session_state.katz_chat:
        if st.button("Clear KatzBot chat", type="secondary", key="clear_katz"):
            st.session_state.katz_chat = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 9 — FACULTY MATCH
# ══════════════════════════════════════════════════════════════
with tab_faculty:
    st.markdown("### 👨‍🏫 Katz School Faculty — Find Your Research Supervisor")
    st.markdown(
        "<div style='background:#161B22;border:1px solid #58A6FF;border-radius:8px;"
        "padding:.8rem 1rem;font-size:.85rem;color:#58A6FF;margin-bottom:1.2rem'>"
        "Based on your research topic, here are the Katz School faculty members "
        "most relevant to your work — with direct contact info from yu.edu/katz/faculty"
        "</div>", unsafe_allow_html=True,
    )

    from katzbot.faculty import match_faculty, KATZ_FACULTY

    # Auto-match from topic if agents have run
    search_topic = st.session_state.topic or topic

    if search_topic:
        matched = match_faculty(search_topic)
        if matched:
            st.markdown(
                f"<div style='font-size:.82rem;color:#8B949E;margin-bottom:1rem'>"
                f"Matched to your topic: <b style='color:#E6EDF3'>{search_topic}</b>"
                f"</div>", unsafe_allow_html=True,
            )
            st.markdown("#### 🎯 Top matches for your research")
            for f in matched:
                expertise_str = " · ".join(f["expertise"][:5])
                st.markdown(f"""
                <div class="fac-card">
                    <div class="fac-name">{f['name']}</div>
                    <div class="fac-title">{f['title']}</div>
                    <div class="fac-dept">{f['dept']}</div>
                    <div class="fac-expertise">{expertise_str}</div>
                    <div class="fac-contact">
                        <a href="mailto:{f['email']}">✉ {f['email']}</a> &nbsp;·&nbsp;
                        <a href="{f['profile']}" target="_blank">Faculty profile →</a>
                    </div>
                    <div class="fac-note">{f['note']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔍 Search all faculty")

    search_q = st.text_input(
        "Search by name, topic, or keyword",
        placeholder="e.g. healthcare, LLMs, cybersecurity, deep learning…",
        label_visibility="collapsed",
    )

    dept_filter = st.selectbox(
        "Filter by department",
        ["All departments",
         "M.S. in Artificial Intelligence",
         "Computer Science & Engineering",
         "Data Analytics & Visualization",
         "Cybersecurity",
         "Mathematical Sciences"],
        label_visibility="collapsed",
    )

    # Filter faculty
    display_faculty = KATZ_FACULTY
    if search_q:
        q = search_q.lower()
        display_faculty = [
            f for f in display_faculty
            if q in f["name"].lower()
            or q in f["note"].lower()
            or any(q in kw.lower() for kw in f["expertise"])
        ]
    if dept_filter != "All departments":
        display_faculty = [f for f in display_faculty if f["dept"] == dept_filter]

    if not display_faculty:
        st.markdown("<div style='color:#8B949E;margin-top:1rem'>No faculty match this filter.</div>",
                    unsafe_allow_html=True)
    else:
        for f in display_faculty:
            expertise_str = " · ".join(f["expertise"][:6])
            color_map = {
                "M.S. in Artificial Intelligence": "#58A6FF",
                "Computer Science & Engineering":  "#3FB950",
                "Data Analytics & Visualization":  "#BC8CFF",
                "Cybersecurity":                   "#F85149",
                "Mathematical Sciences":           "#D29922",
            }
            accent = color_map.get(f["dept"], "#58A6FF")
            st.markdown(f"""
            <div class="fac-card" style="border-left-color:{accent}">
                <div class="fac-name">{f['name']}</div>
                <div class="fac-title">{f['title']}</div>
                <div class="fac-dept" style="color:{accent}">{f['dept']}</div>
                <div class="fac-expertise">{expertise_str}</div>
                <div class="fac-contact">
                    <a href="mailto:{f['email']}">✉ {f['email']}</a> &nbsp;·&nbsp;
                    <a href="{f['profile']}" target="_blank">Faculty profile →</a>
                </div>
                <div class="fac-note">{f['note']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:8px;
    padding:1rem 1.2rem;font-size:.85rem;color:#8B949E;line-height:1.8'>
    <b style='color:#E6EDF3'>📧 How to email a professor about your research:</b><br>
    Subject: <i>Student Research Inquiry — [Your Topic] — Ideathon 2026</i><br><br>
    "Dear Prof. [Name],<br>
    I am a student in the Katz School's M.S. in [your program] participating in the 
    Ideathon 2026. I have built [brief description of PaperMind/your project] and would 
    greatly appreciate 10 minutes of your feedback before the competition on [date].<br>
    My project addresses [specific gap you found]. I believe it aligns with your work 
    in [their expertise area].<br>
    Thank you for your time."
    </div>
    """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════
# TAB 10 — CITATIONS (.bib)
# ══════════════════════════════════════════════════════════════
with tab_citations:
    st.markdown("### 📚 BibTeX Citations — Overleaf Ready")

    st.markdown("""
    <div style="background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;
    padding:.8rem 1rem;font-size:.84rem;color:#58A6FF;margin-bottom:1rem;line-height:1.7">
    <b>How it works:</b> Titles extracted from Papers tab →
    matched against <b>Semantic Scholar API</b> (with automatic retry on rate limits)
    → exact BibTeX built from real metadata. Unmatched titles are filled via topic search.
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.results or not st.session_state.results.get("papers","").strip():
        st.markdown("""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
        padding:2rem;text-align:center;color:#8B949E">
            <div style="font-size:1.8rem;margin-bottom:.4rem">📄</div>
            <div style="color:#C9D1D9;font-weight:500;margin-bottom:.3rem">No papers yet</div>
            Run the agents first (⚡ Run Agents tab), then return here.
        </div>
        """, unsafe_allow_html=True)
    else:
        papers_text   = st.session_state.results.get("papers","")
        current_topic = st.session_state.topic or topic

        # ── Prompt banner if not yet fetched ─────────────────
        if not st.session_state.citations:
            st.markdown("""
            <div style="background:#1C1A10;border:1px solid #D29922;border-radius:8px;
            padding:.7rem 1rem;font-size:.84rem;color:#D29922;margin-bottom:.8rem">
            ⚡ Papers are ready. Click <b>Fetch BibTeX</b> to get exact citations.
            This takes 2–4 minutes (Semantic Scholar rate limits to ~1 request/2s).
            </div>
            """, unsafe_allow_html=True)

        # ── Action bar ────────────────────────────────────────
        ac1, ac2, ac3 = st.columns([2, 2, 3])
        with ac1:
            fetch_btn = st.button(
                "🔍 Fetch BibTeX",
                key="fetch_cit_btn",
                use_container_width=True,
                type="primary",
            )
        with ac2:
            if st.session_state.citations:
                st.download_button(
                    "⬇️ Download references.bib",
                    data=st.session_state.bib_file,
                    file_name=f"references_{current_topic[:20].replace(' ','_')}.bib",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_bib_top",
                )
        with ac3:
            if st.session_state.citations:
                n = len(st.session_state.citations)
                st.markdown(
                    f"<div style='color:#3FB950;font-size:.84rem;padding:.4rem 0'>"
                    f"✅ {n} verified citations ready</div>",
                    unsafe_allow_html=True
                )

        # ── Fetch handler ─────────────────────────────────────
        if fetch_btn:
            prog      = st.progress(0, text="Starting…")
            status    = st.empty()

            try:
                from tools.citation_fetcher import (
                    fetch_citations_from_papers_text,
                    extract_titles_from_text,
                    build_bib_file,
                )

                # Show how many titles we found
                titles = extract_titles_from_text(papers_text)
                n_titles = len(titles)

                status.info(
                    f"📖 Found **{n_titles} paper titles** in agent output. "
                    f"Searching Semantic Scholar… (allow 2–4 min due to API rate limits)"
                )
                prog.progress(10, text=f"Searching {n_titles} titles on Semantic Scholar…")

                if n_titles == 0:
                    status.warning(
                        "Could not extract titles from agent output — "
                        "falling back to topic search."
                    )

                citations = fetch_citations_from_papers_text(
                    papers_text=papers_text,
                    topic=current_topic,
                    delay=2.5,
                )

                prog.progress(95, text="Building .bib file…")

                if citations:
                    from tools.citation_fetcher import (
                        save_bib_to_disk, print_citations_summary
                    )
                    bib      = build_bib_file(citations, topic=current_topic)
                    saved_to = save_bib_to_disk(bib, topic=current_topic,
                                                output_dir="outputs")
                    print_citations_summary(citations)

                    st.session_state.citations  = citations
                    st.session_state.bib_file   = bib
                    st.session_state.bib_saved  = saved_to
                    prog.progress(100, text="Done!")
                    status.success(
                        f"✅ **{len(citations)} citations fetched!** "
                        f"Auto-saved → `{saved_to}`"
                    )
                else:
                    prog.progress(100, text="Failed")
                    status.error(
                        "Could not fetch any citations. "
                        "Semantic Scholar may be temporarily unavailable. "
                        "Wait 60 seconds and try again."
                    )

            except Exception as e:
                prog.progress(100, text="Error")
                status.error(f"Error: {e}")

            st.rerun()

        # ══════════════════════════════════════════════════════
        # RESULTS — shown once citations are fetched
        # ══════════════════════════════════════════════════════
        if st.session_state.citations:
            citations = st.session_state.citations

            # ── Metrics ───────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Papers",   len(citations))
            m2.metric("With DOI",       sum(1 for c in citations if c.get("doi")))
            m3.metric("On arXiv",       sum(1 for c in citations if c.get("arxiv")))
            m4.metric("Conf / Journal", 
                f"{sum(1 for c in citations if '@inproceedings' in c.get('bibtex',''))} / "
                f"{sum(1 for c in citations if '@article' in c.get('bibtex',''))}"
            )

            st.markdown("---")

            # ── Full .bib file block ───────────────────────────
            st.markdown("#### 📄 Full `.bib` File")
            st.markdown(
                "<div style='font-size:.82rem;color:#8B949E;margin-bottom:.4rem'>"
                "Copy everything below and paste into Overleaf, "
                "or use the download button.</div>",
                unsafe_allow_html=True
            )

            # The big text area — easy select-all copy
            st.text_area(
                label="references.bib — select all (Ctrl+A) and copy:",
                value=st.session_state.bib_file,
                height=320,
                key="bib_main_textarea",
                label_visibility="visible",
            )

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download references.bib",
                    data=st.session_state.bib_file,
                    file_name=f"references_{current_topic[:20].replace(' ','_')}.bib",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_bib_main2",
                )
            with dl2:
                if st.button("🔄 Re-fetch citations", key="refetch_btn",
                             use_container_width=True):
                    st.session_state.citations = []
                    st.session_state.bib_file  = ""
                    st.rerun()

            st.markdown("---")

            # ── Individual citation cards ─────────────────────
            st.markdown(f"#### Individual Entries ({len(citations)} papers)")

            for i, cit in enumerate(citations, 1):
                title  = cit.get("title","Unknown")
                auths  = cit.get("authors",[])
                year   = cit.get("year","")
                venue  = cit.get("venue","")
                doi    = cit.get("doi","")
                arxiv  = cit.get("arxiv","")
                url    = cit.get("url","")
                key    = cit.get("cite_key",f"paper{i}")
                cites  = cit.get("citation_count",0)
                bibtex = cit.get("bibtex","")
                icon   = "🏛️" if "@inproceedings" in bibtex else "📰"

                auth_str = ", ".join(auths[:3]) + (" et al." if len(auths)>3 else "")

                links = []
                if doi:
                    links.append(f'<a href="https://doi.org/{doi}" target="_blank" '
                        f'style="color:#58A6FF;font-size:.76rem">🔗 DOI</a>')
                if arxiv:
                    links.append(f'<a href="https://arxiv.org/abs/{arxiv}" target="_blank" '
                        f'style="color:#58A6FF;font-size:.76rem">📋 arXiv</a>')
                if url and not doi and not arxiv:
                    links.append(f'<a href="{url}" target="_blank" '
                        f'style="color:#58A6FF;font-size:.76rem">🔗 Paper</a>')

                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #30363D;
                border-left:3px solid #1F6FEB;border-radius:0 8px 8px 0;
                padding:.85rem 1.1rem;margin:.35rem 0">
                    <div style="font-weight:600;color:#E6EDF3;font-size:.9rem;
                    margin-bottom:.2rem">{icon} {i}. {title}</div>
                    <div style="color:#8B949E;font-size:.76rem;margin-bottom:.3rem">
                        {auth_str}
                        {"· <b style='color:#C9D1D9'>" + str(year) + "</b>" if year else ""}
                        {" · " + venue[:55] if venue else ""}
                        {" · " + str(cites) + " citations" if cites else ""}
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
                        <code style="background:#1F3A5F;color:#79C0FF;padding:2px 9px;
                        border-radius:4px;font-size:.76rem">\\cite{{{key}}}</code>
                        {"  ".join(links)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"BibTeX  \\cite{{{key}}}", expanded=False):
                    st.code(bibtex, language="bibtex")
                    st.text_area(
                        "Copy:", value=bibtex, height=160,
                        key=f"bib_ta_{i}", label_visibility="visible"
                    )
                    st.download_button(
                        f"⬇️ {key}.bib",
                        data=bibtex,
                        file_name=f"{key}.bib",
                        mime="text/plain",
                        key=f"dl_single_{i}",
                    )

            # ── Overleaf guide ────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;
            padding:1.1rem 1.3rem">
              <div style="font-weight:600;color:#E6EDF3;margin-bottom:.7rem">
              📎 How to use in Overleaf
              </div>
              <div style="font-size:.84rem;color:#C9D1D9;line-height:2.1">
              <b style="color:#D29922">1.</b> Download <code>references.bib</code>
              (button above)<br>
              <b style="color:#D29922">2.</b> Overleaf → <b>Upload</b> →
              select <code>references.bib</code><br>
              <b style="color:#D29922">3.</b> Add before
              <code>\\end{document}</code>:<br>
              &nbsp;&nbsp;&nbsp;
              <code style="color:#79C0FF">\\bibliographystyle{ieeetr}</code><br>
              &nbsp;&nbsp;&nbsp;
              <code style="color:#79C0FF">\\bibliography{references}</code><br>
              <b style="color:#D29922">4.</b> In your text:
              <code style="color:#79C0FF">\\cite{He2016Deep}</code> or
              <code style="color:#79C0FF">\\cite{key1,key2,key3}</code><br>
              <b style="color:#D29922">5.</b> Compile twice → references appear ✅
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            
# streamlit run app.py