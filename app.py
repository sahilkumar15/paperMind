"""
app.py — PaperMind main Streamlit application

Run: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PaperMind — Research Synthesis AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Brand palette ── */
:root {
    --ink:      #0D1117;
    --ink2:     #161B22;
    --border:   #30363D;
    --muted:    #8B949E;
    --text:     #E6EDF3;
    --accent:   #58A6FF;
    --gold:     #D29922;
    --green:    #3FB950;
    --red:      #F85149;
    --purple:   #BC8CFF;
}

/* ── Dark base ── */
.stApp { background: #0D1117; color: #E6EDF3; }
section[data-testid="stSidebar"] { background: #161B22 !important; border-right: 1px solid #30363D; }
section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

/* ── Header ── */
.pm-header {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #30363D;
    margin-bottom: 2rem;
}
.pm-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #58A6FF;
    margin-bottom: .6rem;
}
.pm-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #E6EDF3;
    line-height: 1.1;
    margin: 0 0 .5rem;
}
.pm-title em { color: #58A6FF; font-style: italic; }
.pm-sub {
    font-size: 1rem;
    color: #8B949E;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Agent pipeline cards ── */
.agent-pipeline { display: flex; flex-direction: column; gap: 6px; margin: 1rem 0; }
.agent-row {
    display: flex; align-items: center; gap: 12px;
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px; padding: 10px 14px;
    transition: border-color .2s;
}
.agent-row.active  { border-color: #D29922; background: #1C1A10; }
.agent-row.done    { border-color: #3FB950; background: #0D1F12; }
.agent-row.waiting { opacity: .55; }
.agent-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; color: #8B949E;
    min-width: 20px;
}
.agent-name { font-size: .88rem; font-weight: 500; color: #E6EDF3; }
.agent-desc { font-size: .78rem; color: #8B949E; margin-left: auto; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #30363D; flex-shrink:0; }
.dot.active { background: #D29922; animation: pulse 1s infinite; }
.dot.done   { background: #3FB950; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Stat chips ── */
.stats-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 1.2rem 0; }
.stat-chip {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 20px; padding: 4px 14px;
    font-size: .82rem; color: #8B949E;
    font-family: 'JetBrains Mono', monospace;
}
.stat-chip b { color: #E6EDF3; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22; border-bottom: 1px solid #30363D;
    gap: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: .85rem; font-weight: 500;
    color: #8B949E; padding: .7rem 1.4rem;
    border-bottom: 2px solid transparent;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
    background: transparent !important;
}

/* ── Content cards ── */
.content-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: .8rem 0;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem; letter-spacing: .12em;
    text-transform: uppercase; color: #8B949E;
    margin-bottom: .5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #1F6FEB !important;
    color: white !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: .88rem !important;
    padding: .55rem 1.8rem !important;
    transition: background .2s !important;
}
.stButton > button:hover { background: #388BFD !important; }
.stButton > button:disabled { background: #21262D !important; color: #484F58 !important; }

/* ── Chat ── */
.chat-wrap { max-height: 480px; overflow-y: auto; padding: .5rem 0; }
.bubble-user {
    background: #1F6FEB; color: white;
    border-radius: 14px 14px 3px 14px;
    padding: .65rem 1rem; margin: .4rem 0 .4rem 3.5rem;
    font-size: .9rem; line-height: 1.55;
}
.bubble-bot {
    background: #1C2128; border: 1px solid #30363D;
    color: #E6EDF3; border-radius: 14px 14px 14px 3px;
    padding: .65rem 1rem; margin: .4rem 3.5rem .4rem 0;
    font-size: .9rem; line-height: 1.55;
}
.chat-who {
    font-size: .7rem; font-family: 'JetBrains Mono', monospace;
    letter-spacing: .08em; text-transform: uppercase;
    color: #8B949E; margin: .8rem 0 .2rem;
}

/* ── Input overrides ── */
.stTextInput input, .stTextArea textarea {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 6px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #58A6FF !important;
    box-shadow: none !important;
}
label { color: #8B949E !important; font-size: .82rem !important; }

/* ── Gap badges ── */
.gap-card {
    background: #1C1A10; border: 1px solid #D29922;
    border-left: 3px solid #D29922;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin-bottom: .8rem;
}
.gap-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; color: #D29922;
    text-transform: uppercase; letter-spacing: .1em;
    margin-bottom: .3rem;
}

/* ── Markdown inside st content ── */
.stMarkdown h1,h2,h3 { color: #E6EDF3; font-family: 'DM Serif Display', serif; }
.stMarkdown p { color: #C9D1D9; line-height: 1.7; }
.stMarkdown li { color: #C9D1D9; }
.stMarkdown code { background: #1C2128; color: #79C0FF; border-radius: 4px; padding: 1px 5px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
for k, v in {
    "results": None,
    "running": False,
    "chat": [],
    "topic": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 PaperMind")
    st.markdown(
        "<span style='font-size:.75rem;color:#8B949E;font-family:monospace'>"
        "Katz School · Ideathon 2026</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("#### Research Topic")
    topic = st.text_input(
        "Topic",
        value="Agentic AI and Multi-Agent Systems",
        label_visibility="collapsed",
        placeholder="e.g. Transformer attention mechanisms",
    )

    st.markdown("#### Research Question *(optional)*")
    rq = st.text_area(
        "RQ",
        label_visibility="collapsed",
        placeholder="What specific question are you trying to answer?",
        height=80,
    )

    st.markdown("#### Study Timeline")
    days  = st.slider("Days available", 3, 30, 7)
    hours = st.slider("Hours per day", 1, 8, 3)

    st.markdown("---")
    st.markdown("#### Include study planner?")
    incl_planner = st.toggle("Study planner", value=True)

    st.markdown("---")
    api_ok = bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-"))
    if api_ok:
        st.markdown(
            "<span style='color:#3FB950;font-size:.8rem'>● API key loaded</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='color:#F85149;font-size:.8rem'>● Add OPENAI_API_KEY to .env</span>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<span style='color:#58A6FF;font-size:.8rem'>● Semantic Scholar: free, no key needed</span>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="pm-header">
  <div class="pm-eyebrow">Yeshiva University · Katz School of Science and Health</div>
  <div class="pm-title">Paper<em>Mind</em></div>
  <div class="pm-sub">
    Agentic AI that reads 20+ papers, maps their relationships,
    finds research gaps, and writes your literature review — autonomously.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────
tab_run, tab_papers, tab_map, tab_gaps, tab_litrev, tab_plan, tab_chat = st.tabs([
    "⚡ Run Agents",
    "📄 Papers Found",
    "🗺️ Relationship Map",
    "🎯 Research Gaps",
    "✍️ Lit Review Draft",
    "📅 Study Plan",
    "💬 PaperBot",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — RUN AGENTS
# ══════════════════════════════════════════════════════════════
with tab_run:
    col_agents, col_action = st.columns([3, 2], gap="large")

    with col_agents:
        st.markdown(
            "<div style='font-family:monospace;font-size:.75rem;"
            "text-transform:uppercase;letter-spacing:.12em;color:#8B949E;"
            "margin-bottom:.8rem'>Agent pipeline</div>",
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
            state = "done" if is_done else ("active" if is_run else "waiting")
            dot_cls = state
            st.markdown(f"""
            <div class="agent-row {state}">
                <div class="dot {dot_cls}"></div>
                <div class="agent-num">{num}</div>
                <div class="agent-name">{name}</div>
                <div class="agent-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if is_done:
            st.markdown('<div class="stats-row">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="stat-chip"><b>6</b> agents</div>'
                f'<div class="stat-chip"><b>5</b> outputs</div>'
                f'<div class="stat-chip"><b>{days}d</b> plan</div>'
                f'<div class="stat-chip">Semantic Scholar ✓</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col_action:
        st.markdown(
            "<div style='font-family:monospace;font-size:.75rem;"
            "text-transform:uppercase;letter-spacing:.12em;color:#8B949E;"
            "margin-bottom:.8rem'>Configuration</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        <div class="content-card">
            <div class="card-label">Topic</div>
            <div style="color:#E6EDF3;font-size:.95rem;font-weight:500">{topic}</div>
            {"<div class='card-label' style='margin-top:.8rem'>Research question</div><div style='color:#8B949E;font-size:.82rem;line-height:1.5'>" + rq[:120] + "...</div>" if rq else ""}
            <div class="card-label" style="margin-top:.8rem">Timeline</div>
            <div style="color:#E6EDF3;font-size:.88rem">{days} days · {hours}h/day · {days*hours}h total</div>
        </div>
        """, unsafe_allow_html=True)

        if not api_ok:
            st.markdown(
                "<div style='background:#1C0A0A;border:1px solid #F85149;border-radius:8px;"
                "padding:.8rem 1rem;font-size:.82rem;color:#F85149;margin:.8rem 0'>"
                "⚠ Add OPENAI_API_KEY to .env to run agents</div>",
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
                    st.success("✅ Done! Check the tabs above.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.running = False

        if st.session_state.results:
            r = st.session_state.results
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
        st.markdown(
            "<div style='color:#8B949E;margin-top:2rem;font-size:.9rem'>"
            "Run the agents first to see papers found.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Papers found by the Crawler agent")
        st.markdown(st.session_state.results.get("papers", "No output."))
        st.download_button(
            "⬇ Download papers list",
            data=st.session_state.results.get("papers", ""),
            file_name="papers_found.md", mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — RELATIONSHIP MAP
# ══════════════════════════════════════════════════════════════
with tab_map:
    if not st.session_state.results:
        st.markdown(
            "<div style='color:#8B949E;margin-top:2rem;font-size:.9rem'>"
            "Run the agents first.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### How the papers relate to each other")
        st.info(
            "The Mapper agent analyzed all papers and identified agreement clusters, "
            "contradictions, citation lineage, and the methodological landscape."
        )
        st.markdown(st.session_state.results.get("map", "No output."))
        st.download_button(
            "⬇ Download relationship map",
            data=st.session_state.results.get("map", ""),
            file_name="relationship_map.md", mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH GAPS
# ══════════════════════════════════════════════════════════════
with tab_gaps:
    if not st.session_state.results:
        st.markdown(
            "<div style='color:#8B949E;margin-top:2rem;font-size:.9rem'>"
            "Run the agents first.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Research gaps identified by the Gap Finder agent")
        st.markdown(
            "<div style='background:#1C1A10;border:1px solid #D29922;border-radius:8px;"
            "padding:.8rem 1rem;font-size:.85rem;color:#D29922;margin-bottom:1rem'>"
            "🎯 These are genuine opportunities for your original research contribution"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.results.get("gaps", "No output."))
        st.download_button(
            "⬇ Download gap analysis",
            data=st.session_state.results.get("gaps", ""),
            file_name="research_gaps.md", mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════
# TAB 5 — LIT REVIEW
# ══════════════════════════════════════════════════════════════
with tab_litrev:
    if not st.session_state.results:
        st.markdown(
            "<div style='color:#8B949E;margin-top:2rem;font-size:.9rem'>"
            "Run the agents first.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Literature review draft — ready to paste into your paper")
        st.markdown(
            "<div style='background:#0D1F12;border:1px solid #3FB950;border-radius:8px;"
            "padding:.8rem 1rem;font-size:.85rem;color:#3FB950;margin-bottom:1rem'>"
            "✍ This is a starting draft. Review, edit, and add your own voice before submitting."
            "</div>",
            unsafe_allow_html=True,
        )
        lit = st.session_state.results.get("lit_review", "No output.")
        st.markdown(lit)

        also_extractions = st.expander("📋 See per-paper extractions (Reader agent output)")
        with also_extractions:
            st.markdown(st.session_state.results.get("extractions", "No output."))

        st.download_button(
            "⬇ Download literature review draft",
            data=lit,
            file_name="literature_review_draft.md", mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════
# TAB 6 — STUDY PLAN
# ══════════════════════════════════════════════════════════════
with tab_plan:
    if not st.session_state.results:
        st.markdown(
            "<div style='color:#8B949E;margin-top:2rem;font-size:.9rem'>"
            "Run the agents first.</div>",
            unsafe_allow_html=True,
        )
    elif not st.session_state.results.get("study_plan"):
        st.info("Study planner was not enabled. Re-run with the toggle on.")
    else:
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Study days", days)
        with col_p2:
            st.metric("Hours/day", hours)
        with col_p3:
            st.metric("Total hours", days * hours)
        st.markdown("### Your personalized reading & research plan")
        st.markdown(st.session_state.results.get("study_plan", ""))
        st.download_button(
            "⬇ Download study plan",
            data=st.session_state.results.get("study_plan", ""),
            file_name="study_plan.md", mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════
# TAB 7 — PAPERBOT CHAT
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### PaperBot — your AI research discussion partner")

    has_context = bool(st.session_state.results)

    if has_context:
        st.markdown(
            "<div style='background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;"
            "padding:.6rem 1rem;font-size:.82rem;color:#58A6FF;margin-bottom:1rem'>"
            "● PaperBot has full context from your literature analysis"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#161B22;border:1px solid #30363D;border-radius:8px;"
            "padding:.6rem 1rem;font-size:.82rem;color:#8B949E;margin-bottom:1rem'>"
            "ℹ Run the agents first for context-aware answers — or ask general research questions now"
            "</div>",
            unsafe_allow_html=True,
        )

    # Chat display
    if not st.session_state.chat:
        st.markdown("""
        <div class="chat-who">PaperBot</div>
        <div class="bubble-bot">
        I'm PaperBot — your research discussion partner.<br><br>
        Ask me anything about the papers in your analysis, how to position your research,
        what methodology to choose, or how to make your literature review stronger.<br><br>
        Run the agents first to unlock context-aware answers, or ask me a general research question now.
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat:
            who = "You" if msg["role"] == "user" else "PaperBot"
            cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
            st.markdown(
                f'<div class="chat-who">{who}</div>'
                f'<div class="{cls}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Quick prompts
    if not st.session_state.chat:
        st.markdown(
            "<div style='font-size:.78rem;color:#8B949E;margin:.8rem 0 .4rem;"
            "font-family:monospace;text-transform:uppercase;letter-spacing:.1em'>"
            "Quick prompts</div>",
            unsafe_allow_html=True,
        )
        from chatbot import QUICK_PROMPTS
        cols = st.columns(2)
        for i, p in enumerate(QUICK_PROMPTS):
            with cols[i % 2]:
                if st.button(p, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat.append({"role": "user", "content": p})
                    st.rerun()

    # Input
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
                        r.get("map", ""),
                        r.get("gaps", ""),
                        r.get("lit_review", "")[:2000],
                    ])
                reply = chat(st.session_state.chat, context=ctx)
                st.session_state.chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"PaperBot error: {e}")
        st.rerun()

    if st.session_state.chat:
        if st.button("Clear chat", type="secondary"):
            st.session_state.chat = []
            st.rerun()
