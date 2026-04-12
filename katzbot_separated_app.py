"""
app.py — KatzScholarMind: Research Intelligence Platform
======================================================
Yeshiva University · Katz School of Science and Health
Katz School CS & AI Club Ideathon 2026

Tabs:
  1.  ⚡ Run Agents       — 6-agent research pipeline
  2.  📄 Papers Found     — Crawler output
  3.  🗺️ Relationship Map  — Agreements & contradictions
  4.  🎯 Research Gaps    — Unexplored opportunities
  5.  ✍️ Lit Review Draft  — Auto-generated academic text
  6.  📅 Study Plan       — Day-by-day reading schedule
  7.  💬 PaperBot         — Direct AI chat
  8.  🎓 KatzBot          — Katz School RAG assistant
  9.  📅 Katz Events      — Live campus events
  10. 🧠 Smart Advisor    — Faculty + event recommendations
  11. 📚 Citations (.bib) — Verified BibTeX for Overleaf
  12. 👨‍🏫 Faculty Match    — Find the right professor
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="KatzScholarMind",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────
DEFAULTS = {
    "results":      None,
    "topic":        "",
    "katz_chat":    [],
    "katzbot_ready":False,
    "citations":    [],
    "bib_file":     "",
    "bib_saved":    "",
    "events":       [],
    "smart_advice": None,
    "paperbot_chat":[],
    "review_report": None,
    "review_title": "",
    "review_filename": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helper functions (must be defined before use) ────────────
def _build_md_output(results: dict, topic: str) -> str:
    """Build a markdown export of all agent outputs."""
    md = f"# KatzScholarMind Output\n## Topic: {topic}\n\n"
    for key, label in [
        ("papers",     "Papers Found"),
        ("map",        "Relationship Map"),
        ("gaps",       "Research Gaps"),
        ("lit_review", "Literature Review"),
        ("study_plan", "Study Plan"),
    ]:
        val = results.get(key, "")
        if val:
            md += f"## {label}\n\n{val}\n\n---\n\n"
    return md

def _render_chat_history(messages: list, chat_type: str = "generic") -> None:
    """Scrollable chat history renderer used by PaperBot and KatzBot."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if chat_type == "katz":
                if msg.get("sources"):
                    links = []
                    for s in msg["sources"][:3]:
                        if not s:
                            continue
                        label = s.split("/")[-1] or s
                        links.append(f"<a href='{s}' target='_blank' style='color:#58A6FF'>{label}</a>")
                    if links:
                        st.markdown(
                            "<div style='margin-top:.35rem;font-size:.78rem;color:#8B949E'>Sources: "
                            + " · ".join(links)
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                if msg.get("faculty"):
                    for f in msg["faculty"][:2]:
                        st.markdown(
                            f"<div style='margin-top:.3rem;font-size:.78rem;color:#8B949E'>👨‍🏫 <b style='color:#C9D1D9'>{f['name']}</b> — <a href='mailto:{f['email']}' style='color:#58A6FF'>{f['email']}</a></div>",
                            unsafe_allow_html=True,
                        )

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 KatzScholarMind")
    st.caption("Katz School · Ideathon 2026")
    st.markdown("---")

    # AI Provider
    st.markdown("#### ⚙️ AI Provider")
    provider_choice = st.radio(
        "Provider", ["Groq (Free)", "OpenAI (Paid)"],
        horizontal=True, label_visibility="collapsed",
        key="provider_radio",
    )
    provider = "groq" if "Groq" in provider_choice else "openai"
    os.environ["LLM_PROVIDER"] = provider

    from llm_config import (GROQ_MODELS, OPENAI_MODELS,
                             is_api_key_set, get_api_key)

    if provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY","")
        if groq_key and groq_key.startswith("gsk_"):
            st.markdown("● Groq API key loaded ✓")
        else:
            st.warning("Set GROQ_API_KEY in .env")
            groq_key = st.text_input("Groq API key",
                                     type="password", key="groq_key_input")
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key

        model_options = list(GROQ_MODELS.keys())
        groq_model    = st.selectbox(
            "Model", model_options, key="groq_model_select",
            index=0,
        )
        os.environ["MODEL_NAME"] = groq_model
        st.caption(GROQ_MODELS.get(groq_model,""))

        LOW_TPM = {"llama-3.3-70b-versatile","qwen/qwen3-32b"}
        if groq_model in LOW_TPM:
            tpm = 6000 if "qwen" in groq_model else 12000
            st.warning(f"⚠️ {tpm:,} TPM — may rate-limit with 6 agents. "
                       f"Use llama-3.1-8b-instant for reliability.")
    else:
        oai_key = os.getenv("OPENAI_API_KEY","")
        if oai_key and oai_key.startswith("sk-"):
            st.markdown("● OpenAI API key loaded ✓")
        else:
            st.warning("Set OPENAI_API_KEY in .env")
            oai_key = st.text_input("OpenAI API key",
                                    type="password", key="oai_key_input")
            if oai_key:
                os.environ["OPENAI_API_KEY"] = oai_key

        oai_model = st.selectbox(
            "Model", list(OPENAI_MODELS.keys()), key="oai_model_select"
        )
        os.environ["MODEL_NAME"] = oai_model
        st.caption(OPENAI_MODELS.get(oai_model,""))

    st.markdown("---")

    # Research inputs
    st.markdown("#### 🔬 Research Topic")
    topic = st.text_input(
        "Topic", placeholder="e.g. deepfake detection",
        key="topic_input", label_visibility="collapsed",
    )

    st.markdown("#### ❓ Research Question *(optional)*")
    research_question = st.text_area(
        "Question", placeholder="What specific question are you trying to answer?",
        key="rq_input", label_visibility="collapsed", height=80,
    )

    st.markdown("#### 📅 Study Timeline")
    days         = st.slider("Days available", 1, 30, 7, key="days_slider")
    hours_per_day= st.slider("Hours per day",  1, 12, 3, key="hours_slider")

    st.markdown("---")
    include_planner = st.toggle("Study planner", value=True, key="planner_toggle")

    st.markdown("---")
    st.caption("● Semantic Scholar: free")
    st.caption("● KatzBot: crawls yu.edu/katz")

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:.5rem">
<span style="font-size:.78rem;letter-spacing:.12em;color:#58A6FF;
text-transform:uppercase">Yeshiva University · Katz School of Science and Health</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="font-size:2.4rem;font-weight:700;margin:0">
KatzScholar<span style="color:#58A6FF;font-style:italic">Mind</span>
</h1>
<p style="color:#8B949E;margin:.3rem 0 1.2rem;font-size:.95rem">
6 research agents + KatzBot RAG + Faculty Matcher + Live Events —
your complete AI research platform.<br>
Find papers, map relationships, identify gaps, draft your lit review,
and connect with the right Katz professor.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
(tab_run, tab_papers, tab_map, tab_gaps, tab_litrev,
 tab_plan, tab_chat, tab_katzbot, tab_events,
 tab_advisor, tab_citations, tab_faculty, tab_review) = st.tabs([
    "⚡ Run Agents",
    "📄 Papers Found",
    "🗺️ Relationship Map",
    "🎯 Research Gaps",
    "✍️ Lit Review Draft",
    "📅 Study Plan",
    "💬 PaperBot",
    "🎓 KatzBot",
    "📅 Katz Events",
    "🧠 Smart Advisor",
    "📚 Citations (.bib)",
    "👨‍🏫 Faculty Match",
    "🧑‍⚖️ Reviewer Agent",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — RUN AGENTS
# ══════════════════════════════════════════════════════════════
with tab_run:
    col_pipeline, col_config = st.columns([1, 1])

    with col_pipeline:
        st.markdown("#### AGENT PIPELINE")
        agents_info = [
            ("01", "Crawler",      "Searches Semantic Scholar · finds 10-15 papers"),
            ("02", "Reader",       "Extracts claims, methods, findings per paper"),
            ("03", "Mapper",       "Maps agreements, contradictions, lineage"),
            ("04", "Gap Finder",   "Identifies unexplored research opportunities"),
            ("05", "Writer",       "Drafts the literature review section"),
            ("06", "Study Planner","Builds day-by-day reading schedule"),
        ]
        has_results = st.session_state.results is not None
        for num, name, desc in agents_info:
            color = "#3FB950" if has_results else "#484F58"
            st.markdown(
                f"<div style='background:#161B22;border:0.5px solid #30363D;"
                f"border-radius:8px;padding:.6rem 1rem;margin:.25rem 0;"
                f"display:flex;align-items:center;gap:.8rem'>"
                f"<span style='color:{color};font-size:.85rem'>●</span>"
                f"<span style='color:#8B949E;font-size:.8rem'>{num}</span>"
                f"<span style='font-weight:500;color:#E6EDF3;font-size:.9rem'>{name}</span>"
                f"<span style='color:#8B949E;font-size:.78rem;margin-left:auto'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_config:
        st.markdown("#### CONFIGURATION")
        st.markdown(
            f"<div style='background:#161B22;border:0.5px solid #30363D;"
            f"border-radius:8px;padding:1rem;margin-bottom:.8rem'>"
            f"<div style='font-size:.72rem;letter-spacing:.1em;color:#8B949E;"
            f"text-transform:uppercase;margin-bottom:.5rem'>TOPIC</div>"
            f"<div style='font-size:1rem;font-weight:500;color:#E6EDF3'>"
            f"{topic or 'Enter topic in sidebar'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        run_btn = st.button(
            "⚡ Run KatzScholarMind",
            use_container_width=True,
            type="primary",
            key="run_btn",
        )

        if st.session_state.results:
            st.download_button(
                "⬇ Download all outputs (.md)",
                data=_build_md_output(st.session_state.results, topic),
                file_name=f"KatzScholarMind_{topic[:20].replace(' ','_')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_all",
            )
            st.markdown(
                "<div style='background:#0D1F12;border:1px solid #3FB950;"
                "border-radius:6px;padding:.5rem .8rem;font-size:.82rem;"
                "color:#3FB950;margin-top:.5rem'>✅ Done! Switch to tabs above.</div>",
                unsafe_allow_html=True,
            )

    if run_btn:
        if not topic.strip():
            st.error("Enter a research topic in the sidebar first.")
        else:
            st.session_state.topic        = topic
            st.session_state.results      = None
            st.session_state.citations    = []
            st.session_state.bib_file     = ""
            st.session_state.smart_advice = None

            with st.spinner(f"Running 6 agents on '{topic}'… (2-5 min)"):
                try:
                    from crew import run_papermind
                    results = run_papermind(
                        topic=topic,
                        research_question=research_question,
                        days=days,
                        hours_per_day=hours_per_day,
                        include_planner=include_planner,
                    )
                    st.session_state.results = results
                    st.success("✅ Done! Switch to the tabs above.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent error: {e}")




# ══════════════════════════════════════════════════════════════
# TAB 2 — PAPERS FOUND
# ══════════════════════════════════════════════════════════════
with tab_papers:
    st.markdown("#### Papers found by the Crawler agent")
    if not st.session_state.results:
        st.info("Run the agents first.")
    else:
        txt = st.session_state.results.get("papers","")
        if txt:
            st.markdown(txt)
        else:
            st.warning("No papers output found.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — RELATIONSHIP MAP
# ══════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("#### Relationship Map")
    if not st.session_state.results:
        st.info("Run the agents first.")
    else:
        txt = st.session_state.results.get("map","")
        if txt:
            st.markdown(txt)
        else:
            st.warning("No relationship map found.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH GAPS
# ══════════════════════════════════════════════════════════════
with tab_gaps:
    st.markdown("#### Research Gaps")
    if not st.session_state.results:
        st.info("Run the agents first.")
    else:
        txt = st.session_state.results.get("gaps","")
        if txt:
            st.markdown(txt)
        else:
            st.warning("No gaps output found.")


# ══════════════════════════════════════════════════════════════
# TAB 5 — LIT REVIEW DRAFT
# ══════════════════════════════════════════════════════════════
with tab_litrev:
    st.markdown("#### Literature Review Draft")
    if not st.session_state.results:
        st.info("Run the agents first.")
    else:
        txt = st.session_state.results.get("lit_review","")
        if txt:
            st.markdown(txt)
            st.download_button(
                "⬇ Download lit review (.md)",
                data=txt, mime="text/markdown",
                file_name=f"litreview_{topic[:20].replace(' ','_')}.md",
                key="dl_litrev",
            )
        else:
            st.warning("No literature review found.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — STUDY PLAN
# ══════════════════════════════════════════════════════════════
with tab_plan:
    st.markdown("#### Study Plan")
    if not st.session_state.results:
        st.info("Run the agents first.")
    else:
        txt = st.session_state.results.get("study_plan","")
        if txt:
            st.markdown(txt)
        else:
            st.info("Enable the Study Planner toggle and re-run.")


# ══════════════════════════════════════════════════════════════
# TAB 7 — PAPERBOT
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("#### 💬 PaperBot — AI Research Assistant")
    st.caption("Grounded in the current paper set. Ask for the latest paper, novelty directions, or how to position your idea.")

    chat_box = st.container(height=480, border=True)
    with chat_box:
        _render_chat_history(st.session_state.paperbot_chat, chat_type="paper")

    prompt = st.chat_input("Ask PaperBot anything about research…", key="paperbot_input")
    if prompt:
        st.session_state.paperbot_chat.append({"role": "user", "content": prompt})
        try:
            from chatbot import answer_paperbot
            answer = answer_paperbot(
                prompt=prompt,
                chat_history=st.session_state.paperbot_chat[:-1],
                current_topic=st.session_state.topic or topic,
                results=st.session_state.results or {},
            )
        except Exception as e:
            answer = f"PaperBot error: {e}"
        st.session_state.paperbot_chat.append({"role": "assistant", "content": answer})
        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 8 — KATZBOT
# ══════════════════════════════════════════════════════════════
with tab_katzbot:
    st.markdown("### 🎓 KatzBot — Yeshiva University AI Assistant")

    st.markdown("""
    <div style="background:#1C1A10;border:1px solid #D29922;border-radius:8px;
    padding:.7rem 1rem;font-size:.84rem;color:#D29922;margin-bottom:.8rem">
    KatzBot is grounded only in the real Katz School website (yu.edu/katz) and KatzBot's indexed university data.<br>
    It is intentionally separate from PaperBot and does not use research-agent outputs, literature reviews, or paper-analysis context.
    </div>""", unsafe_allow_html=True)

    kb1, kb2 = st.columns([3, 1])
    with kb2:
        build_btn = st.button("🔨 Build / Refresh Index",
                              key="build_index_btn",
                              use_container_width=True)
    with kb1:
        if not st.session_state.katzbot_ready:
            st.markdown(
                "<div style='font-size:.82rem;color:#8B949E;padding:.35rem 0'>"
                "First run: click <b>Build / Refresh Index</b> (~5 min, "
                "saved to disk for instant reload next time)."
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='color:#3FB950;font-size:.82rem;padding:.35rem 0'>"
                "✅ KatzBot index loaded — ask anything!</div>",
                unsafe_allow_html=True
            )

    if build_btn:
        with st.spinner("Crawling yu.edu/katz and building FAISS index… (~5 min first time, instant after)"):
            try:
                from katzbot.rag_engine import get_engine
                engine = get_engine()
                stats  = engine.build(force_refresh=True)
                st.session_state.katzbot_ready = True
                st.success(
                    f"✅ Index ready! "
                    f"{stats.get('web_pages',0)} web pages + "
                    f"{stats.get('faculty_docs',0)} faculty docs → "
                    f"{stats.get('index_vectors',0):,} vectors. "
                    f"Saved to disk — next start loads instantly."
                )
            except Exception as e:
                st.error(f"Index build failed: {e}")
                st.info("Tip: Set GROQ_API_KEY in .env. "
                        "Run: python katzbot/build_index.py")

    # Chat interface
    st.markdown("---")
    if not st.session_state.katz_chat:
        st.markdown("""
        <div style="background:#161B22;border:0.5px solid #30363D;
        border-radius:8px;padding:1rem;margin-bottom:.8rem">
        <div style="font-weight:500;color:#E6EDF3;margin-bottom:.4rem">
        Hi! I'm KatzBot, grounded in the Yeshiva University Katz School website.</div>
        <div style="color:#8B949E;font-size:.84rem">
        Ask me about <b style="color:#C9D1D9">programs</b> (AI, CS, Data Analytics,
        Cybersecurity), <b style="color:#C9D1D9">faculty research</b>,
        <b style="color:#C9D1D9">admissions</b>, <b style="color:#C9D1D9">tuition</b>,
        <b style="color:#C9D1D9">events</b>, or anything else on the Katz School site.
        </div>
        </div>""", unsafe_allow_html=True)

    # Quick prompts
    st.markdown("**Quick prompts:**")
    qp_cols = st.columns(3)
    quick_prompts = [
        "What programs does Katz School offer?",
        "Who is the chair of Computer Science?",
        "Tell me about the M.S. in AI program",
        "What is tuition at Katz School?",
        "Who should I contact about cybersecurity research?",
        "What research is happening in machine learning?",
    ]
    for i, qp in enumerate(quick_prompts):
        with qp_cols[i % 3]:
            if st.button(qp, key=f"qp_{i}", use_container_width=True):
                st.session_state.katz_chat.append({"role":"user","content":qp})

    # Chat display
    chat_box = st.container(height=480, border=True)
    with chat_box:
        _render_chat_history(st.session_state.katz_chat, chat_type="katz")

    # Input
    katz_input = st.chat_input(
        "Ask KatzBot about Yeshiva University / Katz School…",
        key="katzbot_input",
    )
    if katz_input:
        st.session_state.katz_chat.append({"role":"user","content":katz_input})
        try:
            from katzbot.rag_engine import get_engine
            engine = get_engine()
            # Keep KatzBot fully separate from PaperBot/research-agent state.
            # Do NOT pass research gaps or PaperBot context into KatzBot.
            result = engine.ask(
                katz_input,
                history=st.session_state.katz_chat,
            )
            st.session_state.katz_chat.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "faculty": result["faculty_matches"],
            })
        except Exception as e:
            st.session_state.katz_chat.append({"role": "assistant", "content": f"KatzBot error: {e}"})
        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 9 — KATZ EVENTS
# ══════════════════════════════════════════════════════════════
with tab_events:
    st.markdown("### 📅 Katz School Events & News")

    st.markdown("""
    <div style="background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;
    padding:.8rem 1rem;font-size:.84rem;color:#58A6FF;margin-bottom:1rem">
    Live events fetched from <b>yu.edu/katz/events</b>, <b>yu.edu/events</b>,
    and the YU graduate calendar. Cached 6 hours. Refreshes automatically.
    </div>""", unsafe_allow_html=True)

    ev1, ev2 = st.columns([1,1])
    with ev1:
        if st.button("🔄 Refresh Events", key="refresh_ev_btn",
                     use_container_width=True, type="primary"):
            with st.spinner("Fetching from yu.edu/katz…"):
                try:
                    from katzbot.events_fetcher import fetch_events
                    st.session_state.events = fetch_events(force_refresh=True)
                    st.success(f"✅ {len(st.session_state.events)} events loaded!")
                except Exception as e:
                    st.error(f"Events error: {e}")
                    from katzbot.events_fetcher import STATIC_EVENTS
                    st.session_state.events = STATIC_EVENTS
            st.rerun()
    with ev2:
        st.markdown(
            '<a href="https://www.yu.edu/katz/events" target="_blank">'
            '<button style="width:100%;padding:.45rem;background:transparent;'
            'border:1px solid #30363D;border-radius:6px;color:#58A6FF;'
            'cursor:pointer;font-size:.85rem">🌐 Open yu.edu/katz/events</button>'
            '</a>', unsafe_allow_html=True
        )

    # Auto-load
    if not st.session_state.events:
        with st.spinner("Loading Katz events…"):
            try:
                from katzbot.events_fetcher import fetch_events
                st.session_state.events = fetch_events()
            except Exception:
                from katzbot.events_fetcher import STATIC_EVENTS
                st.session_state.events = STATIC_EVENTS

    events = st.session_state.events
    if events:
        # Show relevant events if research has run
        cur_topic = st.session_state.topic or topic
        if cur_topic and st.session_state.results:
            try:
                from katzbot.events_fetcher import match_events_to_topic
                rel = match_events_to_topic(events, cur_topic, top_k=3)
                if rel:
                    st.markdown("#### 🎯 Relevant to your research")
                    for ev in rel:
                        st.markdown(
                            f"<div style='background:#0D1F12;border:1px solid #3FB950;"
                            f"border-radius:8px;padding:.9rem 1.1rem;margin:.4rem 0'>"
                            f"<b style='color:#3FB950'>⭐ {ev.get('title','')}</b><br>"
                            f"<span style='color:#8B949E;font-size:.77rem'>"
                            f"📅 {ev.get('date','TBD')} · 📍 {ev.get('location','')}</span><br>"
                            f"<span style='color:#C9D1D9;font-size:.82rem'>"
                            f"{ev.get('description','')[:180]}</span><br>"
                            f"<a href='{ev.get('url','#')}' target='_blank' "
                            f"style='color:#58A6FF;font-size:.78rem'>🔗 More info →</a>"
                            f"</div>", unsafe_allow_html=True
                        )
                    st.markdown("---")
            except Exception:
                pass

        # All events with category filter
        st.markdown(f"#### All Katz Events ({len(events)} total)")
        cats = ["All"] + sorted(set(ev.get("category","") for ev in events if ev.get("category")))
        cat  = st.selectbox("Filter", cats, key="ev_cat", label_visibility="collapsed")
        show = events if cat == "All" else [e for e in events if e.get("category")==cat]

        COLOR_MAP = {
            "CS & AI Club":"#58A6FF","Research":"#3FB950",
            "Admissions":"#D29922","Graduate School":"#BC8CFF",
        }
        TYPE_ICON = {"competition":"🏆","symposium":"🔬",
                     "info_session":"ℹ️","club":"👥","general":"📢"}
        for ev in show:
            accent = COLOR_MAP.get(ev.get("category",""),"#58A6FF")
            icon   = TYPE_ICON.get(ev.get("type",""),"📅")
            st.markdown(
                f"<div style='background:#161B22;border:1px solid #30363D;"
                f"border-left:4px solid {accent};border-radius:0 8px 8px 0;"
                f"padding:.85rem 1.1rem;margin:.35rem 0'>"
                f"<b style='color:#E6EDF3'>{icon} {ev.get('title','')}</b><br>"
                f"<span style='color:#8B949E;font-size:.76rem'>"
                f"📅 {ev.get('date','TBD')}"
                f"{' · ' + ev.get('time','') if ev.get('time') else ''}"
                f"{' · 📍 ' + ev.get('location','') if ev.get('location') else ''}"
                f"</span><br>"
                f"<span style='color:#C9D1D9;font-size:.82rem'>"
                f"{ev.get('description','')[:180]}</span><br>"
                f"<a href='{ev.get('url','https://www.yu.edu/katz/events')}' "
                f"target='_blank' style='color:#58A6FF;font-size:.76rem'>"
                f"🔗 {ev.get('url','').replace('https://','')[:50]} →</a>"
                f"</div>", unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
# TAB 10 — SMART ADVISOR
# ══════════════════════════════════════════════════════════════
with tab_advisor:
    st.markdown("### 🧠 Smart Advisor")
    st.markdown("""
    <div style="background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;
    padding:.8rem 1rem;font-size:.84rem;color:#58A6FF;margin-bottom:1rem">
    <b>Complete academic loop:</b> Research gap → matching Katz faculty →
    upcoming relevant events → ready-to-send email template.
    </div>""", unsafe_allow_html=True)

    adv_topic = st.session_state.topic or topic
    adv_gaps  = (st.session_state.results.get("gaps","")
                 if st.session_state.results else "")

    advc1, advc2 = st.columns([3,1])
    with advc2:
        run_adv = st.button("🧠 Get Smart Advice", key="run_adv_btn",
                            use_container_width=True, type="primary")
    with advc1:
        st.markdown(
            f"<div style='color:#8B949E;font-size:.84rem;padding:.35rem 0'>"
            f"Topic: <b style='color:#E6EDF3'>{adv_topic or 'run agents first'}</b>"
            f"{'  · Gaps available ✓' if adv_gaps else ''}"
            f"</div>", unsafe_allow_html=True
        )

    if run_adv or (st.session_state.results and not st.session_state.smart_advice):
        with st.spinner("Matching faculty and events…"):
            try:
                from katzbot.smart_advisor import get_smart_advice
                st.session_state.smart_advice = get_smart_advice(
                    topic=adv_topic, gaps_text=adv_gaps
                )
            except Exception as e:
                st.error(f"Smart Advisor error: {e}")
        st.rerun()

    if st.session_state.smart_advice:
        adv = st.session_state.smart_advice

        if adv.get("faculty_matches"):
            st.markdown("#### 👨‍🏫 Recommended Faculty")
            for f in adv["faculty_matches"]:
                exp = " · ".join(f["expertise"][:4])
                st.markdown(
                    f"<div style='background:#161B22;border:0.5px solid #30363D;"
                    f"border-left:4px solid #58A6FF;border-radius:0 8px 8px 0;"
                    f"padding:.9rem 1.1rem;margin:.4rem 0'>"
                    f"<b style='color:#E6EDF3'>{f['name']}</b><br>"
                    f"<span style='color:#8B949E;font-size:.77rem'>"
                    f"{f['title']} · {f['dept']}</span><br>"
                    f"<span style='color:#C9D1D9;font-size:.8rem'>{exp}</span><br>"
                    f"<span style='color:#8B949E;font-size:.77rem;font-style:italic'>"
                    f"{f['note']}</span><br>"
                    f"<a href='mailto:{f['email']}' style='color:#58A6FF;font-size:.8rem'>"
                    f"✉ {f['email']}</a> · "
                    f"<a href='{f['profile']}' target='_blank' "
                    f"style='color:#58A6FF;font-size:.8rem'>Profile →</a>"
                    f"</div>", unsafe_allow_html=True
                )

        if adv.get("event_matches"):
            st.markdown("#### 📅 Relevant Events")
            for ev in adv["event_matches"]:
                st.markdown(
                    f"<div style='background:#1C1A10;border:1px solid #D29922;"
                    f"border-radius:8px;padding:.85rem 1.1rem;margin:.4rem 0'>"
                    f"<b style='color:#D29922'>{ev.get('title','')}</b><br>"
                    f"<span style='color:#8B949E;font-size:.77rem'>"
                    f"📅 {ev.get('date','TBD')} · 📍 {ev.get('location','')}</span><br>"
                    f"<span style='color:#C9D1D9;font-size:.82rem'>"
                    f"{ev.get('description','')[:150]}</span><br>"
                    f"<a href='{ev.get('url','#')}' target='_blank' "
                    f"style='color:#58A6FF;font-size:.78rem'>🔗 More info →</a>"
                    f"</div>", unsafe_allow_html=True
                )

        if adv.get("action_items"):
            st.markdown("#### ✅ Action Items")
            for i, action in enumerate(adv["action_items"], 1):
                st.markdown(
                    f"<div style='background:#161B22;border:0.5px solid #30363D;"
                    f"border-radius:6px;padding:.55rem .9rem;margin:.2rem 0;"
                    f"font-size:.84rem;color:#C9D1D9'>"
                    f"<b style='color:#3FB950'>{i}.</b> {action}</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("#### ✉ Email Template Generator")
        sname = st.text_input("Your name", key="adv_name",
                              placeholder="Enter your name")
        if sname and adv.get("faculty_matches"):
            fac_opts = [f["name"] for f in adv["faculty_matches"]]
            sel_name = st.selectbox("Faculty to email", fac_opts, key="adv_fac")
            sel_fac  = next((f for f in adv["faculty_matches"]
                             if f["name"]==sel_name), adv["faculty_matches"][0])
            top_gap  = ""
            if adv_gaps:
                lines   = [l.strip() for l in adv_gaps.split("\n") if l.strip()]
                top_gap = lines[0][:120] if lines else ""
            try:
                from katzbot.smart_advisor import format_email_template
                email_text = format_email_template(sname, sel_fac, adv_topic, top_gap)
                st.text_area("Copy and send:", value=email_text, height=250,
                             key="email_ta")
                st.download_button("⬇️ Download email",
                    data=email_text,
                    file_name=f"email_{sel_fac['name'].replace(' ','_')}.txt",
                    mime="text/plain", key="dl_email")
            except Exception as e:
                st.error(f"Email error: {e}")

        if st.button("🔄 Re-run Advisor", key="rerun_adv"):
            st.session_state.smart_advice = None
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 11 — CITATIONS
# ══════════════════════════════════════════════════════════════
with tab_citations:
    st.markdown("### 📚 BibTeX Citations — Overleaf Ready")
    st.markdown("""
    <div style="background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;
    padding:.8rem 1rem;font-size:.84rem;color:#58A6FF;margin-bottom:1rem">
    Titles extracted from Crawler output → matched against
    <b>Semantic Scholar API</b> → exact BibTeX (real DOI, authors, venue).
    Download as <code>references.bib</code> → upload to Overleaf.
    </div>""", unsafe_allow_html=True)

    if not st.session_state.results:
        st.info("Run the agents first, then fetch citations.")
    else:
        papers_text   = st.session_state.results.get("papers","")
        current_topic = st.session_state.topic or topic

        if not st.session_state.citations:
            st.markdown("""
            <div style="background:#1C1A10;border:1px solid #D29922;
            border-radius:8px;padding:.7rem 1rem;font-size:.84rem;
            color:#D29922;margin-bottom:.8rem">
            ⚡ Papers found! Click <b>Fetch BibTeX</b> (~2-4 min).
            </div>""", unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns([2,2,3])
        with cc1:
            fetch_cit = st.button("🔍 Fetch BibTeX",
                                  key="fetch_cit", use_container_width=True,
                                  type="primary")
        with cc2:
            if st.session_state.citations:
                st.download_button(
                    "⬇️ Download references.bib",
                    data=st.session_state.bib_file,
                    file_name=f"references_{current_topic[:20].replace(' ','_')}.bib",
                    mime="text/plain",
                    use_container_width=True, key="dl_bib_top",
                )
        with cc3:
            if st.session_state.citations:
                st.markdown(
                    f"<div style='color:#3FB950;font-size:.84rem;padding:.4rem 0'>"
                    f"✅ {len(st.session_state.citations)} verified citations ready</div>",
                    unsafe_allow_html=True
                )

        if fetch_cit:
            prog   = st.progress(0, text="Extracting titles…")
            status = st.empty()
            try:
                from tools.citation_fetcher import (
                    fetch_citations_from_papers_text, build_bib_file,
                    save_bib_to_disk, print_citations_summary,
                )
                prog.progress(15, text="Searching Semantic Scholar…")
                cits = fetch_citations_from_papers_text(
                    papers_text=papers_text,
                    topic=current_topic,
                    delay=2.5,
                )
                prog.progress(90, text="Building .bib file…")
                if cits:
                    bib      = build_bib_file(cits, topic=current_topic)
                    saved_to = save_bib_to_disk(bib, topic=current_topic,
                                                output_dir="outputs")
                    print_citations_summary(cits)
                    st.session_state.citations = cits
                    st.session_state.bib_file  = bib
                    st.session_state.bib_saved = saved_to
                    prog.progress(100, text="Done!")
                    status.success(
                        f"✅ {len(cits)} citations! "
                        f"Auto-saved → `{saved_to}`"
                    )
                else:
                    prog.progress(100, text="No matches")
                    status.warning("No matches found. Try re-running agents.")
            except Exception as e:
                status.error(f"Error: {e}")
            st.rerun()

        if st.session_state.citations:
            cits = st.session_state.citations
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Papers", len(cits))
            m2.metric("With DOI",  sum(1 for c in cits if c.get("doi")))
            m3.metric("On arXiv",  sum(1 for c in cits if c.get("arxiv")))
            m4.metric("Conf/Jour",
                f"{sum(1 for c in cits if '@inproceedings' in c.get('bibtex',''))}"
                f"/{sum(1 for c in cits if '@article' in c.get('bibtex',''))}")
            st.markdown("---")

            with st.expander("📄 Full .bib file — copy or download", expanded=False):
                st.text_area("Select all (Ctrl+A):",
                             value=st.session_state.bib_file, height=300,
                             key="bib_full")
                st.download_button("⬇️ Download references.bib",
                    data=st.session_state.bib_file,
                    file_name=f"references_{current_topic[:20].replace(' ','_')}.bib",
                    mime="text/plain", key="dl_bib_exp")

            st.markdown(f"#### {len(cits)} Individual Citations")
            for i, cit in enumerate(cits, 1):
                key    = cit.get("cite_key", f"paper{i}")
                title  = cit.get("title","")
                auths  = ", ".join(cit.get("authors",[])[:2])
                if len(cit.get("authors",[])) > 2: auths += " et al."
                year   = cit.get("year","")
                doi    = cit.get("doi","")
                arxiv  = cit.get("arxiv","")
                bibtex = cit.get("bibtex","")
                links  = []
                if doi:   links.append(f'<a href="https://doi.org/{doi}" target="_blank" style="color:#58A6FF;font-size:.75rem">DOI</a>')
                if arxiv: links.append(f'<a href="https://arxiv.org/abs/{arxiv}" target="_blank" style="color:#58A6FF;font-size:.75rem">arXiv</a>')
                st.markdown(
                    f"<div style='background:#161B22;border:1px solid #30363D;"
                    f"border-left:3px solid #1F6FEB;border-radius:0 8px 8px 0;"
                    f"padding:.8rem 1.1rem;margin:.3rem 0'>"
                    f"<b style='color:#E6EDF3'>{i}. {title}</b><br>"
                    f"<span style='color:#8B949E;font-size:.76rem'>{auths} · {year}</span><br>"
                    f"<code style='background:#1F3A5F;color:#79C0FF;padding:1px 8px;"
                    f"border-radius:4px;font-size:.75rem'>\\cite{{{key}}}</code>"
                    f"{'  ' + '  '.join(links) if links else ''}"
                    f"</div>", unsafe_allow_html=True
                )
                with st.expander(f"BibTeX [{key}]", expanded=False):
                    st.code(bibtex, language="bibtex")
                    st.download_button(f"⬇️ {key}.bib", data=bibtex,
                        file_name=f"{key}.bib", mime="text/plain",
                        key=f"dl_s_{i}")

            st.markdown("---")
            st.markdown("""
            <div style="background:#161B22;border:0.5px solid #30363D;
            border-radius:8px;padding:1rem 1.2rem;font-size:.84rem">
            <b style="color:#E6EDF3">📎 Overleaf in 3 steps:</b><br>
            <b style="color:#D29922">1.</b> Download <code>references.bib</code> above<br>
            <b style="color:#D29922">2.</b> Overleaf → Upload → select the .bib file<br>
            <b style="color:#D29922">3.</b> Add before <code>\\end{document}</code>:
            <code style="color:#79C0FF">\\bibliographystyle{ieeetr}</code>
            <code style="color:#79C0FF">\\bibliography{references}</code>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 12 — FACULTY MATCH
# ══════════════════════════════════════════════════════════════
with tab_faculty:
    st.markdown("### 👨‍🏫 Faculty Match")
    st.markdown("""
    <div style="background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;
    padding:.8rem 1rem;font-size:.84rem;color:#58A6FF;margin-bottom:1rem">
    Find the right Katz School professor for your research topic.
    Matched by expertise keywords and research focus.
    </div>""", unsafe_allow_html=True)

    fac_topic = st.text_input(
        "Search by topic or faculty name",
        value=st.session_state.topic or topic,
        key="fac_search",
        placeholder="e.g. deep learning, cybersecurity, Prof. Wang…",
    )

    if fac_topic:
        from katzbot.faculty import match_faculty, KATZ_FACULTY
        matches = match_faculty(fac_topic, top_k=5)

        if matches:
            st.markdown(f"#### Top {len(matches)} matches for '{fac_topic}'")
            for f in matches:
                exp = ", ".join(f["expertise"][:5])
                courses = ", ".join(f.get("courses",[])[:3])
                st.markdown(
                    f"<div style='background:#161B22;border:0.5px solid #30363D;"
                    f"border-left:4px solid #58A6FF;border-radius:0 10px 10px 0;"
                    f"padding:1rem 1.2rem;margin:.5rem 0'>"
                    f"<div style='font-size:.95rem;font-weight:600;color:#E6EDF3'>"
                    f"{f['name']}</div>"
                    f"<div style='color:#8B949E;font-size:.78rem;margin:.15rem 0'>"
                    f"{f['title']}</div>"
                    f"<div style='color:#8B949E;font-size:.78rem;margin-bottom:.3rem'>"
                    f"🏛 {f['dept']}</div>"
                    f"<div style='color:#C9D1D9;font-size:.8rem;margin-bottom:.3rem'>"
                    f"<b>Expertise:</b> {exp}</div>"
                    f"{('<div style=\"color:#C9D1D9;font-size:.8rem;margin-bottom:.3rem\"><b>Courses:</b> ' + courses + '</div>') if courses else ''}"
                    f"<div style='color:#8B949E;font-size:.77rem;font-style:italic;"
                    f"margin-bottom:.4rem'>{f['note']}</div>"
                    f"<a href='mailto:{f['email']}' style='color:#58A6FF;font-size:.82rem'>"
                    f"✉ {f['email']}</a>"
                    f" &nbsp;·&nbsp; "
                    f"<a href='{f['profile']}' target='_blank' "
                    f"style='color:#58A6FF;font-size:.82rem'>Profile →</a>"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            st.info("No faculty matched. Try different keywords.")
            st.markdown("**All Katz Faculty:**")
            from katzbot.faculty import KATZ_FACULTY
            for f in KATZ_FACULTY:
                st.markdown(
                    f"• **{f['name']}** ({f['dept']}) — "
                    f"[{f['email']}](mailto:{f['email']})"
                )


# ══════════════════════════════════════════════════════════════
# TAB 13 — REVIEWER AGENT
# ══════════════════════════════════════════════════════════════
with tab_review:
    st.markdown("### 🧑‍⚖️ Reviewer Agent")
    st.markdown("""
    <div style="background:#0A1628;border:1px solid #1F6FEB;border-radius:8px;
    padding:.8rem 1rem;font-size:.84rem;color:#58A6FF;margin-bottom:1rem">
    Analyze your paper like a conference reviewer. Upload a <b>PDF</b> or <b>.tex</b> file,
    choose a target conference and reviewer persona, then get a structured review with
    summary, strengths, weaknesses, venue fit, major questions, and revision priorities.
    </div>""", unsafe_allow_html=True)

    rc1, rc2 = st.columns([1.2, 0.8])
    with rc1:
        uploaded_review_file = st.file_uploader(
            "Upload paper (.pdf, .tex, .txt, .md)",
            type=["pdf", "tex", "txt", "md"],
            key="review_upload",
        )
        pasted_text = st.text_area(
            "Or paste abstract / paper text (optional)",
            height=140,
            key="review_paste",
            placeholder="Paste abstract, intro, or full text here if you do not want to upload a file.",
        )
    with rc2:
        conf = st.selectbox(
            "Target conference",
            ["NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP", "ACM MM", "AAAI", "TMLR", "Interspeech"],
            key="review_conference",
        )
        persona = st.selectbox(
            "Reviewer style",
            ["Balanced", "Skeptical", "Supportive", "Empirical-Rigorous", "Theory/Method", "Area Chair"],
            key="review_persona",
        )
        run_review = st.button(
            "🧑‍⚖️ Analyze Paper",
            key="run_reviewer_agent",
            use_container_width=True,
            type="primary",
        )

    st.caption("Tip: Use Skeptical for stress-testing novelty claims, Empirical-Rigorous for experiment-heavy papers, and Area Chair for high-level venue-fit analysis.")

    if run_review:
        if not uploaded_review_file and not pasted_text.strip():
            st.error("Upload a PDF/TEX/TXT/MD file or paste text first.")
        else:
            try:
                from reviewer_agent import extract_text_from_upload, analyze_paper_text, detect_title

                if uploaded_review_file:
                    review_text, review_title = extract_text_from_upload(uploaded_review_file)
                    review_filename = uploaded_review_file.name
                else:
                    review_text = pasted_text
                    review_title = detect_title(pasted_text, fallback="pasted_paper")
                    review_filename = f"{review_title[:40].replace(' ', '_')}.txt"

                with st.spinner("Reviewer Agent is analyzing the paper…"):
                    report = analyze_paper_text(
                        paper_text=review_text,
                        selected_conference=conf,
                        reviewer_persona=persona,
                        paper_title=review_title,
                    )
                st.session_state.review_report = report
                st.session_state.review_title = review_title
                st.session_state.review_filename = review_filename
                st.success("✅ Review analysis ready below.")
            except Exception as e:
                st.error(f"Reviewer Agent error: {e}")

    if st.session_state.review_report:
        title = st.session_state.review_title or "Uploaded Paper"
        st.markdown(
            f"<div style='background:#161B22;border:0.5px solid #30363D;border-radius:8px;padding:.8rem 1rem;margin:1rem 0'>"
            f"<div style='font-size:.75rem;letter-spacing:.08em;color:#8B949E;text-transform:uppercase'>Detected title</div>"
            f"<div style='font-size:1rem;font-weight:600;color:#E6EDF3'>{title}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download review report (.md)",
            data=st.session_state.review_report,
            file_name=f"review_{title[:40].replace(' ', '_')}.md",
            mime="text/markdown",
            key="dl_review_report",
        )
        st.markdown(st.session_state.review_report)
