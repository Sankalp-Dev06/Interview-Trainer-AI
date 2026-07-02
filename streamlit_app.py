"""
Interview Trainer Agent — Streamlit Web App
Powered by IBM Granite (watsonx.ai) + RAG Pipeline

Features:
  • Light / Dark mode toggle (persisted in session state)
  • Chat-first interface with quick-action buttons
  • Personalised prep plans, question guidance, answer evaluator
  • Role & level explorer with skill pills
  • Live pipeline status panel
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Trainer · AI Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

Path("outputs/logs").mkdir(parents=True, exist_ok=True)
Path("outputs/sessions").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/logs/streamlit.log"),
    ],
)

# ── constants ────────────────────────────────────────────────────────────────
ROLES = {
    "software_engineer": "💻 Software Engineer",
    "data_scientist":    "📊 Data Scientist",
    "product_manager":   "🗂️ Product Manager",
    "devops_engineer":   "⚙️ DevOps Engineer",
}
LEVELS = {
    "entry":  "🌱 Entry  (0–2 yrs)",
    "mid":    "🚀 Mid    (3–6 yrs)",
    "senior": "⭐ Senior (7+ yrs)",
}

MODE_COLORS = {"ibm": "#0f62fe", "local": "#24a148", "stub": "#ff832b"}
MODE_LABELS = {"ibm": "IBM watsonx.ai", "local": "Local model", "stub": "Stub mode"}

# ── theme helpers ────────────────────────────────────────────────────────────
def _is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


def _theme() -> dict:
    """Return colour tokens for the active theme."""
    if _is_dark():
        return dict(
            # surfaces
            bg          = "#0d0d1f",
            surface     = "#13132b",
            surface2    = "#1a1a3a",
            border      = "#2a2a5a",
            border2     = "#3a3a6a",
            # text
            text        = "#e8e8f8",
            text_muted  = "#8888b8",
            text_dim    = "#5a5a8a",
            # brand
            accent      = "#0f62fe",
            accent2     = "#6929c4",
            accent3     = "#ee5396",
            # bubbles
            bubble_agent_bg     = "#1a1a3a",
            bubble_agent_border = "#3a3a7a",
            bubble_agent_text   = "#d4d4f0",
            # sidebar
            sidebar_bg  = "linear-gradient(180deg, #0a0a1a 0%, #0f0f2e 100%)",
            sidebar_border = "#1e1e4a",
            # cards
            card_bg     = "#13132b",
            card_border = "#2a2a5a",
            card_shadow = "rgba(15,98,254,.12)",
        )
    else:
        return dict(
            # surfaces
            bg          = "#f7f8fa",
            surface     = "#ffffff",
            surface2    = "#eef0f5",
            border      = "#e5e7eb",
            border2     = "#d1d5db",
            # text
            text        = "#1f2328",
            text_muted  = "#57606a",
            text_dim    = "#8b949e",
            # brand
            accent      = "#0f62fe",
            accent2     = "#6929c4",
            accent3     = "#ee5396",
            # bubbles
            bubble_agent_bg     = "#eef3ff",
            bubble_agent_border = "#c7d7ff",
            bubble_agent_text   = "#1a1a3a",
            # sidebar
            sidebar_bg  = "linear-gradient(180deg, #1a1f3c 0%, #242851 100%)",
            sidebar_border = "#2d3366",
            # cards
            card_bg     = "#ffffff",
            card_border = "#e5e7eb",
            card_shadow = "rgba(15,98,254,.06)",
        )


def inject_css():
    t = _theme()
    st.markdown(f"""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

/* ── global app background & text ── */
html, body, .stApp, [class*="css"] {{
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
    background-color: {t['bg']} !important;
    color: {t['text']} !important;
}}

/* ── main content area ── */
.block-container {{
    background-color: {t['bg']} !important;
    padding-top: 1.5rem !important;
}}

/* ── headings, markdown text ── */
h1, h2, h3, h4, p, li, label, .stMarkdown {{
    color: {t['text']} !important;
}}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {t['surface2']} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1.5px solid {t['border']} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {t['text_muted']} !important;
    border-radius: 9px !important;
    font-size: .85rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    transition: background .15s, color .15s !important;
}}
.stTabs [aria-selected="true"] {{
    background: {t['accent']} !important;
    color: #fff !important;
}}

/* ── sidebar ── */
section[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']} !important;
    border-right: 1px solid {t['sidebar_border']} !important;
}}
section[data-testid="stSidebar"] * {{ color: #d4d4f0 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: #2a2a5a !important; }}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div > input {{
    background: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    color: #d4d4f0 !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stSlider > div {{ color: #a0a0d0 !important; }}

/* ── metrics ── */
[data-testid="stMetricValue"] {{ color: {t['text']} !important; }}
[data-testid="stMetricLabel"] {{ color: {t['text_muted']} !important; }}

/* ── text areas & inputs in main area ── */
.stTextArea textarea,
.stTextInput > div > div > input {{
    background: {t['surface']} !important;
    color: {t['text']} !important;
    border: 1.5px solid {t['border2']} !important;
    border-radius: 10px !important;
    font-size: .9rem !important;
}}
.stTextArea textarea:focus,
.stTextInput > div > div > input:focus {{
    border-color: {t['accent']} !important;
    box-shadow: 0 0 0 3px rgba(15,98,254,.14) !important;
}}

/* ── selectbox ── */
.stSelectbox > div > div {{
    background: {t['surface']} !important;
    color: {t['text']} !important;
    border: 1.5px solid {t['border2']} !important;
    border-radius: 10px !important;
}}

/* ── expander ── */
.streamlit-expanderHeader {{
    background: {t['surface2']} !important;
    color: {t['text']} !important;
    border-radius: 8px !important;
    border: 1px solid {t['border']} !important;
}}
.streamlit-expanderContent {{
    background: {t['surface']} !important;
    border: 1px solid {t['border']} !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}}

/* ── divider ── */
hr {{ border-color: {t['border']} !important; }}

/* ── buttons ── */
.stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform .12s, box-shadow .12s !important;
    background-color: {t['surface2']} !important;
    color: {t['text']} !important;
    border: 1.5px solid {t['border2']} !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(15,98,254,.22) !important;
    border-color: {t['accent']} !important;
    color: {t['accent']} !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {t['accent']} 0%, {t['accent2']} 100%) !important;
    color: #fff !important;
    border: none !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(15,98,254,.35) !important;
    color: #fff !important;
}}

/* ── download button ── */
.stDownloadButton > button {{
    background: linear-gradient(135deg,#24a148,#0e8a38) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}}

/* ── code blocks ── */
.stCode, code, pre {{
    background: {t['surface2']} !important;
    color: {t['text']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
}}

/* ── info / warning / error boxes ── */
.stAlert {{
    border-radius: 10px !important;
    border-left-width: 4px !important;
}}

/* ── hero banner ── */
.hero {{
    background: linear-gradient(135deg, #0f62fe 0%, #6929c4 55%, #ee5396 100%);
    border-radius: 16px;
    padding: 28px 36px 24px;
    margin-bottom: 4px;
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: "🎯";
    position: absolute;
    right: 32px; top: 18px;
    font-size: 68px;
    opacity: .13;
}}
.hero h1 {{ margin: 0 0 6px; font-size: 1.9rem; font-weight: 700; color: #fff !important; }}
.hero p  {{ margin: 0; font-size: .97rem; color: rgba(255,255,255,.85) !important; }}

/* ── stat chips ── */
.chip-row {{ display: flex; gap: 8px; margin: 12px 0 18px; flex-wrap: wrap; }}
.chip {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 13px; border-radius: 999px;
    font-size: .78rem; font-weight: 600; letter-spacing: .02em;
    border: 1.5px solid;
}}
.chip-blue  {{ background:#dbeafe; color:#1d4ed8; border-color:#93c5fd; }}
.chip-green {{ background:#dcfce7; color:#166534; border-color:#86efac; }}
.chip-amber {{ background:#fef3c7; color:#92400e; border-color:#fcd34d; }}
.chip-pink  {{ background:#fce7f3; color:#9d174d; border-color:#f9a8d4; }}
.chip-purple{{ background:#ede9fe; color:#5b21b6; border-color:#c4b5fd; }}

/* ── chat container ── */
.chat-wrap {{
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 6px 0 20px;
}}

/* ── chat bubbles ── */
.msg {{
    display: flex;
    gap: 10px;
    align-items: flex-start;
    max-width: 88%;
    animation: fadeUp .22s ease;
}}
.msg.user  {{ flex-direction: row-reverse; align-self: flex-end; }}
.msg.agent {{ align-self: flex-start; }}

.avatar {{
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}}
.avatar-agent {{ background: linear-gradient(135deg,#0f62fe,#6929c4); color:#fff; }}
.avatar-user  {{ background: linear-gradient(135deg,#ee5396,#ff832b); color:#fff; }}

.bubble {{
    padding: 11px 15px;
    border-radius: 16px;
    font-size: .9rem;
    line-height: 1.65;
    max-width: 100%;
    white-space: pre-wrap;
    word-break: break-word;
}}
.bubble-agent {{
    background: {t['bubble_agent_bg']};
    border: 1.5px solid {t['bubble_agent_border']};
    border-top-left-radius: 4px;
    color: {t['bubble_agent_text']};
}}
.bubble-user {{
    background: linear-gradient(135deg,#0f62fe,#6929c4);
    color: #fff;
    border-top-right-radius: 4px;
}}

.msg-time {{
    font-size: .68rem;
    color: {t['text_dim']};
    margin-top: 3px;
    text-align: right;
}}

@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(8px); }}
    to   {{ opacity:1; transform:translateY(0);   }}
}}

/* ── typing indicator ── */
.typing {{ display:flex; gap:5px; align-items:center; padding:6px 0; }}
.typing span {{
    width:7px; height:7px; border-radius:50%;
    background:#0f62fe;
    animation: bounce 1.2s infinite;
}}
.typing span:nth-child(2) {{ animation-delay:.2s; }}
.typing span:nth-child(3) {{ animation-delay:.4s; }}
@keyframes bounce {{
    0%,80%,100% {{ transform:translateY(0); opacity:.6; }}
    40%          {{ transform:translateY(-5px); opacity:1; }}
}}

/* ── mode badge ── */
.badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 9px; border-radius: 999px;
    font-size: .71rem; font-weight: 700;
    border: 1.5px solid;
}}

/* ── section cards ── */
.card {{
    background: {t['card_bg']};
    border: 1.5px solid {t['card_border']};
    border-radius: 14px;
    padding: 18px 22px;
    margin: 8px 0;
    box-shadow: 0 2px 12px {t['card_shadow']};
    color: {t['text']};
}}
.card-title {{
    font-size: .77rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    color: {t['accent2']}; margin: 0 0 10px;
}}

/* ── welcome card ── */
.welcome-card {{
    background: {t['card_bg']};
    border: 1.5px solid {t['card_border']};
    border-radius: 14px;
    padding: 28px 28px 24px;
    text-align: center;
    box-shadow: 0 2px 12px {t['card_shadow']};
}}
.welcome-card h2 {{ color: {t['accent']} !important; font-size: 1.6rem; margin-bottom: 6px; }}
.welcome-card p  {{ color: {t['text_muted']} !important; font-size: .95rem; margin: 0; }}

/* ── skill pills ── */
.skill-pill {{
    display: inline-block;
    background: #ede9fe; color: #5b21b6;
    border: 1px solid #c4b5fd;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: .77rem; font-weight: 500;
    margin: 2px 3px 2px 0;
}}
.skill-pill.blue  {{ background:#dbeafe; color:#1d4ed8; border-color:#93c5fd; }}
.skill-pill.pink  {{ background:#fce7f3; color:#9d174d; border-color:#f9a8d4; }}
.skill-pill.green {{ background:#dcfce7; color:#166534; border-color:#86efac; }}

/* ── quick-action grid ── */
.qa-hint {{
    font-size: .8rem; color: {t['text_muted']};
    margin: 0 0 8px;
    font-weight: 500;
}}

/* ── progress steps ── */
.step-row {{
    display: flex; gap: 8px; margin: 10px 0 16px;
    align-items: flex-start; flex-wrap: wrap;
}}
.step {{
    display: flex; align-items: center; gap: 6px;
    background: {t['surface2']};
    border: 1.5px solid {t['border']};
    border-radius: 10px;
    padding: 8px 14px;
    font-size: .82rem; font-weight: 500;
    color: {t['text_muted']};
    flex: 1; min-width: 130px;
}}
.step.active {{
    border-color: {t['accent']};
    color: {t['accent']};
    background: rgba(15,98,254,.08);
    font-weight: 600;
}}

/* ── theme toggle button (sidebar) ── */
.theme-toggle {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 14px; border-radius: 10px;
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.18);
    color: #d4d4f0;
    font-size: .82rem; font-weight: 600;
    cursor: pointer; margin-bottom: 8px;
}}

/* ── scrollable chat area ── */
.chat-scroll {{
    max-height: 520px;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: {t['border2']} transparent;
}}
.chat-scroll::-webkit-scrollbar {{ width: 5px; }}
.chat-scroll::-webkit-scrollbar-thumb {{ background: {t['border2']}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


# ── pipeline loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Warming up your AI Interview Coach…")
def load_pipeline():
    from src.rag_pipeline import InterviewRAGPipeline
    p = InterviewRAGPipeline(config_path="config/config.yaml")
    p.initialize()
    return p


# ── chat helpers ──────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%H:%M")


def chat_agent(text: str):
    st.session_state.setdefault("chat", []).append(
        {"role": "agent", "text": text, "time": _now()}
    )


def chat_user(text: str):
    st.session_state.setdefault("chat", []).append(
        {"role": "user", "text": text, "time": _now()}
    )


def render_chat():
    messages = st.session_state.get("chat", [])
    if not messages:
        return
    parts = ['<div class="chat-scroll"><div class="chat-wrap">']
    for m in messages:
        role = m["role"]
        side = "user" if role == "user" else "agent"
        avatar = "🙋" if role == "user" else "🤖"
        safe_text = (
            m["text"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        parts.append(
            f'<div class="msg {side}">'
            f'<div class="avatar avatar-{side}">{avatar}</div>'
            f'<div><div class="bubble bubble-{side}">{safe_text}</div>'
            f'<div class="msg-time">{m["time"]}</div></div>'
            f'</div>'
        )
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def badge_html(mode: str) -> str:
    color = MODE_COLORS.get(mode, "#888")
    label = MODE_LABELS.get(mode, mode)
    dot = "🟢" if mode == "ibm" else ("🟠" if mode == "stub" else "🔵")
    return (
        f'<span class="badge" style="color:{color};border-color:{color}55;'
        f'background:{color}18">{dot} {label}</span>'
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(pipeline) -> dict:
    t = _theme()
    with st.sidebar:
        # Logo / branding
        st.markdown(
            '<div style="text-align:center;padding:16px 0 8px">'
            '<div style="font-size:2.6rem">🎯</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:#a78bfa;margin-top:2px">Interview Coach</div>'
            '<div style="font-size:.72rem;color:#6b6b9a;margin-top:1px">Powered by IBM Granite</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Light / Dark mode toggle ──────────────────────────────────
        dark = _is_dark()
        icon  = "🌙 Dark mode" if dark else "☀️ Light mode"
        label = "Switch to Light" if dark else "Switch to Dark"
        if st.button(f"{icon}  ·  {label}", use_container_width=True, key="theme_toggle"):
            st.session_state["dark_mode"] = not dark
            st.rerun()

        st.markdown("---")

        # ── Profile ───────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.77rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.1em;color:#7c7caa">👤 Your Profile</p>',
            unsafe_allow_html=True,
        )
        name = st.text_input(
            "Name", placeholder="e.g. Alex Johnson",
            key="sb_name", label_visibility="collapsed",
        )
        st.caption("Your name")

        role_key = st.selectbox(
            "Target Role",
            options=list(ROLES.keys()),
            format_func=lambda k: ROLES[k],
            key="sb_role",
        )
        level_key = st.selectbox(
            "Experience Level",
            options=list(LEVELS.keys()),
            format_func=lambda k: LEVELS[k],
            key="sb_level",
        )

        st.markdown("---")

        # ── Settings ──────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.77rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.1em;color:#7c7caa">⚙️ Settings</p>',
            unsafe_allow_html=True,
        )
        top_k = st.slider("Context docs (top-k)", 3, 10, 5, key="sb_topk")

        # ── Pipeline status ───────────────────────────────────────────
        if pipeline:
            st.markdown("---")
            st.markdown(
                '<p style="font-size:.77rem;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.1em;color:#7c7caa">📡 System</p>',
                unsafe_allow_html=True,
            )
            llm_m = pipeline.llm_client.mode
            emb_m = pipeline.embedding_engine.mode
            docs  = pipeline.vector_store.document_count
            st.markdown(
                f"LLM &nbsp; {badge_html(llm_m)}<br>"
                f"Embed &nbsp; {badge_html(emb_m)}<br>"
                f"<span style='font-size:.78rem;color:#888'>📚 {docs} docs indexed</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🗑️ Clear Session", use_container_width=True):
            for k in ["chat", "prep_result", "guidance_result", "eval_result", "role_info"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.caption("Interview Trainer Agent v1.0\nIBM Granite · FAISS · watsonx.ai")

    return {
        "name":  name.strip(),
        "role":  role_key,
        "level": level_key,
        "top_k": top_k,
    }


# ── hero banner ───────────────────────────────────────────────────────────────
def render_hero(profile: dict):
    name_display = f", {profile['name']}" if profile["name"] else ""
    role_display = ROLES[profile["role"]].split(" ", 1)[-1]
    lvl_display  = LEVELS[profile["level"]].split(" ", 1)[-1]
    st.markdown(
        f'<div class="hero">'
        f'<h1>Hey{name_display}! 👋</h1>'
        f'<p>Your AI Interview Coach for <strong>{role_display}</strong> · {lvl_display}'
        f' &nbsp;—&nbsp; ask me anything or pick a tab below.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Quick-glance chips below hero
    role_disp  = ROLES[profile["role"]].split(" ", 1)[-1]
    level_disp = LEVELS[profile["level"]].split(" ", 1)[-1]
    name_chip  = f'<span class="chip chip-green">👤 {profile["name"]}</span>' if profile["name"] else ""
    st.markdown(
        f'<div class="chip-row">'
        f'<span class="chip chip-blue">💼 {role_disp}</span>'
        f'<span class="chip chip-pink">🚀 {level_disp}</span>'
        f'{name_chip}'
        f'<span class="chip chip-purple">🤖 IBM Granite</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── tab: chat coach ───────────────────────────────────────────────────────────
def tab_chat(pipeline, profile: dict):
    if not profile["name"]:
        st.markdown(
            '<div class="welcome-card">'
            '<h2>Welcome to Interview Coach 🎯</h2>'
            '<p>Enter your <strong>name</strong> in the sidebar to start your personalised session.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # First-load greeting
    if "chat" not in st.session_state:
        chat_agent(
            f"Hi {profile['name']}! 👋 I'm your Interview Coach.\n\n"
            f"I'm here to help you prepare for a **{ROLES[profile['role']].split(' ',1)[-1]}** "
            f"role at the **{LEVELS[profile['level']].split(' ',1)[-1]}** level.\n\n"
            "You can:\n"
            "• Ask me any interview question 💡\n"
            "• Type your answer and I'll evaluate it 📊\n"
            "• Ask for a full prep plan 📋\n\n"
            "What would you like to work on today?"
        )

    # Chat history
    render_chat()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Quick-action row
    st.markdown('<p class="qa-hint">⚡ Quick actions — click to send instantly:</p>', unsafe_allow_html=True)
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    q1 = qcol1.button("📋 Full Prep Plan",     use_container_width=True, key="qb1")
    q2 = qcol2.button("💡 Give me a question", use_container_width=True, key="qb2")
    q3 = qcol3.button("🏆 Evaluate an answer", use_container_width=True, key="qb3")
    q4 = qcol4.button("🗺️ Role overview",      use_container_width=True, key="qb4")

    # Message input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_msg = st.text_area(
            "Message",
            placeholder="Ask me a question, paste an interview Q, or type 'prep plan'…",
            height=80,
            key="chat_input",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        send = st.button("Send ➤", type="primary", use_container_width=True, key="chat_send")

    # Resolve trigger
    trigger_text = None
    if send and user_msg.strip():
        trigger_text = user_msg.strip()
    elif q1:
        trigger_text = "Give me a full interview preparation plan."
    elif q2:
        trigger_text = "Give me a challenging interview question for my role and level."
    elif q3:
        trigger_text = "How should I evaluate my answers? Give me an example question and let me answer it."
    elif q4:
        trigger_text = "Give me an overview of the key skills and interview rounds for my role."

    if trigger_text:
        chat_user(trigger_text)
        with st.spinner(""):
            try:
                lower = trigger_text.lower()
                if any(w in lower for w in ["prep plan", "preparation plan", "full plan", "plan"]):
                    result = pipeline.generate_prep_plan(
                        name=profile["name"], role=profile["role"], level=profile["level"]
                    )
                    response = result["preparation_plan"]
                    st.session_state["prep_result"] = result
                elif any(w in lower for w in ["overview", "role", "skills", "rounds"]):
                    info = pipeline.get_role_info(role=profile["role"], level=profile["level"])
                    li = info.get("level_info", {})
                    ps = info.get("preparation_strategy", {})
                    lines = [
                        f"**{info.get('role', profile['role'])} — {LEVELS[profile['level']]}**\n",
                        f"🛠 **Core Skills:** {', '.join(li.get('core_skills', []))}",
                        f"🤝 **Soft Skills:** {', '.join(li.get('soft_skills', []))}",
                        f"🔄 **Interview Rounds:** {', '.join(li.get('interview_rounds', []))}",
                    ]
                    if ps:
                        lines.append(f"📅 **Prep Timeline:** {ps.get('timeline_weeks','?')} weeks")
                        lines.append(f"🎯 **Focus Areas:** {', '.join(ps.get('focus_areas',[]))}")
                    response = "\n".join(lines)
                else:
                    result = pipeline.get_question_guidance(
                        name=profile["name"], role=profile["role"],
                        level=profile["level"], question=trigger_text,
                    )
                    response = result["guidance"]
                    st.session_state["guidance_result"] = result
                chat_agent(response)
            except Exception as e:
                chat_agent(f"⚠️ Sorry, something went wrong: {e}")
        st.rerun()


# ── tab: prep plan ────────────────────────────────────────────────────────────
def tab_prep_plan(pipeline, profile: dict):
    st.markdown("### 📋 Personalised Interview Prep Plan")
    st.markdown(
        "Generate a complete plan: question set, model answers, improvement tips, "
        "and a study timeline — all tailored to your role and level."
    )

    if not profile["name"]:
        st.info("👈 Enter your name in the sidebar first.")
        return

    role_disp  = ROLES[profile["role"]].split(" ", 1)[-1]
    level_disp = LEVELS[profile["level"]].split(" ", 1)[-1]

    st.markdown(
        f'<div class="chip-row">'
        f'<span class="chip chip-blue">💻 {role_disp}</span>'
        f'<span class="chip chip-pink">🚀 {level_disp}</span>'
        f'<span class="chip chip-green">👤 {profile["name"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Steps indicator
    st.markdown(
        '<div class="step-row">'
        '<div class="step active">① Select role &amp; level</div>'
        '<div class="step active">② Generate plan</div>'
        '<div class="step">③ Download &amp; study</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("🚀 Generate My Prep Plan", type="primary"):
        with st.spinner("🧠 IBM Granite is crafting your personalised plan…"):
            try:
                result = pipeline.generate_prep_plan(
                    name=profile["name"], role=profile["role"], level=profile["level"]
                )
                st.session_state["prep_result"] = result
            except Exception as e:
                st.error(f"❌ {e}")
                return

    if "prep_result" in st.session_state:
        result = st.session_state["prep_result"]
        st.markdown('<div class="card"><div class="card-title">✅ Your Prep Plan</div>', unsafe_allow_html=True)
        st.markdown(result["preparation_plan"])
        st.markdown("</div>", unsafe_allow_html=True)

        col_dl, _ = st.columns([1, 4])
        with col_dl:
            st.download_button(
                "⬇️ Download Plan",
                data=result["preparation_plan"],
                file_name=f"prep_{profile['name'].replace(' ','_')}_{profile['role']}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with st.expander("🔍 RAG Context retrieved", expanded=False):
            st.code(result["context_retrieved"], language=None)


# ── tab: question guidance ────────────────────────────────────────────────────
def tab_guidance(pipeline, profile: dict):
    st.markdown("### 💡 Question-Specific Guidance")
    st.markdown(
        "Paste any interview question — get a model answer, "
        "what the interviewer is really testing, common mistakes, and follow-up questions."
    )

    if not profile["name"]:
        st.info("👈 Enter your name in the sidebar first.")
        return

    # Preset buttons write into session state so the text_area picks them up
    st.markdown("**Try one of these common questions:**")
    pc1, pc2, pc3 = st.columns(3)
    if pc1.button("🙋 Tell me about yourself",          use_container_width=True):
        st.session_state["qg_question"] = "Tell me about yourself."
        st.session_state.pop("guidance_result", None)
        st.rerun()
    if pc2.button("🔥 Describe a challenging project",  use_container_width=True):
        st.session_state["qg_question"] = "Describe a challenging project you worked on and what you learned."
        st.session_state.pop("guidance_result", None)
        st.rerun()
    if pc3.button("🔭 Where do you see yourself in 5y?", use_container_width=True):
        st.session_state["qg_question"] = "Where do you see yourself in 5 years?"
        st.session_state.pop("guidance_result", None)
        st.rerun()

    question = st.text_area(
        "Your interview question",
        placeholder="Paste or type any interview question here…",
        height=100,
        key="qg_question",
    )

    if st.button("💬 Get Expert Guidance", type="primary", key="qg_submit"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return
        with st.spinner("🧠 Analysing and generating guidance…"):
            try:
                result = pipeline.get_question_guidance(
                    name=profile["name"], role=profile["role"],
                    level=profile["level"], question=question.strip(),
                )
                st.session_state["guidance_result"] = result
            except Exception as e:
                st.error(f"❌ {e}")
                return

    if "guidance_result" in st.session_state:
        result = st.session_state["guidance_result"]
        # Card wrapper (no HTML escaping — use st.markdown inside for proper Markdown rendering)
        st.markdown(
            '<div style="display:flex;gap:12px;margin-top:16px;align-items:flex-start">'
            '<div class="avatar avatar-agent" style="width:38px;height:38px;flex-shrink:0">🤖</div>'
            '<div class="card" style="flex:1;margin:0">'
            '<div class="card-title">💡 Guidance from your Coach</div>',
            unsafe_allow_html=True,
        )
        # Render guidance as native Markdown so **bold**, _italic_, bullet lists work
        st.markdown(result["guidance"])
        st.markdown('</div></div>', unsafe_allow_html=True)

        with st.expander("🔍 RAG Context retrieved", expanded=False):
            st.code(result["context_retrieved"], language=None)


# ── tab: answer evaluator ─────────────────────────────────────────────────────
def tab_evaluator(pipeline, profile: dict):
    st.markdown("### 🏆 Answer Evaluator")
    st.markdown(
        "Write your answer to an interview question — the AI coach will score it "
        "and give you structured feedback plus an improved model answer."
    )

    if not profile["name"]:
        st.info("👈 Enter your name in the sidebar first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**❓ The Question**")
        question = st.text_area(
            "question", label_visibility="collapsed",
            placeholder="Paste the interview question here…",
            height=150, key="ev_q",
        )
    with col2:
        st.markdown("**✍️ Your Answer**")
        answer = st.text_area(
            "answer", label_visibility="collapsed",
            placeholder="Write your answer as you would in a real interview…",
            height=150, key="ev_a",
        )

    if st.button("📊 Evaluate My Answer", type="primary"):
        if not question.strip() or not answer.strip():
            st.warning("Please fill in both the question and your answer.")
            return
        with st.spinner("🧠 Evaluating your answer…"):
            try:
                result = pipeline.evaluate_candidate_answer(
                    name=profile["name"], role=profile["role"], level=profile["level"],
                    question=question.strip(), candidate_answer=answer.strip(),
                )
                st.session_state["eval_result"] = result
            except Exception as e:
                st.error(f"❌ {e}")
                return

    if "eval_result" in st.session_state:
        result = st.session_state["eval_result"]
        st.markdown("---")

        # User bubble — their answer
        safe_ans = result['candidate_answer'].replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        st.markdown(
            f'<div class="msg user" style="max-width:78%;margin:10px 0 0 auto">'
            f'<div class="avatar avatar-user">🙋</div>'
            f'<div><div class="bubble bubble-user">{safe_ans}</div>'
            f'<div class="msg-time">{_now()}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Agent feedback bubble
        safe_fb = result["feedback"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        st.markdown(
            f'<div class="msg agent" style="max-width:88%;margin:10px 0">'
            f'<div class="avatar avatar-agent">🤖</div>'
            f'<div><div class="bubble bubble-agent" style="white-space:pre-wrap">{safe_fb}</div>'
            f'<div class="msg-time">{_now()}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🔍 RAG Context retrieved", expanded=False):
            st.code(result["context_retrieved"], language=None)


# ── tab: role explorer ────────────────────────────────────────────────────────
def tab_role_explorer(pipeline, profile: dict):
    st.markdown("### 🗺️ Role & Level Explorer")
    st.markdown("Browse skills, interview rounds, key companies, and prep strategy for any role and level.")

    col1, col2 = st.columns(2)
    with col1:
        role_key = st.selectbox(
            "Role", list(ROLES.keys()),
            format_func=lambda k: ROLES[k],
            key="re_role",
            index=list(ROLES.keys()).index(profile["role"]),
        )
    with col2:
        level_key = st.selectbox(
            "Level", ["entry", "mid", "senior"],
            format_func=lambda k: LEVELS.get(k, k.title()),
            key="re_level",
            index=(["entry","mid","senior"].index(profile["level"])
                   if profile["level"] in ["entry","mid","senior"] else 0),
        )

    if st.button("🔎 Show Role Details", type="primary"):
        try:
            info = pipeline.get_role_info(role=role_key, level=level_key)
            st.session_state["role_info"] = info
        except Exception as e:
            st.error(f"❌ {e}")
            return

    if "role_info" in st.session_state:
        info = st.session_state["role_info"]
        if not info:
            st.warning("No info found for this role/level combination.")
            return

        li = info.get("level_info", {})
        ps = info.get("preparation_strategy", {})

        st.markdown(f"## {ROLES[role_key]} · {LEVELS[level_key]}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="card-title">🛠️ Core Skills</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="skill-pill blue">{s}</span>' for s in li.get("core_skills", []))
            st.markdown(pills + "</div>", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:8px"><div class="card-title">🤝 Soft Skills</div>', unsafe_allow_html=True)
            pills2 = "".join(f'<span class="skill-pill">{s}</span>' for s in li.get("soft_skills", []))
            st.markdown(pills2 + "</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card"><div class="card-title">🔄 Interview Rounds</div>', unsafe_allow_html=True)
            rounds_html = "".join(
                f'<div style="padding:5px 0;border-bottom:1px solid #eee;font-size:.87rem">🔹 {r}</div>'
                for r in li.get("interview_rounds", [])
            )
            st.markdown(rounds_html + "</div>", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:8px"><div class="card-title">🏢 Key Companies</div>', unsafe_allow_html=True)
            cos = "".join(f'<span class="skill-pill pink">{c}</span>' for c in li.get("key_companies", []))
            st.markdown(cos + "</div>", unsafe_allow_html=True)

        if ps:
            st.markdown('<div class="card"><div class="card-title">📅 Preparation Strategy</div>', unsafe_allow_html=True)
            wk = ps.get("timeline_weeks", "?")
            st.markdown(f'<span class="chip chip-amber">⏱ {wk}-week plan</span>', unsafe_allow_html=True)
            fa_pills = "".join(f'<span class="skill-pill green">{fa}</span>' for fa in ps.get("focus_areas", []))
            st.markdown(f"<br><strong>Focus areas:</strong><br>{fa_pills}", unsafe_allow_html=True)
            if "daily_plan" in ps:
                dp = ps["daily_plan"]
                st.markdown(
                    f"<br><strong>Daily plan:</strong>"
                    f"<br>📅 <em>Weekday:</em> {dp.get('weekday','')}"
                    f"<br>🗓️ <em>Weekend:</em> {dp.get('weekend','')}",
                    unsafe_allow_html=True,
                )
            if ps.get("resources"):
                res_html = "".join(f"<li>📚 {r}</li>" for r in ps["resources"])
                st.markdown(f"<br><strong>Resources:</strong><ul>{res_html}</ul>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ── tab: status ───────────────────────────────────────────────────────────────
def tab_status(pipeline):
    st.markdown("### ⚙️ Pipeline Status")

    status = pipeline.status()

    c1, c2, c3 = st.columns(3)
    c1.metric("Initialized",   "✅ Yes" if status["initialized"] else "❌ No")
    c2.metric("Docs indexed",  status["documents_indexed"])
    c3.metric("Index path",    status["index_path"])

    st.markdown("---")
    ca, cb = st.columns(2)
    ca.markdown(f"**LLM** &nbsp;&nbsp; {badge_html(status['llm_mode'])}", unsafe_allow_html=True)
    cb.markdown(f"**Embedding** &nbsp;&nbsp; {badge_html(status['embedding_mode'])}", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Rebuild Vector Store", type="secondary"):
        with st.spinner("Rebuilding index from raw data…"):
            try:
                pipeline.initialize(force_rebuild=True)
                st.success("✅ Vector store rebuilt successfully.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # Initialise theme default (dark)
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True

    # Inject theme-aware CSS
    inject_css()

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"**Pipeline initialisation failed:** {e}")
        st.stop()

    profile = render_sidebar(pipeline)
    render_hero(profile)

    tabs = st.tabs([
        "💬 Chat Coach",
        "📋 Prep Plan",
        "💡 Question Guide",
        "🏆 Answer Evaluator",
        "🗺️ Role Explorer",
        "⚙️ Status",
    ])

    with tabs[0]: tab_chat(pipeline, profile)
    with tabs[1]: tab_prep_plan(pipeline, profile)
    with tabs[2]: tab_guidance(pipeline, profile)
    with tabs[3]: tab_evaluator(pipeline, profile)
    with tabs[4]: tab_role_explorer(pipeline, profile)
    with tabs[5]: tab_status(pipeline)


if __name__ == "__main__":
    main()
