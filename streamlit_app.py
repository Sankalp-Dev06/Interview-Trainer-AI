"""
Interview Trainer Agent — Streamlit Web App  (chat-first redesign)
Powered by IBM Granite (watsonx.ai) + RAG Pipeline
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
    "entry":  "🌱 Entry  (0-2 yrs)",
    "mid":    "🚀 Mid    (3-6 yrs)",
    "senior": "⭐ Senior (7+ yrs)",
}

# Accent colour per mode (used in chat bubbles & badges)
MODE_COLORS = {"ibm": "#0f62fe", "local": "#24a148", "stub": "#ff832b"}
MODE_LABELS = {"ibm": "IBM watsonx.ai", "local": "Local model", "stub": "Stub mode"}

# ── global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #0f0f2e 100%);
    border-right: 1px solid #1e1e4a;
}
section[data-testid="stSidebar"] * { color: #d4d4f0 !important; }
section[data-testid="stSidebar"] hr { border-color: #2a2a5a !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #1a1a3a !important;
    border: 1px solid #3a3a6a !important;
    color: #d4d4f0 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stSlider > div { color: #a0a0d0 !important; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f62fe 0%, #6929c4 55%, #ee5396 100%);
    border-radius: 16px;
    padding: 32px 36px 28px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🎯";
    position: absolute;
    right: 32px; top: 20px;
    font-size: 72px;
    opacity: .15;
}
.hero h1 { margin: 0 0 6px; font-size: 2.1rem; font-weight: 700; color: #fff; }
.hero p  { margin: 0; font-size: 1rem; color: rgba(255,255,255,.82); }

/* ── stat chips ── */
.chip-row { display: flex; gap: 10px; margin: 14px 0 22px; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 999px;
    font-size: .8rem; font-weight: 600; letter-spacing: .03em;
    border: 1.5px solid;
}
.chip-blue  { background:#dbeafe; color:#1d4ed8; border-color:#93c5fd; }
.chip-green { background:#dcfce7; color:#166534; border-color:#86efac; }
.chip-amber { background:#fef3c7; color:#92400e; border-color:#fcd34d; }
.chip-pink  { background:#fce7f3; color:#9d174d; border-color:#f9a8d4; }

/* ── chat container ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 8px 0 24px;
}

/* ── chat bubbles ── */
.msg {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    max-width: 92%;
    animation: fadeUp .25s ease;
}
.msg.user  { flex-direction: row-reverse; align-self: flex-end; }
.msg.agent { align-self: flex-start; }

.avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.avatar-agent { background: linear-gradient(135deg,#0f62fe,#6929c4); color:#fff; }
.avatar-user  { background: linear-gradient(135deg,#ee5396,#ff832b); color:#fff; }

.bubble {
    padding: 12px 16px;
    border-radius: 16px;
    font-size: .91rem;
    line-height: 1.65;
    max-width: 100%;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble-agent {
    background: #f0f4ff;
    border: 1.5px solid #c7d7ff;
    border-top-left-radius: 4px;
    color: #1a1a3a;
}
.bubble-user {
    background: linear-gradient(135deg,#0f62fe,#6929c4);
    color: #fff;
    border-top-right-radius: 4px;
}

.msg-time {
    font-size: .7rem;
    color: #999;
    margin-top: 4px;
    text-align: right;
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0);   }
}

/* ── typing indicator ── */
.typing { display:flex; gap:5px; align-items:center; padding:6px 0; }
.typing span {
    width:8px; height:8px; border-radius:50%;
    background:#0f62fe;
    animation: bounce 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay:.2s; }
.typing span:nth-child(3) { animation-delay:.4s; }
@keyframes bounce {
    0%,80%,100% { transform:translateY(0); opacity:.6; }
    40%          { transform:translateY(-6px); opacity:1; }
}

/* ── mode badge ── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 10px; border-radius: 999px;
    font-size: .72rem; font-weight: 700;
    border: 1.5px solid;
}

/* ── section cards ── */
.card {
    background: #fff;
    border: 1.5px solid #e5e7f0;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: 0 2px 12px rgba(15,98,254,.06);
}
.card-title {
    font-size: .8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    color: #6929c4; margin: 0 0 10px;
}

/* ── skill pill ── */
.skill-pill {
    display: inline-block;
    background: #ede9fe; color: #5b21b6;
    border: 1px solid #c4b5fd;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: .78rem; font-weight: 500;
    margin: 2px 3px 2px 0;
}
.skill-pill.blue { background:#dbeafe; color:#1d4ed8; border-color:#93c5fd; }
.skill-pill.pink { background:#fce7f3; color:#9d174d; border-color:#f9a8d4; }
.skill-pill.green{ background:#dcfce7; color:#166534; border-color:#86efac; }

/* ── quick-action buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform .12s, box-shadow .12s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(15,98,254,.25) !important;
}
button[data-baseweb="tab"] { font-size: .88rem !important; font-weight: 700 !important; }

/* ── input area ── */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #c7d7ff !important;
    font-size: .9rem !important;
}
.stTextArea textarea:focus { border-color: #0f62fe !important; box-shadow: 0 0 0 3px rgba(15,98,254,.12) !important; }

/* ── download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg,#24a148,#0e8a38) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── pipeline ─────────────────────────────────────────────────────────────────
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
    """Append an agent bubble to session chat."""
    st.session_state.setdefault("chat", []).append(
        {"role": "agent", "text": text, "time": _now()}
    )


def chat_user(text: str):
    """Append a user bubble to session chat."""
    st.session_state.setdefault("chat", []).append(
        {"role": "user", "text": text, "time": _now()}
    )


def render_chat():
    """Render all bubbles in st.session_state['chat']."""
    messages = st.session_state.get("chat", [])
    if not messages:
        return
    html_parts = ['<div class="chat-wrap">']
    for m in messages:
        role = m["role"]
        side = "user" if role == "user" else "agent"
        avatar = "🙋" if role == "user" else "🤖"
        bubble_cls = f"bubble-{side}"
        avatar_cls = f"avatar-{side}"
        # Escape HTML in text but keep line breaks
        safe_text = (
            m["text"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html_parts.append(f"""
        <div class="msg {side}">
            <div class="avatar {avatar_cls}">{avatar}</div>
            <div>
                <div class="bubble {bubble_cls}">{safe_text}</div>
                <div class="msg-time">{m['time']}</div>
            </div>
        </div>""")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


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
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:18px 0 10px">
            <div style="font-size:3rem">🎯</div>
            <div style="font-size:1.15rem;font-weight:700;color:#a78bfa">Interview Coach</div>
            <div style="font-size:.75rem;color:#6b6b9a;margin-top:2px">Powered by IBM Granite</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<p style="font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7c7caa">👤 Your Profile</p>', unsafe_allow_html=True)
        name = st.text_input("Name", placeholder="e.g. Alice Johnson", key="sb_name", label_visibility="collapsed")
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
        st.markdown('<p style="font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7c7caa">⚙️ Settings</p>', unsafe_allow_html=True)
        top_k = st.slider("Context docs (top-k)", 3, 10, 5, key="sb_topk")

        # Pipeline status chips
        if pipeline:
            st.markdown("---")
            st.markdown('<p style="font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7c7caa">📡 System</p>', unsafe_allow_html=True)
            llm_m = pipeline.llm_client.mode
            emb_m = pipeline.embedding_engine.mode
            docs  = pipeline.vector_store.document_count
            st.markdown(
                f"LLM &nbsp; {badge_html(llm_m)}<br>"
                f"Embed &nbsp; {badge_html(emb_m)}<br>"
                f"<span style='font-size:.8rem;color:#888'>📚 {docs} docs indexed</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            for k in ["chat", "prep_result", "guidance_result", "eval_result", "role_info"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.caption("Interview Trainer Agent v1.0  \nIBM Granite · FAISS · watsonx.ai")

    return {
        "name": name.strip(),
        "role": role_key,
        "level": level_key,
        "top_k": top_k,
    }


# ── hero header ───────────────────────────────────────────────────────────────
def render_hero(profile: dict):
    name_display = f", {profile['name']}" if profile["name"] else ""
    role_display = ROLES[profile["role"]].split(" ", 1)[-1]   # strip emoji
    lvl_display  = LEVELS[profile["level"]].split(" ", 1)[-1]

    st.markdown(f"""
    <div class="hero">
        <h1>Hey{name_display}! 👋</h1>
        <p>I'm your AI Interview Coach for <strong>{role_display}</strong> · {lvl_display} &nbsp;—&nbsp;
        ask me anything or pick a mode below.</p>
    </div>
    """, unsafe_allow_html=True)


# ── tab: chat coach ───────────────────────────────────────────────────────────
def tab_chat(pipeline, profile: dict):
    if not profile["name"]:
        st.info("👈 Enter your name in the sidebar to start chatting with your coach.")
        return

    # Greeting on first load
    if "chat" not in st.session_state:
        chat_agent(
            f"Hi {profile['name']}! 👋 I'm your Interview Coach.\n\n"
            f"I'm here to help you prepare for a **{ROLES[profile['role']].split(' ',1)[-1]}** "
            f"role at the **{LEVELS[profile['level']].split(' ',1)[-1]}** level.\n\n"
            "You can:\n"
            "• Ask me any interview question 💡\n"
            "• Type an answer and I'll evaluate it 📊\n"
            "• Ask for a full prep plan 📋\n\n"
            "What would you like to work on today?"
        )

    render_chat()

    st.markdown("---")
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
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        send = st.button("Send ➤", type="primary", use_container_width=True)

    # Quick-action chips
    st.markdown("**Quick actions:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    q1 = qcol1.button("📋 Full Prep Plan",      use_container_width=True)
    q2 = qcol2.button("💡 Tell me a question",  use_container_width=True)
    q3 = qcol3.button("🏆 Evaluate my answer",  use_container_width=True)
    q4 = qcol4.button("🗺️ Role overview",       use_container_width=True)

    # Determine what was triggered
    trigger_text = None
    if send and user_msg.strip():
        trigger_text = user_msg.strip()
    elif q1:
        trigger_text = "Give me a full interview preparation plan."
    elif q2:
        trigger_text = "Give me a challenging interview question for my role and level."
    elif q3:
        trigger_text = (
            "How should I evaluate my answers? Give me an example question and let me answer it."
        )
    elif q4:
        trigger_text = "Give me an overview of the key skills and interview rounds for my role."

    if trigger_text:
        chat_user(trigger_text)

        with st.spinner(""):
            try:
                lower = trigger_text.lower()

                # Route to the right pipeline method
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
                    # General Q → question guidance
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

    st.markdown(f"""
    <div class="chip-row">
        <span class="chip chip-blue">💻 {role_disp}</span>
        <span class="chip chip-pink">🚀 {level_disp}</span>
        <span class="chip chip-green">👤 {profile['name']}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Generate My Prep Plan", type="primary", use_container_width=False):
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

        c1, c2 = st.columns([1, 4])
        with c1:
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
        "what the interviewer is really testing, common mistakes, and follow-ups."
    )

    if not profile["name"]:
        st.info("👈 Enter your name in the sidebar first.")
        return

    # Preset question chips
    st.markdown("**Suggested questions to try:**")
    pc1, pc2, pc3 = st.columns(3)
    p1 = pc1.button("Tell me about yourself",         use_container_width=True)
    p2 = pc2.button("Describe a challenging project",  use_container_width=True)
    p3 = pc3.button("Where do you see yourself in 5y?",use_container_width=True)

    preset = ""
    if p1: preset = "Tell me about yourself."
    if p2: preset = "Describe a challenging project you worked on and what you learned."
    if p3: preset = "Where do you see yourself in 5 years?"

    question = st.text_area(
        "Your interview question",
        value=preset,
        placeholder="Paste or type any interview question here…",
        height=100,
        key="qg_question",
    )

    if st.button("💬 Get Expert Guidance", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
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
        # Render as agent chat bubble style
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:16px;align-items:flex-start">
            <div class="avatar avatar-agent" style="width:40px;height:40px;flex-shrink:0">🤖</div>
            <div class="card" style="flex:1;margin:0">
                <div class="card-title">💡 Guidance from your Coach</div>
                {result['guidance'].replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 RAG Context retrieved", expanded=False):
            st.code(result["context_retrieved"], language=None)


# ── tab: answer evaluator ─────────────────────────────────────────────────────
def tab_evaluator(pipeline, profile: dict):
    st.markdown("### 🏆 Answer Evaluator")
    st.markdown(
        "Type your answer to an interview question — the AI coach will score it "
        "and give you structured feedback with an improved model answer."
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
            height=140, key="ev_q",
        )
    with col2:
        st.markdown("**✍️ Your Answer**")
        answer = st.text_area(
            "answer", label_visibility="collapsed",
            placeholder="Write your answer here — be as natural as you would in a real interview…",
            height=140, key="ev_a",
        )

    if st.button("📊 Evaluate My Answer", type="primary", use_container_width=False):
        if not question.strip() or not answer.strip():
            st.warning("Fill in both the question and your answer.")
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

        # User bubble — their answer
        st.markdown(f"""
        <div class="msg user" style="max-width:80%;margin:12px 0 0 auto">
            <div class="avatar avatar-user">🙋</div>
            <div>
                <div class="bubble bubble-user">{result['candidate_answer']}</div>
                <div class="msg-time">{_now()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agent feedback bubble
        safe_fb = result["feedback"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        st.markdown(f"""
        <div class="msg agent" style="max-width:90%;margin:10px 0">
            <div class="avatar avatar-agent">🤖</div>
            <div>
                <div class="bubble bubble-agent" style="white-space:pre-wrap">{safe_fb}</div>
                <div class="msg-time">{_now()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 RAG Context retrieved", expanded=False):
            st.code(result["context_retrieved"], language=None)


# ── tab: role explorer ────────────────────────────────────────────────────────
def tab_role_explorer(pipeline, profile: dict):
    st.markdown("### 🗺️ Role & Level Explorer")
    st.markdown("Browse skills, interview rounds, key companies, and prep strategy for any role.")

    col1, col2 = st.columns(2)
    with col1:
        role_key = st.selectbox("Role", list(ROLES.keys()), format_func=lambda k: ROLES[k],
                                 key="re_role", index=list(ROLES.keys()).index(profile["role"]))
    with col2:
        level_key = st.selectbox("Level", ["entry","mid","senior"],
                                  format_func=lambda k: LEVELS.get(k, k.title()),
                                  key="re_level",
                                  index=["entry","mid","senior"].index(profile["level"])
                                        if profile["level"] in ["entry","mid","senior"] else 0)

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

        st.markdown(f"## {ROLES[role_key]} — {LEVELS[level_key]}")

        # Skill pills
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="card-title">🛠️ Core Skills</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="skill-pill blue">{s}</span>' for s in li.get("core_skills", []))
            st.markdown(pills + "</div>", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:10px"><div class="card-title">🤝 Soft Skills</div>', unsafe_allow_html=True)
            pills2 = "".join(f'<span class="skill-pill">{s}</span>' for s in li.get("soft_skills", []))
            st.markdown(pills2 + "</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card"><div class="card-title">🔄 Interview Rounds</div>', unsafe_allow_html=True)
            rounds_html = "".join(
                f'<div style="padding:5px 0;border-bottom:1px solid #eee;font-size:.88rem">🔹 {r}</div>'
                for r in li.get("interview_rounds", [])
            )
            st.markdown(rounds_html + "</div>", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:10px"><div class="card-title">🏢 Key Companies</div>', unsafe_allow_html=True)
            cos = "".join(f'<span class="skill-pill pink">{c}</span>' for c in li.get("key_companies", []))
            st.markdown(cos + "</div>", unsafe_allow_html=True)

        if ps:
            st.markdown('<div class="card"><div class="card-title">📅 Preparation Strategy</div>', unsafe_allow_html=True)
            wk = ps.get("timeline_weeks", "?")
            st.markdown(f'<span class="chip chip-amber">⏱ {wk} week plan</span>', unsafe_allow_html=True)

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
    c1.metric("Initialized",  "✅ Yes" if status["initialized"] else "❌ No")
    c2.metric("Docs indexed", status["documents_indexed"])
    c3.metric("Index path",   status["index_path"])

    st.markdown("---")
    ca, cb = st.columns(2)
    ca.markdown(f"**LLM** &nbsp;&nbsp; {badge_html(status['llm_mode'])}", unsafe_allow_html=True)
    cb.markdown(f"**Embedding** &nbsp;&nbsp; {badge_html(status['embedding_mode'])}", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Rebuild Vector Store", type="secondary"):
        with st.spinner("Rebuilding index from raw data…"):
            try:
                pipeline.initialize(force_rebuild=True)
                st.success("✅ Vector store rebuilt.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
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
