import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recallr",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Minimalist Premium CSS Styling ──────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global Container Setup */
    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    
    /* Elegant Custom Status Pipeline Design - Premium Sky Blue Combo */
    .pipeline-container {
        background-color: rgba(0, 191, 255, 0.04);
        border: 1px solid rgba(0, 191, 255, 0.15);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .step-line {
        font-size: 0.88rem;
        padding: 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #888888;
    }
    .step-done { 
        color: #00bfff !important; 
        font-weight: 600; 
    }
    .step-active { 
        color: var(--text-color, #111111) !important; 
        font-weight: 700; 
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    
    /* Branding Elements */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.2rem;
    }
    .hero-accent {
        color: #00bfff;
    }
    .tagline {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    
    /* Custom Override for Primary Button to Sky Blue Theme */
    button[data-testid="stBaseButton-primary"] {
        background-color: #00bfff !important;
        border-color: #00bfff !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #009ad3 !important;
        border-color: #009ad3 !important;
        color: #ffffff !important;
    }
    
    /* Secondary Buttons Styling overrides */
    div.stButton > button {
        border-radius: 6px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Init ─────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "validation_error": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

PIPELINE_STEPS = [
    ("audio", "Audio Ingestion"),
    ("transcript", "Transcription"),
    ("title", "Semantic Titling"),
    ("summary", "Summarisation"),
    ("extract", "Insight Extraction"),
    ("rag", "Vector Context Mapping"),
]

def render_steps():
    for key, label in PIPELINE_STEPS:
        state = st.session_state.pipeline_steps.get(key, "pending")
        if state == "done":
            st.markdown(f'<div class="step-line step-done">● {label}</div>', unsafe_allow_html=True)
        elif state == "active":
            st.markdown(f'<div class="step-line step-active">➔ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-line">○ {label}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h1 style="padding: 0; margin: 0; font-weight: 700;">Recallr<span style="color: #00bfff;">.</span></h1>', unsafe_allow_html=True)

    source = st.text_input("YouTube URL or File Path", placeholder="Paste source link or local path.")
    language = st.selectbox("Language Mapping", ["english", "hinglish"], index=0)
    
    st.markdown("##") 
    run_btn = st.button("Analyse", use_container_width=True, type="primary")

    if st.session_state.pipeline_done or st.session_state.pipeline_steps:
        st.markdown("---")
        st.markdown("**Pipeline Activity**")
        render_steps()

# ── Pipeline Run Handler ───────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.session_state.validation_error = True
        st.session_state.result = None
    else:
        st.session_state.validation_error = False
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("Pipeline is active.You'll get meeting insights shortly.")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("Pipeline activity completed.")
            time.sleep(0.5)
            progress_placeholder.empty()

        except Exception as e:
            for k, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"Execution Error: {e}")

# ── Main Interface Router ──────────────────────────────────────────────────────
if st.session_state.result:
    # ── Dashboard Mode when Ingestion is Complete ──────────────────────────────
    r = st.session_state.result

    with st.container(border=False):
        st.markdown('<div class="hero-title">Recallr<span class="hero-accent">.</span></div>', unsafe_allow_html=True)
        st.subheader(r['title'])

    st.markdown("##")

    # Split: Summary Grid vs Transcripts Window
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("### 📄Summary")
            st.markdown(r['summary'])
    with col2:
        with st.expander("🔍 Review Ingested Transcript Lines", expanded=False):
            st.markdown(r["transcript"])

    st.markdown("##")

    # Strategic 3-Column Artifact Distribution
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("🎯 **Action Items**")
            st.markdown("---")
            st.markdown(r['action_items'])
    with c2:
        with st.container(border=True):
            st.markdown("⚖️ **Key Decisions**")
            st.markdown("---")
            st.markdown(r['key_decisions'])
    with c3:
        with st.container(border=True):
            st.markdown("❓ **Open Questions**")
            st.markdown("---")
            st.markdown(r['open_questions'])

    st.markdown("---")

    # Interactive Context Query Sandbox (RAG)
    st.markdown("### 💬Ask anything about the video")
    st.markdown("##")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Query")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Parsing localized vector index..."):
                answer = ask_question(r["rag_chain"], user_input)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        st.markdown("##")
        if st.button("Reset Synthesis Window", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # ── High Contrast Premium Minimal Empty State / Showcase Home ──────────────
    st.markdown('<div class="hero-title">Recallr<span class="hero-accent">.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Stop taking notes. Start capturing intelligence.</div>', unsafe_allow_html=True)
    
    st.markdown("### What Recallr Does?")
    st.markdown("Recallr converts YouTube videos and local video files into interactive knowledge sources.It generates summary and allow users to ask questions and get context-aware answers from the video content.")
    
    # Switch Layout Action Zone right below the description:
    if st.session_state.validation_error:
        st.error("Please provide a valid YouTube URL or local machine path for analysis.")
    else:
        st.info("💡 **Ready to begin?** " \
        " Insert a valid youtube URL or local file path inside the sidebar on the left to activate the system.")
    
    # "How It Works" Section
    st.markdown("### Tactical Engine Workflow")
    
    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="step-line step-done">● 01. Ingestion Layer</div>
                <small style='color: #888888;'>Fetches and prepares audio from local paths or YouTube URLs instantly.</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="step-line step-done">● 04. Summarisation</div>
                <small style='color: #888888;'>Condenses long conversations into highly structured, actionable outlines.</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with p2:
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="step-line step-done">● 02. Transcription</div>
                <small style='color: #888888;'>Converts spoken audio into high-accuracy text in English.</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="step-line step-done">● 05. Extraction</div>
                <small style='color: #888888;'>Automatically isolates decisions made, action dates, and open milestones.</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with p3:
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="step-line step-done">● 03. Semantic Titling</div>
                <small style='color: #888888;'>Generates a context-aware semantic header for organized record-keeping.</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="pipeline-container">
                <div class="step-line step-done">● 06. Vector Indexing</div>
                <small style='color: #888888;'>Builds a secure local database index to power accurate instant-answer Q&A.</small>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # 3-Column Core Features Grid at the bottom base
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        with st.container(border=True):
            st.markdown("⚙️ **Smart Extraction**")
            st.caption("Automatically separates structural decisions, open milestones, and core actionable items from raw audio.")
    with f2:
        with st.container(border=True):
            st.markdown("📊 **Summaries**")
            st.caption("Turns long, unstructured recordings into crisp, scannable summaries in seconds.")
    with f3:
        with st.container(border=True):
            st.markdown("💬 **Conversational RAG**")
            st.caption("Ask questions directly to your transcript to recall exact timelines, quotes, or details.")