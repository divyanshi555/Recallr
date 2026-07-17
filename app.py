# All the essential imports
import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from utils.cleanup import clean_workspace

load_dotenv()

# App configurations
st.set_page_config(
    page_title="Recallr",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
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
    .step-done { color: #00bfff !important; font-weight: 600; }
    .step-active {
        color: var(--text-color, #111111) !important;
        font-weight: 700;
        animation: pulse 1.5s infinite ease-in-out;
    }
    .step-pending {
        color: #999999;   
        font-weight: 300; 
    }
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.2rem;
    }
    .hero-accent { color: #00bfff; }
    .tagline {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 2rem;
    }
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
    div.stButton > button {
        border-radius: 6px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def default_state():
    return {
        "workspace_initialized": False,
        "result": None,
        "rag_chain": None,
        "chat_history": [],
        "pipeline_done": False,
        "pipeline_steps": {},
        "validation_error": False,
    }


# Setting the default session state
for _key, _default in default_state().items():
    st.session_state.setdefault(_key, _default)

# Clearing the workspace to remove stale data if any
if not st.session_state.workspace_initialized:
    clean_workspace()
    st.session_state.workspace_initialized = True

# Pipeline steps that are going to be displayed on sidebar when it is running
PIPELINE_STEPS = [
    ("audio", "Audio Ingestion", "Fetches and prepares audio from local paths or YouTube URLs instantly."),
    ("transcript", "Transcription", "Converts spoken audio into high-accuracy text."),
    ("title", "Semantic Titling", "Generates a context-aware semantic header for organized record-keeping."),
    ("summary", "Summarisation", "Condenses long conversations into highly structured, actionable outlines."),
    ("extract", "Insight Extraction", "Automatically isolates decisions made, action dates, and open milestones."),
    ("rag", "Vector Context Mapping", "Builds a secure local database index to power accurate instant-answer Q&A."),
]

STATE_ICON = {"done": "●", "active": "➔", "pending": "○"}
STATE_CLASS = {"done": "step-done", "active": "step-active", "pending": "step-pending"}


# Function: It renders the Pipeline steps on the sidebar
def render_pipeline_status():
    for key, label, _ in PIPELINE_STEPS:
        state = st.session_state.pipeline_steps.get(key, "pending")
        st.markdown(
            f'<div class="step-line {STATE_CLASS[state]}">{STATE_ICON[state]} {label}</div>',
            unsafe_allow_html=True,
        )


# Function to set each step of pipeline
def set_step(key, state):
    st.session_state.pipeline_steps[key] = state


# Function: It resets the session when the user clicks Clear Session
def reset_session():
    for key, default in default_state().items():
        if key != "workspace_initialized":
            st.session_state[key] = default


# Function to display Title and content of action-items, open-questions and key-decisions
def info_card(title, content):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(content)


# Function: To activate the RAG pipeline
def run_pipeline(source, language):
    st.session_state.pipeline_done = False
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.pipeline_steps = {}

    progress = st.empty()
    success = False

    try:
        with progress.container():
            st.info("Pipeline is active. You'll get the video's insights shortly.")

        set_step("audio", "active")
        chunks = process_input(source)
        set_step("audio", "done")

        set_step("transcript", "active")
        transcript = transcribe_all(chunks, language)
        set_step("transcript", "done")

        set_step("title", "active")
        title = generate_title(transcript)
        set_step("title", "done")

        set_step("summary", "active")
        summary = summarize(transcript)
        set_step("summary", "done")

        set_step("extract", "active")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        set_step("extract", "done")

        set_step("rag", "active")
        rag_chain = build_rag_chain(transcript)
        set_step("rag", "done")

        st.session_state.result = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
        }
        st.session_state.rag_chain = rag_chain
        st.session_state.pipeline_done = True

        progress.success("Pipeline activity completed.")
        time.sleep(0.5)
        progress.empty()
        success = True

    except Exception as e:
        for key, _, _ in PIPELINE_STEPS:
            if st.session_state.pipeline_steps.get(key) == "active":
                st.session_state.pipeline_steps[key] = "pending"
        progress.error(f"Execution Error: {e}")

    if success:
        st.rerun()


# Sidebar that takes File-path or url as input and shows analyze button that activates the RAG Pipeline
# After the Pipeline completes, it shows the Pipeline activity and Clear Session button to reset the current session.
with st.sidebar:
    st.markdown(
        '<h1 style="padding:0;margin:0;font-weight:700;">Recallr<span style="color:#00bfff;">.</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if st.session_state.pipeline_done:
        st.markdown("**Pipeline Activity**")
        render_pipeline_status()
        st.markdown("---")
        if st.button("Clear Session", use_container_width=True, type="primary"):
            st.toast("Clearing current session")
            st.session_state.rag_chain = None
            clean_workspace()
            reset_session()
            time.sleep(0.5)
            st.rerun()
    else:
        source = st.text_input("YouTube URL or File Path", placeholder="Paste source link or local file path.")
        language = st.selectbox("Choose Language", ["english", "hinglish"], index=0)
        st.markdown("##")
        run_clicked = st.button("Analyse", use_container_width=True, type="primary")

        if st.session_state.pipeline_steps:
            st.markdown("---")
            st.markdown("**Pipeline Activity**")
            render_pipeline_status()

        if run_clicked:
            if not source.strip():
                st.session_state.validation_error = True
                st.session_state.result = None
            else:
                st.session_state.validation_error = False
                run_pipeline(source, language)

# Main Area
# 1) Before the Pipeline activation, it shows the Home screen
# 2) After the Pipeline completes, it shows the summary, Original Transcript, action-items, Key decisions, Open questions and Chatbot that answers user queries

if st.session_state.result:
    r = st.session_state.result

    st.markdown('<div class="hero-title">Recallr<span class="hero-accent">.</span></div>', unsafe_allow_html=True)
    st.subheader(r["title"])
    st.markdown("##")

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("### 📄 Summary")
            st.markdown(r["summary"])
    with col2:
        with st.expander("🔍 Review Ingested Transcript Lines", expanded=False):
            st.markdown(r["transcript"])

    st.markdown("##")
    for col, (title, content) in zip(
        st.columns(3, gap="medium"),
        [
            ("🎯 Action Items", r["action_items"]),
            ("⚖️ Key Decisions", r["key_decisions"]),
            ("❓ Open Questions", r["open_questions"]),
        ],
    ):
        with col:
            info_card(title, content)

    st.markdown("---")
    st.markdown("### 💬 Ask anything about the video")
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
            with st.spinner("Searching..."):
                answer = ask_question(st.session_state.rag_chain, user_input)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        st.markdown("##")
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown('<div class="hero-title">Recallr<span class="hero-accent">.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Stop taking notes. Start capturing intelligence.</div>', unsafe_allow_html=True)

    st.markdown("### What Recallr Does?")
    st.markdown(
        "Recallr converts YouTube videos and local video files into interactive knowledge sources. "
        "It generates summaries and allows users to ask questions and get context-aware answers from the video content."
    )

    if st.session_state.validation_error:
        st.error("Please provide a valid YouTube URL or local machine path for analysis.")
    else:
        st.info("💡 **Ready to begin?** Insert a valid YouTube URL or local file path inside the sidebar on the left to activate the system.")

    st.markdown("### Tactical Engine Workflow")
    workflow_cols = st.columns(3, gap="medium")
    for i, (_, label, desc) in enumerate(PIPELINE_STEPS):
        with workflow_cols[i % 3]:
            st.markdown(
                f"""
                <div class="pipeline-container">
                    <div class="step-line step-done">● {i + 1:02d}. {label}</div>
                    <small style='color:#888888;'>{desc}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    for col, (title, caption) in zip(
        st.columns(3, gap="medium"),
        [
            ("⚙️ **Smart Extraction**", "Automatically separates structural decisions, open milestones, and core actionable items from raw audio."),
            ("📊 **Summaries**", "Turns long, unstructured recordings into crisp, scannable summaries in seconds."),
            ("💬 **Conversational RAG**", "Ask questions directly to your transcript to recall exact timelines, quotes, or details."),
        ],
    ):
        with col:
            with st.container(border=True):
                st.markdown(title)
                st.caption(caption)