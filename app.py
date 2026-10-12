"""
Video Notes - Streamlit UI for the AI Video Assistant pipeline.

Run with:  streamlit run app.py
Place this file next to your existing `utils/` and `core/` folders and .env.
"""
import io
import os
import tempfile
import time
from contextlib import redirect_stdout

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Video Notes",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #1B2A33;
  --muted: #5E6E78;
  --paper: #F4F6F8;
  --card: #FFFFFF;
  --line: #DCE3E8;
  --accent: #0E6B7A;
  --accent-soft: #E2F0F2;
}

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--ink);
}
.stApp { background: var(--paper); }

/* Hide default Streamlit chrome */
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container { padding-top: 2rem; max-width: 1100px; }

h1, h2, h3, .display {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--ink);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--card);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .brand {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.6rem; font-weight: 700; letter-spacing: -0.03em;
  margin-bottom: .1rem;
}
section[data-testid="stSidebar"] .tagline {
  color: var(--muted); font-size: .9rem; margin-bottom: 1.4rem;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--line);
  transition: background .15s, border-color .15s;
}
.stButton > button[kind="primary"] {
  background: var(--accent); border-color: var(--accent); color: #fff;
}
.stButton > button[kind="primary"]:hover {
  background: #0B5560; border-color: #0B5560;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 3px solid var(--accent-soft); outline-offset: 1px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .6rem 0; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border-radius: 14px; border-color: var(--line);
}

/* Chat */
[data-testid="stChatMessage"] {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 14px; padding: .9rem 1rem;
}

/* Page header */
.page-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0 0 .4rem 0; }
.meta { color: var(--muted); font-size: .95rem; margin-bottom: 1.4rem; }

/* Stats */
.stats { display: flex; gap: 2.5rem; margin: .2rem 0 1.6rem 0; flex-wrap: wrap; }
.stat b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:1.5rem; }
.stat span { color: var(--muted); font-size: .85rem; }

/* Empty state */
.hero { padding: 3.5rem 0 1rem 0; }
.hero h1 { font-size: 3rem; line-height: 1.05; max-width: 14ch; margin-bottom: 1rem; }
.hero p { color: var(--muted); font-size: 1.1rem; max-width: 52ch; line-height: 1.55; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2.2rem; }
.step { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.2rem; }
.step b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; }
.step p { font-size: .92rem; margin: .35rem 0 0 0; }
@media (max-width: 760px) {
  .steps { grid-template-columns: 1fr; }
  .hero h1 { font-size: 2.2rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("source_label", "")
st.session_state.setdefault("language", "english")

# Pause between LLM calls to stay under API rate limits (same as the CLI).
CALL_DELAY_SECONDS = 1.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def save_upload(uploaded) -> str:
    """Write an uploaded file to a temp path so process_input() can read it."""
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def run_pipeline_ui(source: str, language: str) -> dict:
    """Same steps as the CLI pipeline, with live progress in the UI."""
    with st.status("Working on your video", expanded=True) as status:
        st.write("Downloading and preparing audio")
        chunks = process_input(source)

        st.write("Transcribing speech")
        transcript = transcribe_all(chunks, language=language)

        st.write("Writing title")
        title = generate_title(transcript)
        time.sleep(CALL_DELAY_SECONDS)

        st.write("Writing summary")
        summary = summarize(transcript)
        time.sleep(CALL_DELAY_SECONDS)

        st.write("Finding action items")
        action_items = extract_action_items(transcript)
        time.sleep(CALL_DELAY_SECONDS)

        st.write("Finding key decisions")
        decisions = extract_key_decisions(transcript)
        time.sleep(CALL_DELAY_SECONDS)

        st.write("Finding open questions")
        questions = extract_questions(transcript)
        time.sleep(CALL_DELAY_SECONDS)

        st.write("Indexing transcript for chat")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Done", state="complete", expanded=False)

    return {
        "transcript": transcript,
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def get_answer(rag_chain, question: str) -> str:
    """
    Your CLI's ask_question() may print the answer instead of returning it.
    This handles both: it uses the return value if there is one, otherwise it
    captures whatever was printed.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        ret = ask_question(rag_chain, question)

    if isinstance(ret, str) and ret.strip():
        return ret.strip()
    if isinstance(ret, dict):
        for key in ("answer", "result", "output", "text"):
            if ret.get(key):
                return str(ret[key]).strip()
    if hasattr(ret, "content") and ret.content:
        return str(ret.content).strip()

    printed = buf.getvalue().strip()
    return printed or "I couldn't find an answer to that in the transcript."


def build_report(result: dict) -> str:
    return (
        f"# {result['title']}\n\n"
        f"## Summary\n{result['summary']}\n\n"
        f"## Action items\n{result['action_items']}\n\n"
        f"## Key decisions\n{result['key_decisions']}\n\n"
        f"## Open questions\n{result['open_questions']}\n\n"
        f"## Transcript\n{result['transcript']}\n"
    )


# --------------------------------------------------------------------------
# Sidebar - input
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Video Notes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Summaries, tasks and answers from any video.</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Video source", ["YouTube link", "Upload a file"], label_visibility="collapsed"
    )

    source, source_label = None, ""
    if input_mode == "YouTube link":
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        if url.strip():
            source, source_label = url.strip(), url.strip()
    else:
        uploaded = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded:
            source_label = uploaded.name
            source = uploaded  # saved to disk only when the user clicks Analyze

    language = st.selectbox("Spoken language", ["english", "hinglish"])

    analyze = st.button(
        "Analyze video", type="primary", use_container_width=True, disabled=source is None
    )

    if st.session_state.result:
        st.divider()
        if st.button("Start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if analyze and source is not None:
    st.session_state.messages = []
    path = save_upload(source) if input_mode == "Upload a file" else source
    try:
        st.session_state.result = run_pipeline_ui(path, language)
        st.session_state.source_label = source_label
        st.session_state.language = language
    except Exception as e:
        st.session_state.result = None
        st.error(
            f"Something went wrong while processing the video.\n\n`{e}`\n\n"
            "Check the link or file, and make sure your API keys are set in .env."
        )

result = st.session_state.result

# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------
if not result:
    st.markdown(
        """
        <div class="hero">
          <h1>Turn any video into notes you can use.</h1>
          <p>Paste a YouTube link or upload a recording. You get a summary,
          action items, decisions and open questions, and you can ask the
          video questions in plain language.</p>
        </div>
        <div class="steps">
          <div class="step"><b>1. Add a video</b>
            <p>A YouTube link, or a video or audio file from your computer.</p></div>
          <div class="step"><b>2. Analyze</b>
            <p>The video is transcribed and read for the points that matter.</p></div>
          <div class="step"><b>3. Ask</b>
            <p>Chat with the transcript and get answers grounded in what was said.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
words = len(result["transcript"].split())
read_min = max(1, round(words / 200))

st.markdown(f'<div class="page-title">{result["title"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="meta">{st.session_state.source_label} &nbsp;|&nbsp; '
    f'{st.session_state.language.capitalize()}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="stats">
      <div class="stat"><b>{words:,}</b><span>words spoken</span></div>
      <div class="stat"><b>{read_min} min</b><span>to read the transcript</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_summary, tab_notes, tab_chat, tab_transcript = st.tabs(
    ["Summary", "Action items and decisions", "Ask the video", "Transcript"]
)

with tab_summary:
    with st.container(border=True):
        st.markdown(result["summary"])

with tab_notes:
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.container(border=True):
            st.subheader("Action items")
            st.markdown(result["action_items"])
    with col_b:
        with st.container(border=True):
            st.subheader("Key decisions")
            st.markdown(result["key_decisions"])
    with st.container(border=True):
        st.subheader("Open questions")
        st.markdown(result["open_questions"])

with tab_chat:
    if not st.session_state.messages:
        st.caption("Ask anything about the video, for example: “What were the main takeaways?”")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about this video")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript"):
                try:
                    answer = get_answer(result["rag_chain"], question)
                except Exception as e:
                    answer = f"I couldn't answer that: `{e}`"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

with tab_transcript:
    st.download_button(
        "Download full report (.md)",
        data=build_report(result),
        file_name="video_notes.md",
        mime="text/markdown",
    )
    with st.container(border=True, height=480):
        st.write(result["transcript"])