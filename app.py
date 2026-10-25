"""
Video Notes - Streamlit UI for the AI Video Assistant pipeline.

Run with:  streamlit run app.py
Place this file next to your existing `utils/` and `core/` folders and .env.
"""
import io
import os
import shutil
import tempfile
import time
import uuid
from contextlib import redirect_stdout

from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all, resolve_language_mode
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    extract_all_insights,
    extract_structured_action_items,
    extract_structured_decisions,
    ActionItem,
    MeetingDecision,
    MeetingIntelligence,
)
from core.report_generator import (
    generate_pdf_report,
    generate_markdown_report,
)
from core.rag_engine import build_rag_chain, ask_question
from core.audio_recorder import (
    AudioRecorder,
    AudioSource,
    RecordingState,
    query_available_devices,
)
from core.live_transcriber import LiveTranscriber, LiveAudioChunker
from core.meeting_qa import (
    ask_live_meeting_question,
    ask_post_meeting_question,
    AnswerState,
    MeetingQAResponse,
    format_timestamp,
)

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
st.session_state.setdefault("language", "English transcription")
st.session_state.setdefault("session_id", "")
st.session_state.setdefault("recorder", None)
st.session_state.setdefault("recorded_audio_path", None)
st.session_state.setdefault("live_transcriber", None)
st.session_state.setdefault("live_transcript_text", "")
st.session_state.setdefault("live_messages", [])
if st.session_state.recorder is None:
    st.session_state.recorder = AudioRecorder()

# Supported language options matching requirements
LANGUAGE_OPTIONS = [
    "English transcription",
    "Hinglish transcription using the existing Sarvam integration",
    "Nepali transcription",
    "Nepali speech → English text",
    "Automatic language detection",
]

# Pause between LLM calls to stay under API rate limits (same as the CLI).
CALL_DELAY_SECONDS = 1.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def save_upload(uploaded) -> str:
    """Write an uploaded file to a unique temp directory so process_input() can read it without collisions."""
    tmp_dir = tempfile.mkdtemp(prefix="video_assistant_")
    clean_name = os.path.basename(uploaded.name)
    target = os.path.join(tmp_dir, clean_name)
    with open(target, "wb") as f:
        f.write(uploaded.getbuffer())
    return target


def run_pipeline_ui(source: str, language: str, keep_original_nepali: bool = False, session_id: str = None) -> dict:
    """Same steps as the CLI pipeline, with live progress in the UI and stage-by-stage timing metrics."""
    if not session_id:
        source_base = os.path.splitext(os.path.basename(source))[0][:16].replace(" ", "_")
        session_id = f"{source_base}_{uuid.uuid4().hex[:6]}" if source_base else f"ui_{uuid.uuid4().hex[:8]}"

    mode_info = resolve_language_mode(language)
    is_nepali_translate = (mode_info.get("whisper_code") == "ne" and mode_info.get("task") == "translate")

    stage_timings = {}
    t_pipeline_start = time.perf_counter()

    with st.status("Working on your video", expanded=True) as status:
        progress_bar = st.progress(0.0, text="Starting processing...")

        st.write("📥 Downloading and preparing audio...")
        t0 = time.perf_counter()
        chunks = process_input(source)
        stage_timings["Audio Preparation"] = time.perf_counter() - t0
        st.write(f"✓ Audio prepared into {len(chunks)} chunk(s) ({stage_timings['Audio Preparation']:.2f}s)")
        progress_bar.progress(0.15, text="Audio prepared. Starting transcription...")

        t0 = time.perf_counter()
        def on_transcribe_chunk(curr, total, msg):
            pct = 0.15 + 0.50 * (curr / max(1, total))
            progress_bar.progress(min(pct, 0.65), text=f"Transcribing speech: chunk {curr}/{total}...")
            st.write(f"  • Transcribing chunk {curr}/{total}...")

        if is_nepali_translate:
            st.write("🎙️ Translating Nepali speech to English...")
            if keep_original_nepali:
                st.write("🎙️ Transcribing original Nepali speech (Devanagari)...")
            res = transcribe_all(chunks, language=language, return_original=keep_original_nepali, progress_callback=on_transcribe_chunk)
        else:
            st.write(f"🎙️ Transcribing speech ({mode_info['display_name']})...")
            res = transcribe_all(chunks, language=language, return_original=False, progress_callback=on_transcribe_chunk)

        stage_timings["Speech Transcription"] = time.perf_counter() - t0
        st.write(f"✓ Transcription complete ({stage_timings['Speech Transcription']:.2f}s)")
        progress_bar.progress(0.65, text="Transcription finished. Generating summary & notes...")

        if isinstance(res, dict):
            transcript = res.get("transcript", "")
            original_transcript = res.get("original_transcript")
            segments = res.get("segments")
        else:
            transcript = str(res)
            original_transcript = None
            segments = None

        st.write("🧠 Generating insights in parallel (Title, Summary, Action Items, Decisions, Questions)...")
        t_llm = time.perf_counter()

        def on_llm_progress(curr, total, name):
            pct = 0.65 + 0.30 * (curr / max(1, total))
            progress_bar.progress(min(pct, 0.95), text=f"Analyzing transcript: {name} completed ({curr}/{total})...")
            st.write(f"  • {name} completed")

        llm_res = extract_all_insights(
            transcript,
            progress_callback=on_llm_progress,
        )
        stage_timings.update(llm_res["stage_timings"])
        stage_timings["LLM Analysis (Parallel)"] = time.perf_counter() - t_llm

        title = llm_res["title"]
        summary = llm_res["summary"]
        action_items = llm_res["action_items"]
        decisions = llm_res["key_decisions"]
        questions = llm_res["open_questions"]

        st.write(f"🔍 Indexing transcript for chat (session: {session_id})...")
        progress_bar.progress(0.95, text=f"Indexing transcript into ChromaDB ({session_id})...")
        t0 = time.perf_counter()
        rag_chain = build_rag_chain(transcript, session_id=session_id, segments=segments)
        stage_timings["ChromaDB Indexing"] = time.perf_counter() - t0

        t_total = time.perf_counter() - t_pipeline_start
        stage_timings["Total Pipeline Time"] = t_total
        progress_bar.progress(1.0, text=f"Done in {t_total:.1f}s!")
        if status is not None and hasattr(status, "update"):
            status.update(label=f"Done in {t_total:.1f}s", state="complete", expanded=False)

        st.write("🎯 Extracting structured action items and decisions with citations...")
        structured_actions = extract_structured_action_items(transcript, segments=segments)
        structured_decisions = extract_structured_decisions(transcript, segments=segments)

        meta_info = {
            "duration": f"{t_total:.1f}s",
            "language": mode_info.get("display_name", language),
            "audio_source": "Video/Audio Source",
            "session_id": session_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        try:
            pdf_bytes = generate_pdf_report({
                "title": title,
                "summary": summary,
                "action_items": structured_actions,
                "key_decisions": structured_decisions,
                "open_questions": [q.strip() for q in questions.split("\n") if q.strip()],
                "transcript": transcript,
                "segments": segments,
                "metadata": meta_info,
            })
        except Exception as e:
            print(f"[!] PDF generation warning: {e}")
            pdf_bytes = None

    return {
        "transcript": transcript,
        "original_transcript": original_transcript,
        "segments": segments,
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "structured_action_items": structured_actions,
        "structured_key_decisions": structured_decisions,
        "rag_chain": rag_chain,
        "session_id": session_id,
        "display_language": mode_info["display_name"],
        "is_translated": is_nepali_translate,
        "stage_timings": stage_timings,
        "pdf_bytes": pdf_bytes,
    }


def process_live_intelligence_ui(
    transcript: str,
    segments: list,
    language: str,
    audio_source: str = "Live Recording",
    duration_seconds: float = 0.0,
    session_id: str = None,
    dropped_chunks_count: int = 0,
    dropped_audio_seconds: float = 0.0,
) -> dict:
    """Generate meeting intelligence directly from finalized live transcript segments."""
    if not session_id:
        session_id = f"live_{uuid.uuid4().hex[:8]}"

    mode_info = resolve_language_mode(language)
    stage_timings = {}
    t_start = time.perf_counter()

    with st.status("Analyzing meeting transcript...", expanded=True) as status:
        st.write("🧠 Generating insights in parallel (Title, Summary, Action Items, Decisions, Questions)...")
        t_llm = time.perf_counter()
        llm_res = extract_all_insights(transcript)
        stage_timings.update(llm_res.get("stage_timings", {}))
        stage_timings["LLM Analysis (Parallel)"] = time.perf_counter() - t_llm

        st.write("🎯 Extracting structured action items and decisions with citations...")
        structured_actions = extract_structured_action_items(transcript, segments=segments, max_duration=duration_seconds)
        structured_decisions = extract_structured_decisions(transcript, segments=segments, max_duration=duration_seconds)

        st.write(f"🔍 Indexing transcript for grounded Q&A (session: {session_id})...")
        t_rag = time.perf_counter()
        rag_chain = build_rag_chain(transcript, session_id=session_id, segments=segments)
        stage_timings["ChromaDB Indexing"] = time.perf_counter() - t_rag

        t_total = time.perf_counter() - t_start
        stage_timings["Total Pipeline Time"] = t_total
        if status is not None and hasattr(status, "update"):
            status.update(label=f"Done in {t_total:.1f}s", state="complete", expanded=False)

    title = llm_res.get("title", "Meeting Notes")
    summary = llm_res.get("summary", "")
    action_items = llm_res.get("action_items", "")
    decisions = llm_res.get("key_decisions", "")
    questions = llm_res.get("open_questions", "")

    meta_info = {
        "duration": f"{duration_seconds:.1f}s" if duration_seconds else "N/A",
        "language": mode_info.get("display_name", language),
        "audio_source": audio_source,
        "session_id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    try:
        pdf_bytes = generate_pdf_report({
            "title": title,
            "summary": summary,
            "action_items": structured_actions,
            "key_decisions": structured_decisions,
            "open_questions": [q.strip() for q in questions.split("\n") if q.strip()],
            "transcript": transcript,
            "segments": segments,
            "metadata": meta_info,
        })
    except Exception as e:
        print(f"[!] PDF generation warning: {e}")
        pdf_bytes = None

    return {
        "transcript": transcript,
        "original_transcript": None,
        "segments": segments,
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "structured_action_items": structured_actions,
        "structured_key_decisions": structured_decisions,
        "rag_chain": rag_chain,
        "session_id": session_id,
        "display_language": mode_info.get("display_name", language),
        "is_translated": False,
        "stage_timings": stage_timings,
        "duration_seconds": duration_seconds,
        "dropped_chunks_count": dropped_chunks_count,
        "dropped_audio_seconds": dropped_audio_seconds,
        "pdf_bytes": pdf_bytes,
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
    lang_info = result.get("display_language", "Transcript")
    report = (
        f"# {result['title']}\n\n"
        f"**Language Mode**: {lang_info}\n\n"
        f"## Summary\n{result['summary']}\n\n"
        f"## Action items\n{result['action_items']}\n\n"
        f"## Key decisions\n{result['key_decisions']}\n\n"
        f"## Open questions\n{result['open_questions']}\n\n"
        f"## Transcript\n{result['transcript']}\n"
    )
    if result.get("original_transcript"):
        report += f"\n## Original Nepali Transcript (Devanagari)\n{result['original_transcript']}\n"
    return report


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
        "Video source",
        ["YouTube link", "Upload a file", "Live recording"],
        label_visibility="collapsed",
    )

    default_idx = 0
    if st.session_state.language in LANGUAGE_OPTIONS:
        default_idx = LANGUAGE_OPTIONS.index(st.session_state.language)

    language = st.selectbox(
        "Spoken language",
        LANGUAGE_OPTIONS,
        index=default_idx,
    )
    st.session_state.language = language

    keep_original_nepali = False
    if language == "Nepali speech → English text":
        keep_original_nepali = st.checkbox(
            "Keep original Devanagari transcript alongside English translation",
            value=True,
            help="Transcribes original Nepali speech in Devanagari script in addition to the English translation.",
        )

    source, source_label = None, ""
    if input_mode == "YouTube link":
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        if url.strip():
            source, source_label = url.strip(), url.strip()
    elif input_mode == "Upload a file":
        uploaded = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded:
            source_label = uploaded.name
            source = uploaded  # saved to disk only when the user clicks Analyze
    else:
        # Live audio recording controls (Microphone + System Loopback)
        rec = st.session_state.recorder
        rec_state = rec.state
        elapsed = rec.get_elapsed_seconds()

        audio_mode_choice = st.selectbox(
            "Audio capture source",
            ["Microphone + Meeting Audio", "Microphone only", "Meeting Audio only (System Loopback)"],
            index=0,
            disabled=(rec_state in (RecordingState.RECORDING, RecordingState.PAUSED)),
            help="Captures your microphone voice and Windows WASAPI system meeting audio simultaneously.",
        )
        source_map = {
            "Microphone + Meeting Audio": AudioSource.BOTH,
            "Microphone only": AudioSource.MICROPHONE,
            "Meeting Audio only (System Loopback)": AudioSource.SYSTEM,
        }
        chosen_source = source_map[audio_mode_choice]

        if rec_state == RecordingState.IDLE:
            st.info("⚪ Ready to record. Click **Start recording** to begin capturing.")
        elif rec_state == RecordingState.RECORDING:
            st.error(f"🔴 **Recording in progress...** ({elapsed:.1f}s)")
        elif rec_state == RecordingState.PAUSED:
            st.warning(f"⏸️ **Recording paused** ({elapsed:.1f}s)")
        elif rec_state == RecordingState.STOPPED:
            st.success(f"⏹️ **Recording finished** ({elapsed:.1f}s)")

        col_r1, col_r2 = st.columns(2)
        if rec_state == RecordingState.IDLE:
            if col_r1.button("Start recording", type="primary", use_container_width=True):
                try:
                    # Initialize live transcriber & chunker with selected language
                    live_trans = LiveTranscriber(language=language)
                    chunker = LiveAudioChunker(live_trans)
                    rec.attach_live_chunker(chunker)
                    live_trans.start()
                    st.session_state.live_transcriber = live_trans
                    rec.start(source=chosen_source)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start recording: {e}")
        elif rec_state == RecordingState.RECORDING:
            if col_r1.button("Pause", use_container_width=True):
                rec.pause()
                if st.session_state.get("live_transcriber"):
                    st.session_state.live_transcriber.pause()
                st.rerun()
            if col_r2.button("Stop", type="primary", use_container_width=True):
                try:
                    wav_file = rec.stop()
                    st.session_state.recorded_audio_path = wav_file
                    if st.session_state.get("live_transcriber"):
                        final_live_text = st.session_state.live_transcriber.stop()
                        st.session_state.live_transcript_text = final_live_text
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop recording: {e}")
        elif rec_state == RecordingState.PAUSED:
            if col_r1.button("Resume", type="primary", use_container_width=True):
                rec.resume()
                if st.session_state.get("live_transcriber"):
                    st.session_state.live_transcriber.resume()
                st.rerun()
            if col_r2.button("Stop", use_container_width=True):
                try:
                    wav_file = rec.stop()
                    st.session_state.recorded_audio_path = wav_file
                    if st.session_state.get("live_transcriber"):
                        final_live_text = st.session_state.live_transcriber.stop()
                        st.session_state.live_transcript_text = final_live_text
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop recording: {e}")
        elif rec_state == RecordingState.STOPPED:
            if col_r1.button("New recording", use_container_width=True):
                rec.reset()
                if st.session_state.get("live_transcriber"):
                    st.session_state.live_transcriber.reset()
                st.session_state.live_transcriber = None
                st.session_state.live_transcript_text = ""
                st.session_state.live_messages = []
                st.session_state.recorded_audio_path = None
                st.session_state.result = None
                st.rerun()

            btn_gen_notes = col_r2.button("Generate Notes", type="primary", use_container_width=True)

            retranscribe_offline = st.checkbox(
                "Re-transcribe with offline Whisper from scratch",
                value=False,
                help="By default, meeting intelligence reuses finalized live transcript segments. Check this if you want an offline second pass.",
            )

            if btn_gen_notes:
                # Flush pending transcription before generating meeting notes
                if st.session_state.get("live_transcriber"):
                    final_live_text = st.session_state.live_transcriber.stop()
                    st.session_state.live_transcript_text = final_live_text

                if not retranscribe_offline and st.session_state.get("live_transcriber") and st.session_state.live_transcript_text:
                    snap = st.session_state.live_transcriber.transcript.get_snapshot()
                    sess_id = f"live_{uuid.uuid4().hex[:8]}"
                    st.session_state.result = process_live_intelligence_ui(
                        transcript=snap.get("full_text", snap.get("text", "")),
                        segments=snap.get("segments", []),
                        language=language,
                        audio_source=f"Live Recording ({rec.source.value})",
                        duration_seconds=rec.get_elapsed_seconds(),
                        session_id=sess_id,
                        dropped_chunks_count=snap.get("dropped_chunks_count", 0),
                        dropped_audio_seconds=snap.get("dropped_audio_seconds", 0.0),
                    )
                    st.session_state.source_label = f"Live Recording ({rec.source.value})"
                    st.session_state.language = language
                    st.session_state.session_id = sess_id
                    st.rerun()
                elif st.session_state.recorded_audio_path:
                    source = st.session_state.recorded_audio_path
                    source_label = f"Live Recording ({rec.source.value})"
                    analyze = True

        # Display live transcript streaming status & live Q&A during recording
        if st.session_state.get("live_transcriber"):
            snap = st.session_state.live_transcriber.transcript.get_snapshot()
            if snap["segments"]:
                with st.expander(f"📝 Live Transcript ({snap['segment_count']} segment{'s' if snap['segment_count'] != 1 else ''})", expanded=True):
                    for s in snap["segments"]:
                        st.caption(f"[{format_timestamp(s['start'])} - {format_timestamp(s['end'])}]")
                        st.write(s["text"])

                with st.expander("💬 Ask Live Meeting Assistant", expanded=(len(st.session_state.live_messages) > 0)):
                    st.caption("Ask questions about what has been spoken so far.")
                    for m in st.session_state.live_messages:
                        with st.chat_message(m["role"]):
                            st.markdown(m["content"])

                    live_q = st.text_input("Question for active meeting", key="live_q_input", placeholder="e.g. What action items were just mentioned?")
                    if st.button("Ask Assistant", key="btn_ask_live"):
                        if live_q.strip():
                            st.session_state.live_messages.append({"role": "user", "content": live_q.strip()})
                            with st.spinner("Checking live transcript..."):
                                resp = ask_live_meeting_question(
                                    transcript_snapshot=snap,
                                    question=live_q.strip(),
                                    chat_history=st.session_state.live_messages[:-1],
                                    session_id=st.session_state.get("session_id", ""),
                                )
                                badge = f"⏱️ *As of {resp.cutoff_timestamp} — Status: {resp.state.value.replace('_', ' ').title()}*\n\n"
                                full_resp_text = badge + resp.answer
                                if resp.citations:
                                    full_resp_text += "\n\n**Evidence:**"
                                    for c in resp.citations:
                                        full_resp_text += f"\n- `{c.timestamp_str}`: *\"{c.excerpt}\"*"
                                st.session_state.live_messages.append({"role": "assistant", "content": full_resp_text})
                            st.rerun()

        if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
            st.caption("🎧 Recording preview:")
            st.audio(st.session_state.recorded_audio_path)
            source = st.session_state.recorded_audio_path
            source_label = f"Live Recording ({rec.source.value})"
    analyze = st.button(
        "Analyze video", type="primary", use_container_width=True, disabled=source is None
    )

    if st.session_state.result:
        st.divider()
        if st.button("Start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.session_state.session_id = ""
            if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                try:
                    os.remove(st.session_state.recorded_audio_path)
                except Exception:
                    pass
            st.session_state.recorded_audio_path = None
            if st.session_state.recorder:
                st.session_state.recorder.reset()
            if st.session_state.get("live_transcriber"):
                st.session_state.live_transcriber.reset()
            st.session_state.live_transcriber = None
            st.session_state.live_transcript_text = ""
            st.session_state.live_messages = []
            st.rerun()

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if analyze and source is not None:
    st.session_state.messages = []
    is_uploaded_file = (input_mode == "Upload a file")
    temp_dir = None
    if is_uploaded_file:
        path = save_upload(source)
        temp_dir = os.path.dirname(path)
    else:
        path = source

    try:
        st.session_state.result = run_pipeline_ui(
            path,
            language,
            keep_original_nepali=keep_original_nepali,
        )
        st.session_state.source_label = source_label
        st.session_state.language = language
        st.session_state.session_id = st.session_state.result.get("session_id", "")
    except Exception as e:
        st.session_state.result = None
        st.error(
            f"Something went wrong while processing the video.\n\n`{e}`\n\n"
            "Check the link or file, and make sure your API keys are set in .env."
        )
    finally:
        # Reliable cleanup: remove temporary upload directory and generated audio artifacts
        if is_uploaded_file and temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

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
def render_results(result: dict):
    """Render metrics, summary, action items, decisions, chat, and export tabs."""
    words = len(result["transcript"].split())
    read_min = max(1, round(words / 200))

    st.markdown(f'<div class="page-title">{result["title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="meta">{st.session_state.source_label} &nbsp;|&nbsp; '
        f'{result.get("display_language", st.session_state.language)}</div>',
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

    if "stage_timings" in result and result["stage_timings"]:
        st_timings = result["stage_timings"]
        total_time = st_timings.get("Total Pipeline Time", 0.0)
        transcribe_time = st_timings.get("Speech Transcription", 0.0)
        pct_transcribe = (transcribe_time / total_time * 100) if total_time > 0 else 0
        with st.expander(f"⏱️ Pipeline Performance & Timings (Total: {total_time:.1f}s — Transcription: {pct_transcribe:.0f}%)", expanded=False):
            t_cols = st.columns(4)
            t_cols[0].metric("Audio Prep", f"{st_timings.get('Audio Preparation', 0.0):.2f}s")
            t_cols[1].metric("Transcription", f"{transcribe_time:.2f}s", delta=f"{pct_transcribe:.0f}% of total" if total_time else None, delta_color="inverse")
            llm_total = sum(st_timings.get(k, 0.0) for k in ["Title Generation", "Summary Generation", "Action Items", "Key Decisions", "Open Questions"])
            t_cols[2].metric("LLM Analysis", f"{llm_total:.2f}s")
            t_cols[3].metric("Vector Store", f"{st_timings.get('ChromaDB Indexing', 0.0):.2f}s")

    tab_transcript, tab_summary, tab_notes, tab_chat = st.tabs(
        ["Transcript", "Summary", "Action items and decisions", "Ask the video"]
    )

    with tab_transcript:
        if result.get("dropped_chunks_count", 0) > 0:
            st.warning(
                f"⚠️ Note: {result['dropped_chunks_count']} audio chunk(s) "
                f"({result.get('dropped_audio_seconds', 0.0):.1f}s) were dropped during live recording "
                "due to high system load. Some speech may be missing."
            )

        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
        with col_dl1:
            # 1. Executive PDF Download
            pdf_bytes = result.get("pdf_bytes")
            if not pdf_bytes:
                try:
                    pdf_bytes = generate_pdf_report({
                        "title": result.get("title", ""),
                        "summary": result.get("summary", ""),
                        "action_items": result.get("structured_action_items", result.get("action_items", "")),
                        "key_decisions": result.get("structured_key_decisions", result.get("key_decisions", "")),
                        "open_questions": result.get("open_questions", []),
                        "transcript": result.get("transcript", ""),
                        "segments": result.get("segments", []),
                        "metadata": {
                            "duration": f"{result.get('duration_seconds', 0.0):.1f}s" if result.get("duration_seconds") else "N/A",
                            "language": result.get("display_language", "English"),
                            "audio_source": st.session_state.get("source_label", "Recording"),
                            "session_id": result.get("session_id", ""),
                        }
                    })
                except Exception as e:
                    print(f"[!] PDF generation error: {e}")
                    pdf_bytes = b"%PDF-1.4\n%Fallback PDF\n"

            st.download_button(
                "Download executive report (.pdf)",
                data=pdf_bytes,
                file_name="meeting_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col_dl2:
            # 2. Markdown Download
            md_content = generate_markdown_report({
                "title": result.get("title", ""),
                "summary": result.get("summary", ""),
                "action_items": result.get("structured_action_items", result.get("action_items", "")),
                "key_decisions": result.get("structured_key_decisions", result.get("key_decisions", "")),
                "open_questions": result.get("open_questions", []),
                "transcript": result.get("transcript", ""),
                "segments": result.get("segments", []),
                "original_transcript": result.get("original_transcript"),
                "metadata": {
                    "duration": f"{result.get('duration_seconds', 0.0):.1f}s" if result.get("duration_seconds") else "N/A",
                    "language": result.get("display_language", "English"),
                    "audio_source": st.session_state.get("source_label", "Recording"),
                    "session_id": result.get("session_id", ""),
                }
            })
            st.download_button(
                "Download full report (.md)",
                data=md_content,
                file_name="meeting_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl3:
            # 3. Transcript .txt download button
            if result.get("is_translated") or result.get("display_language") == "English":
                txt_label = "Download transcript (.txt)"
                txt_filename = "english_transcript.txt"
            else:
                txt_label = "Download transcript (.txt)"
                txt_filename = "transcript.txt"

            st.download_button(
                txt_label,
                data=result["transcript"],
                file_name=txt_filename,
                mime="text/plain",
                use_container_width=True,
            )

        # If original Nepali transcript was also generated, display it separately
        if result.get("original_transcript"):
            with st.expander("Original Nepali Transcript (नेपाली - Devanagari)", expanded=False):
                st.download_button(
                    "Download original Nepali transcript (.txt)",
                    data=result["original_transcript"],
                    file_name="nepali_original_transcript.txt",
                    mime="text/plain",
                )
                st.write(result["original_transcript"])

        with st.container(border=True, height=480):
            segments = result.get("segments")
            if segments:
                for seg in segments:
                    t_start = format_timestamp(seg.get("start", 0.0))
                    t_end = format_timestamp(seg.get("end", 0.0))
                    seg_text = seg.get("text", "").strip()
                    if seg_text:
                        st.markdown(f"**`[{t_start} - {t_end}]`** {seg_text}")
            else:
                st.write(result["transcript"])

    with tab_summary:
        with st.container(border=True):
            st.markdown(result["summary"])

    with tab_notes:
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            with st.container(border=True):
                st.subheader("Action items")
                structured_items = result.get("structured_action_items")
                if structured_items:
                    for i, item in enumerate(structured_items, 1):
                        p_badge = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(getattr(item, "priority", "Medium"), "🟡 Medium")
                        ts_badge = f"⏱️ `[{item.citation.timestamp_str}]`" if (hasattr(item, "citation") and item.citation) else ""
                        st.markdown(f"**{i}. {item.task}**")
                        owner_val = getattr(item, "owner", "Unknown") or "Unknown"
                        deadline_val = getattr(item, "deadline", "Not specified") or "Not specified"
                        st.caption(f"👤 **Assignee:** {owner_val} &nbsp;|&nbsp; {p_badge} &nbsp;|&nbsp; 📅 **Due:** {deadline_val} {f'&nbsp;|&nbsp; ' + ts_badge if ts_badge else ''}")
                        st.divider()
                else:
                    st.markdown(result["action_items"])
        with col_b:
            with st.container(border=True):
                st.subheader("Key decisions")
                structured_decs = result.get("structured_key_decisions")
                if structured_decs:
                    for i, dec in enumerate(structured_decs, 1):
                        ts_badge = f"⏱️ `[{dec.citation.timestamp_str}]`" if (hasattr(dec, "citation") and dec.citation) else ""
                        st.markdown(f"**{i}. {dec.decision}**")
                        rat_val = getattr(dec, "context_rationale", "Not specified") or "Not specified"
                        owner_val = getattr(dec, "consensus_owner", "Unknown") or "Unknown"
                        st.caption(f"💡 **Rationale:** {rat_val} &nbsp;|&nbsp; 🤝 **Consensus:** {owner_val} {f'&nbsp;|&nbsp; ' + ts_badge if ts_badge else ''}")
                        st.divider()
                else:
                    st.markdown(result["key_decisions"])
        with st.container(border=True):
            st.subheader("Open questions")
            st.markdown(result["open_questions"])

    with tab_chat:
        if not st.session_state.messages:
            st.caption("Ask anything about the meeting, for example: “What action items were decided?”")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question about this meeting")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the meeting transcript..."):
                    try:
                        resp = ask_post_meeting_question(
                            rag_chain=result.get("rag_chain"),
                            question=question,
                            chat_history=st.session_state.messages[:-1],
                            session_id=result.get("session_id", ""),
                        )
                        status_badge = f"📊 *Status: {resp.state.value.replace('_', ' ').title()}*\n\n"
                        full_answer = status_badge + resp.answer
                        if resp.citations:
                            full_answer += "\n\n**Evidence & Citations:**"
                            for c in resp.citations:
                                full_answer += f"\n- `{c.timestamp_str}`: *\"{c.excerpt}\"*"
                    except Exception as e:
                        full_answer = f"I couldn't answer that: `{e}`"
                st.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})

        if st.session_state.messages and st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


if result:
    render_results(result)