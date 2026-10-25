import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    extract_all_insights,
)
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english", session_id: str = None) -> dict:
    print("\nStarting AI Video Assistant...")
    stage_timings = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    chunks = process_input(source)
    stage_timings["Audio Preparation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    transcript = transcribe_all(chunks, language=language)
    stage_timings["Speech Transcription"] = time.perf_counter() - t0
    print(f"\nRaw transcription (first 300 chars):\n{transcript[:300]}...\n")

    t_llm = time.perf_counter()
    print("Running parallel LLM analysis (Title, Summary, Actions, Decisions, Questions)...")
    llm_res = extract_all_insights(
        transcript,
        progress_callback=lambda curr, total, name: print(f"  [+] ({curr}/{total}) {name} completed")
    )
    stage_timings.update(llm_res["stage_timings"])
    stage_timings["LLM Analysis (Parallel)"] = time.perf_counter() - t_llm

    title = llm_res["title"]
    summary = llm_res["summary"]
    action_items = llm_res["action_items"]
    decisions = llm_res["key_decisions"]
    questions = llm_res["open_questions"]

    import uuid
    if not session_id:
        source_base = os.path.splitext(os.path.basename(source))[0][:16].replace(" ", "_")
        session_id = f"{source_base}_{uuid.uuid4().hex[:6]}" if source_base else f"cli_{uuid.uuid4().hex[:8]}"

    t0 = time.perf_counter()
    print(f"Building RAG vector store and chain (session: {session_id})...")
    rag_chain = build_rag_chain(transcript, session_id=session_id)
    stage_timings["ChromaDB Indexing"] = time.perf_counter() - t0

    total_time = time.perf_counter() - t_start
    stage_timings["Total Pipeline Time"] = total_time

    print("\n" + "=" * 60)
    print("STAGE TIMINGS BREAKDOWN")
    print("=" * 60)
    for k, v in stage_timings.items():
        print(f"  - {k:<25}: {v:>6.2f}s")
    print("=" * 60)

    return {
        "transcript": transcript,
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "session_id": session_id,
        "stage_timings": stage_timings,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english / nepali / nepali-to-english / hinglish / auto): ").strip() or "english"

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print("=" * 60)

    print("\n[SUMMARY]")
    print("-" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("[ACTION ITEMS]")
    print("=" * 60)
    print(result["action_items"])

    print("\n" + "=" * 60)
    print("[KEY DECISIONS]")
    print("=" * 60)
    print(result["key_decisions"])

    print("\n" + "=" * 60)
    print("[OPEN QUESTIONS]")
    print("=" * 60)
    print(result["open_questions"])

    print("\n" + "=" * 60)
    print("[TRANSCRIPT (preview)]")
    print("=" * 60)
    print(result["transcript"][:500] + "..." if len(result["transcript"]) > 500 else result["transcript"])

    # Phase 2 - Chat with your meeting via RAG
    print("\n" + "=" * 60)
    print("Chat with your video transcript via RAG (type 'exit' to quit)")
    print("=" * 60 + "\n")
    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        ask_question(rag_chain, question)
        print("-" * 40)