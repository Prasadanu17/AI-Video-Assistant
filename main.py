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
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
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

    t0 = time.perf_counter()
    print("Generating title...")
    title = generate_title(transcript)
    stage_timings["Title Generation"] = time.perf_counter() - t0
    time.sleep(1.0)

    t0 = time.perf_counter()
    print("Generating summary...")
    summary = summarize(transcript)
    stage_timings["Summary Generation"] = time.perf_counter() - t0
    time.sleep(1.0)

    t0 = time.perf_counter()
    print("Extracting action items...")
    actions_item = extract_action_items(transcript)
    stage_timings["Action Items"] = time.perf_counter() - t0
    time.sleep(1.0)

    t0 = time.perf_counter()
    print("Extracting key decisions...")
    decisions = extract_key_decisions(transcript)
    stage_timings["Key Decisions"] = time.perf_counter() - t0
    time.sleep(1.0)

    t0 = time.perf_counter()
    print("Extracting open questions...")
    questions = extract_questions(transcript)
    stage_timings["Open Questions"] = time.perf_counter() - t0
    time.sleep(1.0)

    t0 = time.perf_counter()
    print("Building RAG vector store and chain...")
    rag_chain = build_rag_chain(transcript)
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
        "action_items": actions_item,
        "actions_items": actions_item,
        "key_decisions": decisions,
        "decisions": decisions,
        "open_questions": questions,
        "questions": questions,
        "rag_chain": rag_chain,
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