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

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language=language)
    print(f"\nRaw transcription (first 300 chars):\n{transcript[:300]}...\n")

    print("Generating title...")
    title = generate_title(transcript)
    time.sleep(1.0)

    print("Generating summary...")
    summary = summarize(transcript)
    time.sleep(1.0)

    print("Extracting action items...")
    actions_item = extract_action_items(transcript)
    time.sleep(1.0)

    print("Extracting key decisions...")
    decisions = extract_key_decisions(transcript)
    time.sleep(1.0)

    print("Extracting open questions...")
    questions = extract_questions(transcript)
    time.sleep(1.0)

    print("Building RAG vector store and chain...")
    rag_chain = build_rag_chain(transcript)

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
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

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