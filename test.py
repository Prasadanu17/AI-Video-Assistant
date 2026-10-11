import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()   # MUST be before any core/ imports

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


source = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"
language = "english"   # "english" → Whisper, "hinglish" → Sarvam

CACHE_FILE = "transcript.txt"

# If a cached transcript exists, reuse it to avoid re-running slow Whisper CPU transcription
if os.path.exists(CACHE_FILE):
    print(f"[i] Found cached transcript in {CACHE_FILE}. Loading...")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        transcript = f.read()
else:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"[+] Saved transcript to {CACHE_FILE}")

print("\n" + "=" * 60)
print("[TRANSCRIPT]")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


title = generate_title(transcript)
time.sleep(1.0)
summary = summarize(transcript)

print("\n" + "=" * 60)
print(f"[TITLE] {title}")
print("=" * 60)
print("\n[SUMMARY]")
print("-" * 60)
print(summary)


action_items = extract_action_items(transcript)
decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)

print("\n" + "=" * 60)
print("[ACTION ITEMS]")
print("=" * 60)
print(action_items)

print("\n" + "=" * 60)
print("[KEY DECISIONS]")
print("=" * 60)
print(decisions)

print("\n" + "=" * 60)
print("[OPEN QUESTIONS]")
print("=" * 60)
print(questions)