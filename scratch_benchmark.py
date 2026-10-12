import os
import sys
import time

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import convert_to_wav, chunk_audio
from core.transcriber import load_model, transcribe_chunk_whisper
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.vector_store import build_vector_store

print("=" * 60)
print("PHASE 1 BOTTLENECK BENCHMARK")
print("=" * 60)

timings = {}

# 1. File Audio Conversion & Chunking
if os.path.exists("test_audio.webm"):
    t0 = time.perf_counter()
    wav_path = convert_to_wav("test_audio.webm", force_reconvert=True)
    t_conv = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_audio(wav_path, force_rechunk=True)
    t_chunk = time.perf_counter() - t0

    timings["Audio Conversion (6.5 min WebM -> WAV)"] = t_conv
    timings["Audio Chunking (3-min chunks)"] = t_chunk
    print(f"[*] Audio Conversion: {t_conv:.2f}s")
    print(f"[*] Audio Chunking:   {t_chunk:.2f}s ({len(chunks)} chunks created)")

    # 2. Speech Transcription (Measure 1 chunk with current Whisper small CPU)
    if chunks:
        print("[*] Benchmarking Whisper small on 1 audio chunk...")
        t0 = time.perf_counter()
        whisper_text = transcribe_chunk_whisper(chunks[0], language="hi", task="transcribe")
        t_transcribe = time.perf_counter() - t0
        timings["Speech Transcription (1 chunk: 3 min audio)"] = t_transcribe
        print(f"[*] Whisper Transcription (1 chunk / 3 min): {t_transcribe:.2f}s")
        estimated_total_transcribe = t_transcribe * len(chunks)
        timings[f"Estimated Total Transcription ({len(chunks)} chunks / 6.5 min)"] = estimated_total_transcribe
        print(f"[*] Estimated Full Video Transcription: {estimated_total_transcribe:.2f}s")

# 3. LLM Processing on transcript.txt
if os.path.exists("transcript.txt"):
    with open("transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()

    print(f"\n[*] Transcript length: {len(transcript)} chars, {len(transcript.split())} words")

    t0 = time.perf_counter()
    title = generate_title(transcript)
    timings["LLM Title Generation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    summary = summarize(transcript)
    timings["LLM Summary"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    actions = extract_action_items(transcript)
    timings["LLM Action Items"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    decisions = extract_key_decisions(transcript)
    timings["LLM Key Decisions"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    questions = extract_questions(transcript)
    timings["LLM Open Questions"] = time.perf_counter() - t0

    total_llm = (timings["LLM Title Generation"] + timings["LLM Summary"] + 
                 timings["LLM Action Items"] + timings["LLM Key Decisions"] + timings["LLM Open Questions"])
    timings["Total LLM Calls (5 serial calls)"] = total_llm

    print(f"[*] LLM Title:         {timings['LLM Title Generation']:.2f}s")
    print(f"[*] LLM Summary:       {timings['LLM Summary']:.2f}s")
    print(f"[*] LLM Actions:       {timings['LLM Action Items']:.2f}s")
    print(f"[*] LLM Decisions:     {timings['LLM Key Decisions']:.2f}s")
    print(f"[*] LLM Questions:     {timings['LLM Open Questions']:.2f}s")
    print(f"[*] Total LLM Calls:   {total_llm:.2f}s")

    # 4. Embedding Generation & ChromaDB Indexing
    t0 = time.perf_counter()
    vs = build_vector_store(transcript, collection_name="bench_phase1", clear_existing=True)
    timings["Embeddings & ChromaDB Indexing"] = time.perf_counter() - t0
    print(f"[*] VectorStore & Indexing: {timings['Embeddings & ChromaDB Indexing']:.2f}s")

print("\n" + "=" * 60)
print("SUMMARY TABLE OF MEASURED TIMES")
print("=" * 60)
for stage, elapsed in timings.items():
    print(f"{stage:<50}: {elapsed:>7.2f}s")
print("=" * 60)
