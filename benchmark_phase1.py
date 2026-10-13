import os
import sys
import time

# Ensure UTF-8 output on Windows terminal
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
from pydub import AudioSegment

def run_phase1_benchmark():
    print("=" * 70)
    print("PHASE 1 BOTTLENECK PROFILING")
    print("=" * 70)

    timings = {}

    # Stage 1: Audio conversion & chunking
    if os.path.exists("test_audio.webm"):
        print("\n[1] Profiling Audio Conversion (WebM -> WAV 16kHz mono)...")
        t0 = time.perf_counter()
        wav_path = convert_to_wav("test_audio.webm", force_reconvert=True)
        t_conv = time.perf_counter() - t0
        timings["Audio Conversion (6.5 min WebM -> WAV)"] = t_conv
        print(f"    Done in {t_conv:.2f}s")

        print("\n[2] Profiling Audio Chunking (3-minute slices)...")
        t0 = time.perf_counter()
        chunks = chunk_audio(wav_path, force_rechunk=True)
        t_chunk = time.perf_counter() - t0
        timings["Audio Chunking (3-min chunks)"] = t_chunk
        print(f"    Done in {t_chunk:.2f}s ({len(chunks)} chunks created)")

        # Create a 30-second slice to benchmark Whisper speed per second of audio
        audio = AudioSegment.from_wav(chunks[0])
        slice_30s_path = "test_slice_30s.wav"
        audio[:30000].export(slice_30s_path, format="wav")

        print("\n[3] Profiling Whisper small (CPU FP32) on 30 seconds of English audio...")
        t0 = time.perf_counter()
        text_30s = transcribe_chunk_whisper(slice_30s_path, language="en", task="transcribe")
        t_whisper_30s = time.perf_counter() - t0
        timings["Whisper small CPU (30s audio)"] = t_whisper_30s
        rtf_30s = t_whisper_30s / 30.0
        timings["Whisper Real-Time Factor (RTF)"] = rtf_30s
        print(f"    Done in {t_whisper_30s:.2f}s (Real-Time Factor: {rtf_30s:.2f}x audio duration)")
        print(f"    Sample text: {text_30s[:100]}...")

        # Estimate full 6.5 min audio transcription time
        est_6_5_min = rtf_30s * 390.0
        timings["Projected Transcription (6.5 min audio)"] = est_6_5_min
        print(f"    Projected 6.5 min transcription time: {est_6_5_min:.2f}s ({est_6_5_min/60:.2f} min)")

    # Stage 2: LLM Calls (Mistral)
    if os.path.exists("transcript.txt"):
        with open("transcript.txt", "r", encoding="utf-8") as f:
            transcript = f.read()

        words = len(transcript.split())
        chars = len(transcript)
        print(f"\n[4] Profiling LLM Calls on transcript ({words} words, {chars} chars)...")

        t0 = time.perf_counter()
        title = generate_title(transcript)
        t_title = time.perf_counter() - t0
        timings["LLM Title Generation"] = t_title
        print(f"    Title ({t_title:.2f}s): {title}")

        t0 = time.perf_counter()
        summary = summarize(transcript)
        t_sum = time.perf_counter() - t0
        timings["LLM Summary Generation"] = t_sum
        print(f"    Summary ({t_sum:.2f}s): {len(summary)} chars")

        t0 = time.perf_counter()
        actions = extract_action_items(transcript)
        t_act = time.perf_counter() - t0
        timings["LLM Action Items"] = t_act
        print(f"    Action Items ({t_act:.2f}s): {len(actions)} chars")

        t0 = time.perf_counter()
        decisions = extract_key_decisions(transcript)
        t_dec = time.perf_counter() - t0
        timings["LLM Key Decisions"] = t_dec
        print(f"    Key Decisions ({t_dec:.2f}s): {len(decisions)} chars")

        t0 = time.perf_counter()
        questions = extract_questions(transcript)
        t_q = time.perf_counter() - t0
        timings["LLM Open Questions"] = t_q
        print(f"    Open Questions ({t_q:.2f}s): {len(questions)} chars")

        total_llm = t_title + t_sum + t_act + t_dec + t_q
        timings["Total LLM Calls (Sequential)"] = total_llm
        print(f"    Total LLM time: {total_llm:.2f}s")

        # Stage 3: Embedding & ChromaDB Indexing
        print(f"\n[5] Profiling Embedding Generation & ChromaDB Indexing...")
        t0 = time.perf_counter()
        build_vector_store(transcript, collection_name="phase1_bench_coll", clear_existing=True)
        t_vec = time.perf_counter() - t0
        timings["Embedding & ChromaDB Indexing"] = t_vec
        print(f"    Vector store built in {t_vec:.2f}s")

    print("\n" + "=" * 70)
    print("PHASE 1 BOTTLENECK BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Stage / Operation':<45} | {'Elapsed Time (s)':>16}")
    print("-" * 70)
    for name, elapsed in timings.items():
        if "Factor" in name:
            print(f"{name:<45} | {elapsed:>15.2f}x")
        else:
            print(f"{name:<45} | {elapsed:>15.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    run_phase1_benchmark()
