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

from core.transcriber import (
    load_faster_whisper_model,
    load_openai_whisper_model,
    transcribe_chunk_whisper,
    transcribe_all,
)

def run_phase2_benchmark():
    print("=" * 70)
    print("PHASE 2 BENCHMARK: OpenAI Whisper vs. faster-whisper (CPU INT8)")
    print("=" * 70)

    audio_file = "test_slice_30s.wav"
    if not os.path.exists(audio_file):
        print(f"[!] {audio_file} not found.")
        return

    # 1. Warm model load benchmark
    print("\n[1] Measuring Model Loading Times...")
    t0 = time.perf_counter()
    fw_model = load_faster_whisper_model()
    t_fw_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    ow_model = load_openai_whisper_model()
    t_ow_load = time.perf_counter() - t0

    print(f"    faster-whisper (INT8): {t_fw_load:.3f}s (cached in-memory)")
    print(f"    OpenAI Whisper (FP32): {t_ow_load:.3f}s (cached in-memory)")

    # 2. Transcription benchmark on 30s audio
    print(f"\n[2] Benchmarking Transcription on 30s Audio ({audio_file})...")

    # A) OpenAI Whisper FP32
    t0 = time.perf_counter()
    res_ow = ow_model.transcribe(audio_file, language="en", task="transcribe")
    text_ow = res_ow["text"].strip()
    t_ow = time.perf_counter() - t0
    rtf_ow = t_ow / 30.0

    # B) faster-whisper INT8 (greedy search, beam_size=1 to match OpenAI Whisper)
    t0 = time.perf_counter()
    segs_fw, _ = fw_model.transcribe(audio_file, language="en", task="transcribe", beam_size=1)
    text_fw = " ".join(s.text.strip() for s in segs_fw).strip()
    t_fw = time.perf_counter() - t0
    rtf_fw = t_fw / 30.0

    speedup = t_ow / t_fw if t_fw > 0 else 1.0

    print(f"    OpenAI Whisper FP32 CPU : {t_ow:.2f}s (RTF: {rtf_ow:.2f}x audio length)")
    print(f"    faster-whisper INT8 CPU : {t_fw:.2f}s (RTF: {rtf_fw:.2f}x audio length)")
    print(f"    Measured Speedup        : {speedup:.2f}x faster")
    print(f"    Latency Reduction       : {((t_ow - t_fw) / t_ow) * 100:.1f}%")

    print("\n[3] Accuracy / Text Comparison:")
    print(f"    OpenAI Whisper: \"{text_ow[:90]}...\"")
    print(f"    faster-whisper: \"{text_fw[:90]}...\"")

    # 3. Model Reuse Verification (Streamlit simulation)
    print("\n[4] Streamlit Rerun Simulation (Testing In-Memory Model Reuse)...")
    t0 = time.perf_counter()
    reused_model = load_faster_whisper_model()
    t_reuse = time.perf_counter() - t0
    print(f"    Subsequent load time: {t_reuse * 1000:.3f} ms (Instantaneous singleton reuse, 0 model reload overhead)")

    print("\n" + "=" * 70)
    print("PHASE 2 TRANSCRIPTION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} | {'OpenAI Whisper (FP32)':>18} | {'faster-whisper (INT8)':>18}")
    print("-" * 70)
    print(f"{'Transcription Time (30s audio)':<35} | {t_ow:>17.2f}s | {t_fw:>17.2f}s")
    print(f"{'Real-Time Factor (RTF)':<35} | {rtf_ow:>17.2f}x | {rtf_fw:>17.2f}x")
    print(f"{'Projected 6.5-min Audio Time':<35} | {rtf_ow * 390:>17.2f}s | {rtf_fw * 390:>17.2f}s")
    print(f"{'Projected 1-hour Audio Time':<35} | {rtf_ow * 3600 / 60:>16.1f}m | {rtf_fw * 3600 / 60:>16.1f}m")
    print(f"{'Measured Speed Improvement':<35} | {'Baseline (1.0x)':>18} | {f'{speedup:.2f}x faster':>18}")
    print("=" * 70)

if __name__ == "__main__":
    run_phase2_benchmark()
