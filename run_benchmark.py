import time
import tracemalloc
from core.transcriber import (
    transcribe_all,
    set_processing_mode,
    get_active_model_name,
    unload_all_models,
    load_model,
)

test_audio = "test_slice_30s.wav"
modes = ["fast", "balanced"]

print("Starting benchmarks...")
for mode in modes:
    unload_all_models()
    set_processing_mode(mode)
    model_name = get_active_model_name()

    t0 = time.perf_counter()
    _ = load_model()
    load_time = time.perf_counter() - t0

    tracemalloc.start()
    t_start = time.perf_counter()
    res = transcribe_all(test_audio, language="auto", return_original=True, processing_mode=mode)
    elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rtf = elapsed / 30.0
    text_sample = res.get("text", "")[:60]
    orig_sample = res.get("original_text", "")[:60]
    detected_lang = res.get("detected_language", "unknown")

    print(f"--- Mode: {mode} (Model: {model_name}) ---")
    print(f"  Load Time: {load_time:.3f}s")
    print(f"  Transcribe Time (30s audio): {elapsed:.3f}s")
    print(f"  Real-Time Factor (RTF): {rtf:.3f}x")
    print(f"  Peak RAM Increase: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"  Detected Lang: {detected_lang}")
    print(f"  Original Text: {orig_sample}")
    print(f"  English Text: {text_sample}")
