"""
Real Multilingual Audio Verification & End-to-End Pipeline Validation.

Tests real speech audio processing, automatic language detection, translation,
diarization, and hierarchical meeting intelligence.
"""

import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from core.transcriber import transcribe_all, set_processing_mode, get_processing_mode
from core.speaker_diarizer import diarize_segments
from core.meeting_intelligence import extract_hierarchical_intelligence
from core.meeting_qa import format_timestamp

audio_path = "test_slice_30s.wav"

print("======================================================================")
print("  REAL MULTILINGUAL AUDIO & END-TO-END INTELLIGENCE VERIFICATION")
print("======================================================================")

if not os.path.exists(audio_path):
    print(f"[!] Audio file '{audio_path}' not found. Skipping real audio probe.")
    exit(0)

# 1. Transcribe Real Audio with Fast Mode
print("\n[Step 1] Running Speech Recognition (Fast Mode - tiny on CPU)...")
t0 = time.perf_counter()
tx_res = transcribe_all(
    audio_path,
    language="auto",
    return_original=True,
    processing_mode="fast",
)
t_trans = time.perf_counter() - t0

orig_text = tx_res.get("original_text", "")
eng_text = tx_res.get("text", "")
segments = tx_res.get("segments", [])
det_lang = tx_res.get("detected_language", "Auto")

print(f"  -> Transcribed in: {t_trans:.2f}s (Audio: 30.0s, RTF: {t_trans / 30.0:.2f}x)")
print(f"  -> Detected Language: {det_lang}")
print(f"  -> Segments Extracted: {len(segments)}")
print(f"  -> Original Sample: {orig_text[:80]}...")
print(f"  -> English Sample:  {eng_text[:80]}...")

# 2. Speaker Diarization on Real Audio
print("\n[Step 2] Running CPU Speaker Diarization across segments...")
t_diar0 = time.perf_counter()
diar_segs = diarize_segments(audio_path, segments=segments, sample_rate=16000)
t_diar = time.perf_counter() - t_diar0

distinct_speakers = set(s.get("speaker", "Unknown") for s in diar_segs)
print(f"  -> Diarization completed in: {t_diar:.2f}s")
print(f"  -> Discovered Speakers: {sorted(list(distinct_speakers))}")
for s in diar_segs[:3]:
    print(f"     [{s.get('start', 0.0):.1f}s - {s.get('end', 0.0):.1f}s] {s.get('speaker')}: {s.get('text', '')[:40]}...")

# 3. Comprehensive Meeting Intelligence
print("\n[Step 3] Running Hierarchical Meeting Intelligence Extraction directly on real audio segments...")
# We use the actual segments produced by Step 1 & 2 on the real audio file!
actual_audio_duration = max((s.get("end", 0.0) for s in diar_segs), default=30.0)
real_transcript = " ".join(f"[{format_timestamp(s.get('start', 0.0))} - {format_timestamp(s.get('end', 0.0))}] {s.get('speaker', 'Speaker 1')}: {s.get('text', '')}" for s in diar_segs)

t_intel0 = time.perf_counter()
report = extract_hierarchical_intelligence(
    transcript=real_transcript,
    segments=diar_segs,
    duration_seconds=actual_audio_duration,
)
t_intel = time.perf_counter() - t_intel0

print(f"  -> Intelligence extracted in: {t_intel:.2f}s")
print(f"  -> Title: {report.title}")
print(f"  -> Overview: {report.meeting_overview[:100]}...")
print(f"  -> Topics Extracted: {len(report.topics_discussed)}")
for t in report.topics_discussed:
    print(f"     • {t.topic_title} [{t.start_time} - {t.end_time}]")
print(f"  -> Key Decisions Extracted: {len(report.key_decisions)}")
for d in report.key_decisions:
    print(f"     • {d.decision} ({d.timestamp})")
print(f"  -> Action Items Extracted: {len(report.action_items)}")
for a in report.action_items:
    print(f"     • {a.task} [Assignee: {a.assignee}, Due: {a.deadline}, Priority: {a.priority}] ({a.timestamp})")
print(f"  -> Next Meeting: {report.next_meeting_details}")
print(f"  -> MoM Chair: {report.minutes_of_meeting.chair_or_facilitator}")

print("\n[+] Verification successfully completed.")
