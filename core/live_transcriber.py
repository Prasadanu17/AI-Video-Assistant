"""
Live Transcription Engine for AI Video Assistant (Phase 5).

Provides production-grade incremental live speech-to-text with:
- Non-blocking background worker thread consuming from a thread-safe Queue.
- Configurable live chunking (e.g. 5.0s chunk, 0.8s overlap) with boundary deduplication.
- Accurate recording timeline timestamps, accounting for pauses and overlap.
- Native multilingual transcription (English, Hindi, Nepali in Devanagari, Auto-detect) without forced translation.
- Graceful error isolation, queue backpressure handling, and flush-on-stop guarantees.
"""

import math
import os
import queue
import re
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.transcriber import resolve_language_mode, transcribe_chunk


@dataclass
class LiveTranscriptSegment:
    """Represents a single transcribed speech segment with timeline bounds."""
    index: int
    start: float  # Timeline start in seconds
    end: float    # Timeline end in seconds
    text: str
    language: str
    is_final: bool = True


@dataclass
class LiveTranscript:
    """Thread-safe state container holding the progressive live transcript."""
    segments: List[LiveTranscriptSegment] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    is_busy: bool = False
    total_audio_seconds: float = 0.0
    total_inference_seconds: float = 0.0
    dropped_chunks_count: int = 0
    dropped_audio_seconds: float = 0.0
    last_error: Optional[str] = None
    session_token: Optional[str] = None
    is_finalized: bool = False

    def add_segment(self, segment: LiveTranscriptSegment, session_token: Optional[str] = None) -> bool:
        with self._lock:
            # Reject late results if transcript has been finalized
            if self.is_finalized:
                return False
            # Reject results from an old/superseded session
            if session_token is not None and self.session_token is not None and session_token != self.session_token:
                return False
            self.segments.append(segment)
            return True

    @property
    def full_text(self) -> str:
        with self._lock:
            return " ".join(s.text.strip() for s in self.segments if s.text.strip()).strip()

    @property
    def real_time_factor(self) -> float:
        with self._lock:
            if self.total_audio_seconds <= 0.0:
                return 0.0
            return self.total_inference_seconds / self.total_audio_seconds

    def get_snapshot(self) -> dict:
        """Return a thread-safe copy of the live transcript state for UI rendering."""
        with self._lock:
            text_val = " ".join(s.text.strip() for s in self.segments if s.text.strip()).strip()
            rtf = (self.total_inference_seconds / self.total_audio_seconds) if self.total_audio_seconds > 0.0 else 0.0
            return {
                "segments": [
                    {
                        "index": s.index,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "language": s.language,
                        "is_final": s.is_final,
                    }
                    for s in self.segments
                ],
                "full_text": text_val,
                "text": text_val,
                "is_busy": self.is_busy,
                "segment_count": len(self.segments),
                "total_audio_seconds": self.total_audio_seconds,
                "total_inference_seconds": self.total_inference_seconds,
                "dropped_chunks_count": self.dropped_chunks_count,
                "dropped_audio_seconds": self.dropped_audio_seconds,
                "real_time_factor": rtf,
                "last_error": self.last_error,
                "session_token": self.session_token,
                "is_finalized": self.is_finalized,
            }

    def clear(self) -> None:
        with self._lock:
            self.segments.clear()
            self.is_busy = False
            self.total_audio_seconds = 0.0
            self.total_inference_seconds = 0.0
            self.dropped_chunks_count = 0
            self.dropped_audio_seconds = 0.0
            self.last_error = None
            self.is_finalized = False
            self.session_token = None



def deduplicate_overlap(
    prev_text: str,
    new_text: str,
    max_check_words: int = 6,
    min_match_words: int = 2,
) -> str:
    """
    Remove boundary duplicate words caused by audio chunk overlap.
    Works for English, Hindi (Devanagari), Nepali (Devanagari), and mixed text.
    Avoids stripping valid words on coincidental 1-word matches, while preserving
    intentional repetitions (e.g. 'very very', 'yes yes') and boundary phrases.
    """
    if not prev_text or not new_text:
        return new_text.strip()

    prev_words = prev_text.strip().split()
    new_words = new_text.strip().split()

    if not prev_words or not new_words:
        return new_text.strip()

    def clean_word(w: str) -> str:
        return re.sub(r"[^\w\s]", "", w.strip().lower())

    # Check for matching word sequences of length k down to min_match_words
    max_k = min(len(prev_words), len(new_words), max_check_words)
    best_overlap = 0

    for k in range(max_k, min_match_words - 1, -1):
        prev_tail = [clean_word(w) for w in prev_words[-k:]]
        new_head = [clean_word(w) for w in new_words[:k]]
        if prev_tail == new_head and all(len(w) > 0 for w in prev_tail):
            best_overlap = k
            break

    # If no multi-word match, check if 1-word overlap is a distinctive long word (>= 6 chars)
    # AND that prev_words does not end with a deliberate duplicate
    if best_overlap == 0 and len(prev_words) >= 1 and len(new_words) >= 1:
        prev_w = clean_word(prev_words[-1])
        new_w = clean_word(new_words[0])
        if prev_w == new_w and len(prev_w) >= 6:
            # Check it's not intentional repetition like 'really really'
            if len(prev_words) < 2 or clean_word(prev_words[-2]) != prev_w:
                best_overlap = 1

    if best_overlap > 0:
        remaining = new_words[best_overlap:]
        if remaining:
            return " ".join(remaining).strip()
        # If the entire new chunk is subsumed by previous overlap, return empty
        return ""

    return new_text.strip()


def preprocess_audio_chunk(
    pcm_data: np.ndarray,
    sample_rate: int = 16000,
    target_peak: float = 0.65,
    max_gain: float = 2.8,
) -> Tuple[np.ndarray, dict]:
    """
    Safely preprocess audio samples for robust transcription:
    - Normalizes dtype to float32.
    - Measures peak amplitude and RMS energy.
    - Detects clipping and near-silence.
    - Boosts quiet speech without amplifying background noise floor.
    - Applies gentle soft-limiting to eliminate hard clipping distortion.
    """
    if len(pcm_data) == 0:
        return np.zeros(0, dtype=np.float32), {
            "status": "empty",
            "peak": 0.0,
            "rms": 0.0,
            "is_clipped": False,
            "is_near_silence": True,
            "applied_gain": 1.0,
        }

    # Ensure float32 representation in [-1.0, 1.0]
    if pcm_data.dtype == np.int16:
        samples = pcm_data.astype(np.float32) / 32768.0
    else:
        samples = pcm_data.astype(np.float32).copy()
        max_val = np.max(np.abs(samples)) if len(samples) > 0 else 0.0
        if max_val > 1.5:
            samples = samples / 32768.0

    peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

    # Sustained flat-top clipping: 3 or more consecutive samples pinned to digital full scale (>=0.99)
    if peak >= 0.99 and len(samples) >= 3:
        rail_hits = np.abs(samples) >= 0.99
        is_clipped = bool(np.any(rail_hits[:-2] & rail_hits[1:-1] & rail_hits[2:]))
    else:
        is_clipped = False

    is_near_silence = rms < 0.002

    applied_gain = 1.0
    # Apply safe dynamic gain boost only when speech energy is present (rms >= 0.003)
    # and signal is quiet (peak < 0.35)
    if not is_near_silence and rms >= 0.003 and peak > 0.01 and peak < 0.35:
        applied_gain = min(target_peak / max(peak, 1e-4), max_gain)
        samples = samples * applied_gain
        peak = float(np.max(np.abs(samples)))

    # Soft-limiting: smooth out any harsh clipping
    if peak > 0.95 or is_clipped:
        samples = np.tanh(samples / 0.95) * 0.95

    metrics = {
        "status": "ok" if not is_near_silence else "near_silence",
        "peak": peak,
        "rms": rms,
        "is_clipped": is_clipped,
        "is_near_silence": is_near_silence,
        "applied_gain": applied_gain,
    }
    return samples, metrics



class LiveTranscriber:
    """
    Background worker orchestrating incremental live transcription.
    Decoupled from audio capture via a thread-safe Queue with backpressure monitoring.
    """

    def __init__(
        self,
        language: str = "english",
        chunk_seconds: float = 5.0,
        overlap_seconds: float = 0.8,
        sample_rate: int = 16000,
        mock_mode: bool = False,
        on_segment_callback: Optional[Callable[[LiveTranscriptSegment], None]] = None,
    ):
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.sample_rate = sample_rate
        self.mock_mode = mock_mode
        self.on_segment_callback = on_segment_callback

        self.transcript = LiveTranscript()
        self._queue: queue.Queue = queue.Queue(maxsize=50)

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._paused = False
        self._segment_index = 0

        # Session lifecycle & isolation
        self._session_generation: int = 0
        self._current_session_token: str = f"sess_0_{uuid.uuid4().hex[:8]}"
        self._is_finalized: bool = False
        self.transcript.session_token = self._current_session_token

        # Backpressure monitoring
        self._chunks_enqueued = 0
        self._chunks_processed = 0
        self._chunks_dropped = 0
        self._dropped_audio_seconds = 0.0

    @property
    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def session_token(self) -> str:
        return self._current_session_token

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @property
    def pending_chunk_count(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_chunk_count(self) -> int:
        return self._chunks_dropped

    def start(self) -> str:
        """Start the background transcription worker thread with a unique session token."""
        if self.is_running and not self._is_finalized:
            return self._current_session_token

        self._session_generation += 1
        session_token = f"sess_{self._session_generation}_{uuid.uuid4().hex[:8]}"
        self._current_session_token = session_token
        self._is_finalized = False
        with self.transcript._lock:
            self.transcript.session_token = session_token
            self.transcript.is_finalized = False

        self._stop_event.clear()
        self._paused = False
        self._segment_index = 0
        self._chunks_enqueued = 0
        self._chunks_processed = 0
        self._chunks_dropped = 0
        self._dropped_audio_seconds = 0.0

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"LiveTranscriberWorker-{self._session_generation}",
                daemon=True,
            )
            self._worker_thread.start()
        print(f"[LiveTranscriber] Started live transcription worker (session='{session_token}', lang='{self.language}').")
        return session_token

    def pause(self) -> None:
        """Pause live transcription. In-flight chunks finish, but incoming chunks are ignored."""
        self._paused = True
        print("[LiveTranscriber] Paused live transcription.")

    def resume(self) -> None:
        """Resume live transcription."""
        self._paused = False
        print("[LiveTranscriber] Resumed live transcription.")

    def enqueue_chunk(
        self,
        pcm_data: np.ndarray,
        start_time: float,
        end_time: float,
        is_final: bool = False,
    ) -> bool:
        """
        Thread-safely enqueue a completed audio chunk for transcription.
        Non-blocking: will never stall audio recording threads.
        """
        if self._paused and not is_final:
            return False

        if self._is_finalized:
            print(f"[LiveTranscriber] Notice: Rejected chunk for finalized session '{self._current_session_token}'.")
            return False

        if len(pcm_data) == 0:
            return False

        item = {
            "pcm_data": pcm_data.copy(),
            "start_time": start_time,
            "end_time": end_time,
            "is_final": is_final,
            "session_token": self._current_session_token,
            "session_generation": self._session_generation,
        }

        try:
            # Backpressure handling: if queue is full, record dropped metrics
            self._queue.put_nowait(item)
            self._chunks_enqueued += 1
            return True
        except queue.Full:
            self._chunks_dropped += 1
            duration = max(0.0, end_time - start_time)
            self._dropped_audio_seconds += duration
            with self.transcript._lock:
                self.transcript.dropped_chunks_count = self._chunks_dropped
                self.transcript.dropped_audio_seconds = self._dropped_audio_seconds
                self.transcript.last_error = (
                    f"Queue overflow: dropped {self._chunks_dropped} chunk(s) "
                    f"({self._dropped_audio_seconds:.1f}s) due to inference backpressure."
                )
            print(f"[LiveTranscriber] Warning: Queue full! Audio backpressure detected. Dropped {self._chunks_dropped} chunk(s) ({self._dropped_audio_seconds:.1f}s).")
            return False


    def stop(self, timeout: float = None) -> str:
        """
        Stop worker, flush all remaining audio chunks in the queue for the current session,
        and return final transcript. Guarantees that pending chunks of this session are drained
        and processed before finalizing. Once finalized, late or stale results are discarded.
        """
        print(f"[LiveTranscriber] Stopping live transcriber (session='{self._current_session_token}')...")
        self._stop_event.set()

        # Enqueue sentinel to wake up worker if blocked
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        # Dynamically scale timeout based on queued chunks (minimum 30s)
        pending = self._queue.qsize()
        effective_timeout = timeout if timeout is not None else max(30.0, pending * 45.0 + 30.0)

        worker_timed_out = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=effective_timeout)
            if self._worker_thread.is_alive():
                worker_timed_out = True

        # Seal and finalize the session so any delayed or lingering worker inference
        # cannot append to this transcript or contaminate subsequent sessions.
        self._is_finalized = True
        with self.transcript._lock:
            self.transcript.is_finalized = True

        if worker_timed_out:
            rem = self._queue.qsize()
            warning_msg = (
                f"Stop timed out after {effective_timeout:.1f}s: worker still active in background "
                f"({rem} pending chunk(s) un-drained). Session is finalized; late results will be discarded."
            )
            print(f"[LiveTranscriber] Warning: {warning_msg}")
            with self.transcript._lock:
                self.transcript.last_error = warning_msg
        elif not self._queue.empty():
            rem = self._queue.qsize()
            warning_msg = f"Stop completed with {rem} pending chunk(s) remaining in queue."
            print(f"[LiveTranscriber] Notice: {warning_msg}")
            with self.transcript._lock:
                self.transcript.last_error = warning_msg

        self._worker_thread = None
        final_text = self.transcript.full_text
        print(f"[LiveTranscriber] Finished. Final transcript: {len(final_text)} chars, {len(self.transcript.segments)} segments.")
        return final_text

    def reset(self) -> None:
        """Reset transcriber state, finalize previous session, and drain queued items."""
        self._stop_event.set()
        self._is_finalized = True
        self._current_session_token = None

        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                break

        self.transcript.clear()
        self._segment_index = 0
        self._chunks_enqueued = 0
        self._chunks_processed = 0
        print("[LiveTranscriber] Reset to initial state.")

    # ----------------------------------------------------------------------
    # Worker Thread Processing Loop
    # ----------------------------------------------------------------------
    def _worker_loop(self) -> None:
        """Dedicated thread executing inference on enqueued audio chunks."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                # Sentinel encountered
                self._queue.task_done()
                break

            # Fast drop if chunk belongs to an already-finalized or superseded session
            chunk_session = item.get("session_token")
            if self._is_finalized or (chunk_session is not None and chunk_session != self._current_session_token):
                print(f"[LiveTranscriber] Notice: Discarding queued chunk from inactive/finalized session '{chunk_session}'.")
                self._queue.task_done()
                continue

            self.transcript.is_busy = True
            try:
                self._process_single_chunk(item)
            except Exception as e:
                print(f"[LiveTranscriber] Error processing chunk: {e}")
                self.transcript.last_error = str(e)
            finally:
                self.transcript.is_busy = False
                self._chunks_processed += 1
                self._queue.task_done()

    def _process_single_chunk(self, item: dict) -> None:
        pcm_data: np.ndarray = item["pcm_data"]
        start_time: float = item["start_time"]
        end_time: float = item["end_time"]
        chunk_duration = end_time - start_time

        t0 = time.perf_counter()

        if self.mock_mode:
            # Deterministic simulation for tests
            text = f"Simulated speech segment {self._segment_index + 1}."
            t_inf = 0.05
            time.sleep(0.02)
        else:
            # Preprocess audio: normalize, boost quiet speech, soft-limit clipping
            clean_pcm, metrics = preprocess_audio_chunk(pcm_data, sample_rate=self.sample_rate)
            if len(clean_pcm) == 0:
                return

            # Direct in-memory NumPy inference with safe fallback to temporary WAV
            try:
                res = transcribe_chunk(
                    clean_pcm,
                    language=self.language,
                    task="transcribe",
                    return_segments=True,
                )
                text = res["text"] if isinstance(res, dict) else str(res)
            except Exception as direct_err:
                # Safe fallback to temporary WAV file if in-memory fails
                tmp_wav = self._write_temp_chunk(clean_pcm)
                try:
                    res = transcribe_chunk(
                        tmp_wav,
                        language=self.language,
                        task="transcribe",
                        return_segments=True,
                    )
                    text = res["text"] if isinstance(res, dict) else str(res)
                finally:
                    if os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except Exception:
                            pass
            t_inf = time.perf_counter() - t0

        chunk_session = item.get("session_token")

        # Record audio duration and inference timing only if session is still active
        with self.transcript._lock:
            if self._is_finalized or (chunk_session is not None and chunk_session != self._current_session_token):
                print(f"[LiveTranscriber] Notice: Discarding completed inference for inactive/finalized session '{chunk_session}'.")
                return
            self.transcript.total_audio_seconds += chunk_duration
            self.transcript.total_inference_seconds += t_inf

        clean_text = text.strip()
        if not clean_text:
            return

        # Deduplicate overlap against previous segment
        prev_text = self.transcript.full_text
        deduped_text = deduplicate_overlap(prev_text, clean_text)

        if not deduped_text:
            return

        # Create finalized segment
        segment = LiveTranscriptSegment(
            index=self._segment_index,
            start=round(start_time, 2),
            end=round(end_time, 2),
            text=deduped_text,
            language=self.language,
            is_final=True,
        )
        self._segment_index += 1

        added = self.transcript.add_segment(segment, session_token=chunk_session)
        if not added:
            print(f"[LiveTranscriber] Notice: Segment rejected by transcript (session mismatch or finalized).")
            return

        if self.on_segment_callback:
            try:
                self.on_segment_callback(segment)
            except Exception as e:
                print(f"[LiveTranscriber] Notice in segment callback: {e}")

    def _write_temp_chunk(self, pcm_data: np.ndarray) -> str:
        """Write 16kHz mono int16 numpy array to a temporary WAV file."""
        temp_file = tempfile.NamedTemporaryFile(
            prefix="live_chunk_",
            suffix=".wav",
            delete=False,
        )
        temp_path = os.path.abspath(temp_file.name)
        temp_file.close()

        # Handle float32 normalized samples or raw int16
        if pcm_data.dtype in (np.float32, np.float64):
            max_val = np.max(np.abs(pcm_data)) if len(pcm_data) > 0 else 0.0
            if max_val <= 1.5:
                int16_arr = np.clip(pcm_data * 32767.0, -32768, 32767).astype(np.int16)
            else:
                int16_arr = np.clip(pcm_data, -32768, 32767).astype(np.int16)
        else:
            int16_arr = pcm_data.astype(np.int16)

        int16_bytes = int16_arr.tobytes()
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16_bytes)

        return temp_path


class LiveAudioChunker:
    """
    Slices a continuous stream of 16 kHz mono audio samples into
    overlapping live chunks (e.g. 5.0s duration, 0.8s overlap) and
    dispatches them to the LiveTranscriber queue.
    """

    def __init__(
        self,
        transcriber: LiveTranscriber,
        sample_rate: int = 16000,
        chunk_seconds: float = 5.0,
        overlap_seconds: float = 0.8,
    ):
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

        self.chunk_samples = int(chunk_seconds * sample_rate)
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples

        self._buffer: List[float] = []
        self._lock = threading.Lock()
        self._timeline_sample_offset = 0

    def feed_samples(self, samples: np.ndarray, is_paused: bool = False) -> None:
        """
        Feed newly captured 16kHz mono samples into the chunker.
        Automatically emits overlapping chunks when buffer threshold is reached.
        """
        if is_paused or len(samples) == 0:
            return

        with self._lock:
            self._buffer.extend(samples.tolist())

            while len(self._buffer) >= self.chunk_samples:
                chunk_data = np.array(self._buffer[:self.chunk_samples], dtype=np.float32)

                start_sec = self._timeline_sample_offset / self.sample_rate
                end_sec = start_sec + self.chunk_seconds

                self.transcriber.enqueue_chunk(
                    pcm_data=chunk_data,
                    start_time=start_sec,
                    end_time=end_sec,
                    is_final=False,
                )

                # Slide window forward by step_samples (preserving overlap)
                self._buffer = self._buffer[self.step_samples:]
                self._timeline_sample_offset += self.step_samples

    def flush(self) -> None:
        """
        Flush any remaining unchunked samples marked as is_final=True on recording stop.
        Pads minimal silence if buffer is short (between 0.1s and 0.5s) to ensure
        the final spoken words are captured.
        """
        with self._lock:
            min_samples = int(0.1 * self.sample_rate)  # 0.1s threshold (1600 samples)
            if len(self._buffer) >= min_samples:
                chunk_data = np.array(self._buffer, dtype=np.float32)
                pad_target = int(0.5 * self.sample_rate)
                if len(chunk_data) < pad_target:
                    pad_len = pad_target - len(chunk_data)
                    chunk_data = np.pad(chunk_data, (0, pad_len), mode="constant")

                start_sec = self._timeline_sample_offset / self.sample_rate
                end_sec = start_sec + (len(self._buffer) / self.sample_rate)

                self.transcriber.enqueue_chunk(
                    pcm_data=chunk_data,
                    start_time=start_sec,
                    end_time=end_sec,
                    is_final=True,
                )
            self._buffer.clear()

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._timeline_sample_offset = 0
