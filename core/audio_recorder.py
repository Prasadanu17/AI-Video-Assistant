"""
Audio Recorder Module for AI Video Assistant.

Provides production-grade audio recording with:
- Start, Pause, Resume, Stop state machine.
- Microphone input and Windows WASAPI system-audio loopback capture.
- Intelligent dual-channel mixing with clipping prevention.
- Seamless export to 16 kHz mono WAV compatible with Whisper.
- Hardware device detection, permission handling, and simulated mock mode for tests.
"""

import enum
import math
import os
import tempfile
import threading
import time
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal

# Safe import of pyaudiowpatch with graceful fallback
try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    try:
        import pyaudio
        HAS_PYAUDIO = True
    except ImportError:
        HAS_PYAUDIO = False


class RecordingState(str, enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


class AudioSource(str, enum.Enum):
    MICROPHONE = "microphone"
    SYSTEM = "system"
    BOTH = "both"


class AudioRecorderError(Exception):
    """Base exception for all audio recording errors."""
    pass


class InvalidStateError(AudioRecorderError):
    """Raised when an invalid state transition is attempted."""
    pass


class DeviceNotFoundError(AudioRecorderError):
    """Raised when a requested audio input or loopback device is unavailable."""
    pass


class PermissionOrBusyError(AudioRecorderError):
    """Raised when device access is denied or exclusively locked by another app."""
    pass


def query_available_devices() -> Dict[str, List[dict]]:
    """
    Query system for available microphones and Windows WASAPI loopback devices.
    Returns:
        dict with "microphones" and "system_loopback" lists of device info dicts.
    """
    if not HAS_PYAUDIO:
        return {"microphones": [], "system_loopback": []}

    microphones = []
    loopback_devices = []

    try:
        p = pyaudio.PyAudio()
        try:
            device_count = p.get_device_count()
            for i in range(device_count):
                try:
                    info = p.get_device_info_by_index(i)
                except Exception:
                    continue

                # Check if it's a loopback device
                if info.get("isLoopbackDevice", False) or "[loopback]" in info.get("name", "").lower():
                    loopback_devices.append({
                        "index": info["index"],
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "host_api": p.get_host_api_info_by_index(info["hostApi"])["name"],
                    })
                elif info.get("maxInputChannels", 0) > 0:
                    microphones.append({
                        "index": info["index"],
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "host_api": p.get_host_api_info_by_index(info["hostApi"])["name"],
                    })
        finally:
            p.terminate()
    except Exception as e:
        print(f"[AudioRecorder] Notice during device enumeration: {e}")

    return {
        "microphones": microphones,
        "system_loopback": loopback_devices,
    }


class AudioRecorder:
    """
    Thread-safe live audio recorder supporting Microphone and Windows WASAPI Loopback.
    """

    TARGET_SAMPLE_RATE = 16000  # 16 kHz standard for Whisper models
    FRAMES_PER_BUFFER = 1024

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        mock_mode: bool = False,
    ):
        self.target_sample_rate = target_sample_rate
        self.mock_mode = mock_mode

        self._lock = threading.RLock()
        self._state = RecordingState.IDLE
        self._source = AudioSource.MICROPHONE

        self._pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self._mic_stream = None
        self._loopback_stream = None

        self._mic_frames: List[bytes] = []
        self._loopback_frames: List[bytes] = []

        self._mic_sample_rate = 16000
        self._mic_channels = 1
        self._lb_sample_rate = 48000
        self._lb_channels = 2

        self._start_time: float = 0.0
        self._paused_accumulated_time: float = 0.0
        self._pause_start_time: float = 0.0

        self._output_wav_path: Optional[str] = None
        self._mock_thread: Optional[threading.Thread] = None
        self._mock_stop_event = threading.Event()

        # Diagnostic audio energy meters (0.0 to 1.0)
        self._mic_peak: float = 0.0
        self._loopback_peak: float = 0.0

        # Optional live audio chunker for real-time transcription
        self._live_chunker = None

    def attach_live_chunker(self, chunker) -> None:
        """Attach a LiveAudioChunker to receive live 16kHz audio frames in real time."""
        with self._lock:
            self._live_chunker = chunker

    @property
    def state(self) -> RecordingState:
        with self._lock:
            return self._state

    @property
    def source(self) -> AudioSource:
        with self._lock:
            return self._source

    def get_elapsed_seconds(self) -> float:
        """Return total elapsed recording duration in seconds, excluding pauses."""
        with self._lock:
            if self._state == RecordingState.IDLE:
                return 0.0
            if self._state == RecordingState.RECORDING:
                return (time.time() - self._start_time) - self._paused_accumulated_time
            if self._state == RecordingState.PAUSED:
                return (self._pause_start_time - self._start_time) - self._paused_accumulated_time
            if self._state == RecordingState.STOPPED:
                return self._paused_accumulated_time  # stored total duration at stop

        return 0.0

    def get_peak_levels(self) -> Tuple[float, float]:
        """Return the latest normalized peak levels (0.0 - 1.0) for (microphone, system)."""
        with self._lock:
            return self._mic_peak, self._loopback_peak

    def start(
        self,
        source: AudioSource = AudioSource.MICROPHONE,
        mic_index: Optional[int] = None,
        loopback_index: Optional[int] = None,
    ) -> None:
        """
        Start recording from the specified audio source.
        """
        with self._lock:
            if self._state != RecordingState.IDLE:
                raise InvalidStateError(
                    f"Cannot start recording while in '{self._state.value}' state. "
                    "Reset or stop existing session first."
                )

            self._source = source
            self._mic_frames = []
            self._loopback_frames = []
            self._paused_accumulated_time = 0.0
            self._mic_peak = 0.0
            self._loopback_peak = 0.0
            self._output_wav_path = None

            if self.mock_mode or not HAS_PYAUDIO:
                self._start_mock_recording()
            else:
                self._start_hardware_recording(source, mic_index, loopback_index)

            self._start_time = time.time()
            self._state = RecordingState.RECORDING
            print(f"[AudioRecorder] Started recording ({source.value}).")

    def pause(self) -> None:
        """Pause active recording. Incoming audio during pause will not be recorded."""
        with self._lock:
            if self._state != RecordingState.RECORDING:
                raise InvalidStateError(f"Cannot pause recording from '{self._state.value}' state.")

            self._pause_start_time = time.time()
            self._state = RecordingState.PAUSED
            print(f"[AudioRecorder] Paused at {self.get_elapsed_seconds():.1f}s.")

    def resume(self) -> None:
        """Resume recording from paused state."""
        with self._lock:
            if self._state != RecordingState.PAUSED:
                raise InvalidStateError(f"Cannot resume recording from '{self._state.value}' state.")

            pause_duration = time.time() - self._pause_start_time
            self._paused_accumulated_time += pause_duration
            self._state = RecordingState.RECORDING
            print(f"[AudioRecorder] Resumed recording (paused for {pause_duration:.1f}s).")

    def stop(self) -> str:
        """
        Stop recording, finalize audio streams, downmix/resample, and save to WAV.
        Returns the absolute filepath to the created 16 kHz mono WAV file.
        """
        with self._lock:
            if self._state not in (RecordingState.RECORDING, RecordingState.PAUSED):
                raise InvalidStateError(f"Cannot stop recording from '{self._state.value}' state.")

            total_duration = self.get_elapsed_seconds()

            # 1. Terminate active capture streams
            if self.mock_mode or not HAS_PYAUDIO:
                self._stop_mock_recording()
            else:
                self._stop_hardware_recording()

            # Flush any remaining live chunks
            if self._live_chunker:
                try:
                    self._live_chunker.flush()
                except Exception:
                    pass

            # 2. Process and mix audio
            wav_path = self._process_and_export_wav()
            self._output_wav_path = wav_path
            self._paused_accumulated_time = total_duration  # retain total time
            self._state = RecordingState.STOPPED
            print(f"[AudioRecorder] Stopped. Saved {total_duration:.1f}s of audio to: {wav_path}")
            return wav_path

    def reset(self) -> None:
        """Reset recorder to IDLE state and cleanup temporary files."""
        with self._lock:
            if self._state in (RecordingState.RECORDING, RecordingState.PAUSED):
                try:
                    self.stop()
                except Exception:
                    pass

            # Remove generated WAV file if it exists and wasn't consumed
            if self._output_wav_path and os.path.exists(self._output_wav_path):
                try:
                    os.remove(self._output_wav_path)
                except Exception:
                    pass

            self._output_wav_path = None
            self._mic_frames = []
            self._loopback_frames = []
            if self._live_chunker:
                try:
                    self._live_chunker.reset()
                except Exception:
                    pass
            self._state = RecordingState.IDLE
            print("[AudioRecorder] Reset to IDLE.")

    # ----------------------------------------------------------------------
    # Hardware Recording Implementation (pyaudiowpatch)
    # ----------------------------------------------------------------------
    def _start_hardware_recording(
        self,
        source: AudioSource,
        mic_index: Optional[int],
        loopback_index: Optional[int],
    ) -> None:
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
        except Exception as e:
            raise AudioRecorderError(f"Failed to initialize audio subsystem: {e}")

        # 1. Microphone setup
        if source in (AudioSource.MICROPHONE, AudioSource.BOTH):
            mic_info = self._resolve_mic_device(mic_index)
            self._mic_channels = max(1, min(2, mic_info.get("maxInputChannels", 1)))
            self._mic_sample_rate = int(mic_info.get("defaultSampleRate", 16000))

            def _mic_callback(in_data, frame_count, time_info, status):
                with self._lock:
                    if self._state == RecordingState.RECORDING:
                        self._mic_frames.append(in_data)
                        self._update_meter(in_data, is_mic=True)
                        if self._live_chunker:
                            try:
                                arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
                                if self._mic_channels > 1:
                                    arr = arr.reshape(-1, self._mic_channels).mean(axis=1)
                                if self._mic_sample_rate != self.target_sample_rate:
                                    arr = self._resample_to_target(arr, self._mic_sample_rate)
                                self._live_chunker.feed_samples(arr, is_paused=False)
                            except Exception:
                                pass
                return (None, pyaudio.paContinue)

            try:
                self._mic_stream = self._pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self._mic_channels,
                    rate=self._mic_sample_rate,
                    input=True,
                    input_device_index=mic_info["index"],
                    frames_per_buffer=self.FRAMES_PER_BUFFER,
                    stream_callback=_mic_callback,
                )
                self._mic_stream.start_stream()
            except Exception as e:
                self._cleanup_streams()
                raise PermissionOrBusyError(
                    f"Could not open microphone '{mic_info['name']}': {e}. "
                    "Ensure microphone permissions are granted in Windows Settings."
                )

        # 2. System Loopback setup
        if source in (AudioSource.SYSTEM, AudioSource.BOTH):
            lb_info = self._resolve_loopback_device(loopback_index)
            self._lb_channels = max(1, lb_info.get("maxInputChannels", 2))
            self._lb_sample_rate = int(lb_info.get("defaultSampleRate", 48000))

            def _lb_callback(in_data, frame_count, time_info, status):
                with self._lock:
                    if self._state == RecordingState.RECORDING:
                        self._loopback_frames.append(in_data)
                        self._update_meter(in_data, is_mic=False)
                        if self._live_chunker and self._source == AudioSource.SYSTEM:
                            try:
                                arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
                                if self._lb_channels > 1:
                                    arr = arr.reshape(-1, self._lb_channels).mean(axis=1)
                                if self._lb_sample_rate != self.target_sample_rate:
                                    arr = self._resample_to_target(arr, self._lb_sample_rate)
                                self._live_chunker.feed_samples(arr, is_paused=False)
                            except Exception:
                                pass
                return (None, pyaudio.paContinue)

            try:
                self._loopback_stream = self._pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self._lb_channels,
                    rate=self._lb_sample_rate,
                    input=True,
                    input_device_index=lb_info["index"],
                    frames_per_buffer=self.FRAMES_PER_BUFFER,
                    stream_callback=_lb_callback,
                )
                self._loopback_stream.start_stream()
            except Exception as e:
                self._cleanup_streams()
                raise DeviceNotFoundError(
                    f"Could not open system loopback device '{lb_info['name']}': {e}. "
                    "Ensure Windows WASAPI audio is enabled and speakers are connected."
                )

    def _resolve_mic_device(self, mic_index: Optional[int]) -> dict:
        if mic_index is not None:
            try:
                return self._pyaudio_instance.get_device_info_by_index(mic_index)
            except Exception as e:
                raise DeviceNotFoundError(f"Requested microphone index {mic_index} is invalid: {e}")

        try:
            return self._pyaudio_instance.get_default_input_device_info()
        except Exception:
            # Fallback search for any input-capable device
            for i in range(self._pyaudio_instance.get_device_count()):
                info = self._pyaudio_instance.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice", False):
                    return info
        raise DeviceNotFoundError("No working microphone device detected on this system.")

    def _resolve_loopback_device(self, loopback_index: Optional[int]) -> dict:
        if loopback_index is not None:
            try:
                return self._pyaudio_instance.get_device_info_by_index(loopback_index)
            except Exception as e:
                raise DeviceNotFoundError(f"Requested loopback index {loopback_index} is invalid: {e}")

        # Attempt to find the WASAPI loopback device corresponding to default output speakers
        try:
            wasapi_info = self._pyaudio_instance.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = self._pyaudio_instance.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            default_name = default_out.get("name", "")

            if hasattr(self._pyaudio_instance, "get_loopback_device_info_generator"):
                for lb in self._pyaudio_instance.get_loopback_device_info_generator():
                    if default_name in lb.get("name", ""):
                        return lb
                # If specific match not found, take first available loopback
                for lb in self._pyaudio_instance.get_loopback_device_info_generator():
                    return lb
        except Exception as e:
            print(f"[AudioRecorder] Notice during loopback resolution: {e}")

        # Fallback search by string name
        for i in range(self._pyaudio_instance.get_device_count()):
            info = self._pyaudio_instance.get_device_info_by_index(i)
            if info.get("isLoopbackDevice", False) or "[loopback]" in info.get("name", "").lower():
                return info

        raise DeviceNotFoundError(
            "Windows WASAPI loopback audio device was not found. "
            "System audio recording requires an active audio output endpoint (e.g. Speakers or Headphones)."
        )

    def _update_meter(self, raw_bytes: bytes, is_mic: bool) -> None:
        if not raw_bytes:
            return
        try:
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            if len(samples) > 0:
                peak = float(np.max(np.abs(samples)) / 32768.0)
                if is_mic:
                    self._mic_peak = peak
                else:
                    self._loopback_peak = peak
        except Exception:
            pass

    def _cleanup_streams(self) -> None:
        if self._mic_stream:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        if self._loopback_stream:
            try:
                self._loopback_stream.stop_stream()
                self._loopback_stream.close()
            except Exception:
                pass
            self._loopback_stream = None

        if self._pyaudio_instance:
            try:
                self._pyaudio_instance.terminate()
            except Exception:
                pass
            self._pyaudio_instance = None

    def _stop_hardware_recording(self) -> None:
        self._cleanup_streams()

    # ----------------------------------------------------------------------
    # Mock / Simulation Implementation (Headless & Testing)
    # ----------------------------------------------------------------------
    def _start_mock_recording(self) -> None:
        self._mock_stop_event.clear()
        self._mic_sample_rate = self.target_sample_rate
        self._mic_channels = 1
        self._lb_sample_rate = self.target_sample_rate
        self._lb_channels = 1

        def _mock_worker():
            chunk_samples = 800  # 50ms at 16kHz
            step = 0
            while not self._mock_stop_event.is_set():
                with self._lock:
                    if self._state == RecordingState.RECORDING:
                        t = (np.arange(chunk_samples) + step * chunk_samples) / self.target_sample_rate
                        # Mic: 440 Hz tone
                        mic_sig = (0.25 * np.sin(2 * math.pi * 440 * t) * 32767).astype(np.int16)
                        self._mic_frames.append(mic_sig.tobytes())
                        self._mic_peak = 0.25

                        # Loopback: 880 Hz tone
                        lb_sig = (0.35 * np.sin(2 * math.pi * 880 * t) * 32767).astype(np.int16)
                        self._loopback_frames.append(lb_sig.tobytes())
                        self._loopback_peak = 0.35

                        if self._live_chunker:
                            try:
                                self._live_chunker.feed_samples(mic_sig.astype(np.float32), is_paused=False)
                            except Exception:
                                pass
                step += 1
                time.sleep(0.05)

        self._mock_thread = threading.Thread(target=_mock_worker, daemon=True)
        self._mock_thread.start()

    def _stop_mock_recording(self) -> None:
        if self._mock_thread:
            self._mock_stop_event.set()
            self._mock_thread.join(timeout=2.0)
            self._mock_thread = None

    # ----------------------------------------------------------------------
    # Audio Processing & Mixing
    # ----------------------------------------------------------------------
    def _process_and_export_wav(self) -> str:
        """
        Convert buffered chunks into a normalized 16 kHz 16-bit mono PCM WAV file.
        """
        has_mic = len(self._mic_frames) > 0 and self._source in (AudioSource.MICROPHONE, AudioSource.BOTH)
        has_lb = len(self._loopback_frames) > 0 and self._source in (AudioSource.SYSTEM, AudioSource.BOTH)

        if not has_mic and not has_lb:
            # If completely silent or empty, generate a minimal 1-second silence file
            silence = np.zeros(self.target_sample_rate, dtype=np.int16)
            return self._write_wav_file(silence)

        mic_pcm = None
        if has_mic:
            raw_mic = b"".join(self._mic_frames)
            mic_arr = np.frombuffer(raw_mic, dtype=np.int16)
            if self._mic_channels > 1:
                mic_arr = mic_arr.reshape(-1, self._mic_channels).mean(axis=1)
            mic_pcm = self._resample_to_target(mic_arr, self._mic_sample_rate)

        lb_pcm = None
        if has_lb:
            raw_lb = b"".join(self._loopback_frames)
            lb_arr = np.frombuffer(raw_lb, dtype=np.int16)
            if self._lb_channels > 1:
                lb_arr = lb_arr.reshape(-1, self._lb_channels).mean(axis=1)
            lb_pcm = self._resample_to_target(lb_arr, self._lb_sample_rate)

        if mic_pcm is not None and lb_pcm is not None:
            # Mix both streams
            mixed_pcm = self._mix_audio_streams(mic_pcm, lb_pcm)
        elif mic_pcm is not None:
            mixed_pcm = mic_pcm
        else:
            mixed_pcm = lb_pcm

        # Ensure valid integer range
        final_int16 = np.clip(mixed_pcm, -32768, 32767).astype(np.int16)
        return self._write_wav_file(final_int16)

    def _resample_to_target(self, signal: np.ndarray, orig_rate: int) -> np.ndarray:
        if orig_rate == self.target_sample_rate or len(signal) == 0:
            return signal.astype(np.float32)

        gcd = math.gcd(orig_rate, self.target_sample_rate)
        up = self.target_sample_rate // gcd
        down = orig_rate // gcd
        return scipy.signal.resample_poly(signal.astype(np.float32), up, down)

    def _mix_audio_streams(self, stream_a: np.ndarray, stream_b: np.ndarray) -> np.ndarray:
        """
        Harmoniously blend two audio streams (e.g. microphone voice + speaker meeting audio).
        Applies length alignment, normalization headroom, and soft limiting to eliminate clipping.
        """
        len_a = len(stream_a)
        len_b = len(stream_b)
        max_len = max(len_a, len_b)

        padded_a = np.zeros(max_len, dtype=np.float32)
        padded_b = np.zeros(max_len, dtype=np.float32)

        padded_a[:len_a] = stream_a
        padded_b[:len_b] = stream_b

        # 0.5 headroom scaling on each channel to guarantee the mix never hard-clips
        mixed = 0.5 * padded_a + 0.5 * padded_b

        # Peak normalization if signals are loud
        peak = np.max(np.abs(mixed))
        if peak > 30000.0:
            mixed = (mixed / peak) * 30000.0

        return mixed

    def _write_wav_file(self, int16_data: np.ndarray) -> str:
        temp_file = tempfile.NamedTemporaryFile(
            prefix="meeting_recording_",
            suffix=".wav",
            delete=False,
        )
        temp_path = os.path.abspath(temp_file.name)
        temp_file.close()

        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit (2 bytes)
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(int16_data.tobytes())

        return temp_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
