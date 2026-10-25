"""
Comprehensive Test Suite for Live Audio Recorder (Phase 4).

Tests state machine controls, state transitions, audio output format,
dual-stream mixing, time tracking, error handling, pipeline compatibility,
and hardware probes (clearly distinguishing simulated vs real hardware).
"""

import os
import sys
import time
import wave
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.audio_recorder import (
    AudioRecorder,
    RecordingState,
    AudioSource,
    InvalidStateError,
    DeviceNotFoundError,
    query_available_devices,
    HAS_PYAUDIO,
)
from utils.audio_processor import convert_to_wav, chunk_audio


class TestAudioRecorderSimulated:
    """Simulated and unit tests for AudioRecorder controls, mixing, and state machine."""

    def test_01_state_transitions_lifecycle(self):
        """Verify full Start -> Pause -> Resume -> Stop -> Reset lifecycle."""
        recorder = AudioRecorder(mock_mode=True)
        assert recorder.state == RecordingState.IDLE

        # 1. Start recording
        recorder.start(source=AudioSource.BOTH)
        assert recorder.state == RecordingState.RECORDING
        time.sleep(0.15)

        # 2. Pause
        recorder.pause()
        assert recorder.state == RecordingState.PAUSED
        elapsed_at_pause = recorder.get_elapsed_seconds()
        assert elapsed_at_pause > 0.05
        time.sleep(0.15)

        # 3. Resume
        recorder.resume()
        assert recorder.state == RecordingState.RECORDING
        time.sleep(0.15)

        # 4. Stop
        wav_path = recorder.stop()
        assert recorder.state == RecordingState.STOPPED
        assert os.path.isfile(wav_path)
        assert os.path.getsize(wav_path) > 100

        # Verify WAV format specifications (16kHz, 16-bit, mono)
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono (1 channel)"
            assert wf.getsampwidth() == 2, "Expected 16-bit (2 bytes per sample)"
            assert wf.getframerate() == 16000, "Expected 16,000 Hz sample rate"
            assert wf.getnframes() > 0, "Expected non-empty audio frames"

        # 5. Reset
        recorder.reset()
        assert recorder.state == RecordingState.IDLE
        assert not os.path.exists(wav_path), "Reset should clean up temporary recording file"

    def test_02_invalid_state_transitions(self):
        """Verify that improper state transitions raise InvalidStateError."""
        recorder = AudioRecorder(mock_mode=True)

        # Cannot pause or resume when idle
        with pytest.raises(InvalidStateError):
            recorder.pause()

        with pytest.raises(InvalidStateError):
            recorder.resume()

        with pytest.raises(InvalidStateError):
            recorder.stop()

        # Start recording
        recorder.start(source=AudioSource.MICROPHONE)

        # Cannot start again when recording
        with pytest.raises(InvalidStateError):
            recorder.start()

        # Cannot resume when already recording
        with pytest.raises(InvalidStateError):
            recorder.resume()

        # Pause
        recorder.pause()

        # Cannot start when paused
        with pytest.raises(InvalidStateError):
            recorder.start()

        # Cannot pause again when already paused
        with pytest.raises(InvalidStateError):
            recorder.pause()

        # Stop
        wav_path = recorder.stop()

        # Cannot pause or resume when stopped
        with pytest.raises(InvalidStateError):
            recorder.pause()

        with pytest.raises(InvalidStateError):
            recorder.resume()

        # Cleanup
        recorder.reset()

    def test_03_elapsed_time_excludes_pauses(self):
        """Verify that pauses do not artificially inflate recording duration."""
        recorder = AudioRecorder(mock_mode=True)
        recorder.start(source=AudioSource.MICROPHONE)

        time.sleep(0.2)
        recorder.pause()
        paused_duration = recorder.get_elapsed_seconds()

        # Sleep during pause
        time.sleep(0.3)
        assert abs(recorder.get_elapsed_seconds() - paused_duration) < 0.05, (
            "Elapsed time should not advance while in PAUSED state"
        )

        recorder.resume()
        time.sleep(0.2)
        total_duration = recorder.get_elapsed_seconds()

        recorder.stop()
        recorder.reset()

        # Active time was ~0.4s, while wall-clock time was ~0.7s
        assert 0.3 <= total_duration <= 0.6, f"Expected active duration ~0.4s, got {total_duration:.2f}s"

    def test_04_audio_mixing_and_headroom_safety(self):
        """Verify that mixing two maximum-amplitude signals does not clip or overflow."""
        recorder = AudioRecorder(mock_mode=True)

        samples = 1600
        # Maximum amplitude 16-bit signals (+32767)
        stream_a = np.full(samples, 32767.0, dtype=np.float32)
        stream_b = np.full(samples, 32767.0, dtype=np.float32)

        mixed = recorder._mix_audio_streams(stream_a, stream_b)

        assert len(mixed) == samples
        assert np.max(mixed) <= 32767.0, "Mixed audio must not exceed maximum 16-bit amplitude"
        assert np.min(mixed) >= -32768.0, "Mixed audio must not fall below minimum 16-bit amplitude"

    def test_05_resampling_different_input_rates(self):
        """Verify resampling from 48kHz and 44.1kHz to 16kHz produces expected length."""
        recorder = AudioRecorder(target_sample_rate=16000, mock_mode=True)

        # 48 kHz (0.5s = 24,000 samples)
        sig_48k = np.zeros(24000, dtype=np.int16)
        res_16k_from_48k = recorder._resample_to_target(sig_48k, 48000)
        assert len(res_16k_from_48k) == 8000, "0.5s at 16kHz should equal 8,000 samples"

        # 44.1 kHz (1.0s = 44,100 samples)
        sig_44k = np.zeros(44100, dtype=np.int16)
        res_16k_from_44k = recorder._resample_to_target(sig_44k, 44100)
        assert len(res_16k_from_44k) == 16000, "1.0s at 16kHz should equal 16,000 samples"

    def test_06_pipeline_compatibility(self):
        """Verify that AudioRecorder WAV output is directly readable by the audio processing pipeline."""
        recorder = AudioRecorder(mock_mode=True)
        recorder.start(source=AudioSource.BOTH)
        time.sleep(0.3)
        wav_path = recorder.stop()

        try:
            # 1. Pipeline convert_to_wav
            converted = convert_to_wav(wav_path, force_reconvert=True)
            assert os.path.exists(converted)

            # 2. Pipeline chunk_audio
            chunks = chunk_audio(converted, chunk_minutes=1, force_rechunk=True)
            assert len(chunks) >= 1
            assert os.path.exists(chunks[0])
        finally:
            recorder.reset()
            if os.path.exists(converted):
                os.remove(converted)


class TestAudioRecorderHardware:
    """Hardware probe tests for actual physical devices (distinguished from simulated)."""

    def test_07_device_enumeration_structure(self):
        """Verify query_available_devices returns valid device catalogs."""
        devices = query_available_devices()
        assert isinstance(devices, dict)
        assert "microphones" in devices
        assert "system_loopback" in devices
        assert isinstance(devices["microphones"], list)
        assert isinstance(devices["system_loopback"], list)

    @pytest.mark.skipif(not HAS_PYAUDIO, reason="pyaudiowpatch not installed")
    def test_08_real_hardware_microphone_probe(self):
        """Live hardware probe: capture a short snippet from real microphone if present."""
        devices = query_available_devices()
        if not devices["microphones"]:
            pytest.skip("No physical microphone available on this host")

        recorder = AudioRecorder(mock_mode=False)
        try:
            recorder.start(source=AudioSource.MICROPHONE)
            time.sleep(0.3)
            wav_path = recorder.stop()

            assert os.path.isfile(wav_path)
            assert os.path.getsize(wav_path) > 500

            with wave.open(wav_path, "rb") as wf:
                assert wf.getframerate() == 16000
                assert wf.getnchannels() == 1
        finally:
            recorder.reset()

    @pytest.mark.skipif(not HAS_PYAUDIO, reason="pyaudiowpatch not installed")
    def test_09_real_hardware_loopback_probe(self):
        """Live hardware probe: open and record from Windows WASAPI loopback device."""
        devices = query_available_devices()
        if not devices["system_loopback"]:
            pytest.skip("No WASAPI loopback device detected on this host")

        recorder = AudioRecorder(mock_mode=False)
        try:
            recorder.start(source=AudioSource.SYSTEM)
            time.sleep(0.3)
            wav_path = recorder.stop()

            assert os.path.isfile(wav_path)
            assert os.path.getsize(wav_path) > 100
        finally:
            recorder.reset()
