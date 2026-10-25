import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.transcriber import (
    resolve_language_mode,
    get_device_and_compute,
    load_model,
    load_faster_whisper_model,
    load_openai_whisper_model,
    transcribe_chunk_whisper,
    transcribe_chunk,
    transcribe_all,
    validate_whisper_model,
    _model_lock,
)


class DummySegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class TestPhase2TranscriptionOptimization(unittest.TestCase):
    """
    Comprehensive test suite for Phase 2 Speech Transcription optimization:
    faster-whisper, CPU INT8, GPU fallback, singleton caching, segment parsing,
    timestamps, and language preservation.
    """

    def test_01_device_and_compute_detection(self):
        """Verify hardware accelerator detection and compute_type configuration."""
        # 1. When CUDA is available
        with patch("torch.cuda.is_available", return_value=True):
            device, compute = get_device_and_compute()
            self.assertEqual(device, "cuda")
            self.assertEqual(compute, "float16")

        # 2. When CUDA is NOT available (CPU INT8 fallback)
        with patch("torch.cuda.is_available", return_value=False):
            device, compute = get_device_and_compute()
            self.assertEqual(device, "cpu")
            self.assertEqual(compute, "int8")

        # 3. Environment overrides
        with patch.dict(os.environ, {"WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE_TYPE": "int8"}):
            device, compute = get_device_and_compute()
            self.assertEqual(device, "cpu")
            self.assertEqual(compute, "int8")

    def test_02_singleton_model_caching(self):
        """Verify that models are loaded once and strictly reused across requests."""
        import core.transcriber as t_mod

        # Reset singleton cache for test
        old_faster = t_mod._faster_model
        try:
            mock_fw_instance = MagicMock()
            mock_fw_class = MagicMock(return_value=mock_fw_instance)

            with patch("core.transcriber._faster_model", None), \
                 patch("core.transcriber.WhisperModel", mock_fw_class, create=True), \
                 patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_fw_class)}):

                # First call loads model
                m1 = load_faster_whisper_model("small")
                self.assertIs(m1, mock_fw_instance)
                self.assertEqual(mock_fw_class.call_count, 1)

                # Second call reuses cached instance
                m2 = load_faster_whisper_model("small")
                self.assertIs(m2, mock_fw_instance)
                # Call count MUST still be 1 (no reload!)
                self.assertEqual(mock_fw_class.call_count, 1)
        finally:
            t_mod._faster_model = old_faster

    def test_03_faster_whisper_tuple_transcription_and_timestamps(self):
        """Verify parsing of faster-whisper (segments, info) tuple output with timestamps."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True

        dummy_segments = [
            DummySegment(0.0, 1.5, "Welcome to the meeting."),
            DummySegment(1.5, 3.2, "We have three agenda items."),
        ]
        dummy_info = MagicMock(language="en")

        mock_model.transcribe.return_value = (iter(dummy_segments), dummy_info)

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            # Test text return
            text = transcribe_chunk_whisper("test_chunk.wav", language="en", task="transcribe")
            self.assertEqual(text, "Welcome to the meeting. We have three agenda items.")

            # Test segments and timestamps return
            mock_model.transcribe.return_value = (iter(dummy_segments), dummy_info)
            res = transcribe_chunk_whisper("test_chunk.wav", language="en", task="transcribe", return_segments=True)
            self.assertEqual(res["text"], "Welcome to the meeting. We have three agenda items.")
            self.assertEqual(len(res["segments"]), 2)
            self.assertEqual(res["segments"][0]["start"], 0.0)
            self.assertEqual(res["segments"][0]["end"], 1.5)
            self.assertEqual(res["segments"][1]["start"], 1.5)
            self.assertEqual(res["segments"][1]["end"], 3.2)

    def test_04_openai_whisper_fallback_compatibility(self):
        """Verify seamless backward-compatibility when dict output is returned."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True
        mock_model.transcribe.return_value = {
            "text": "Fallback text from standard Whisper.",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Fallback text"}],
        }

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            text = transcribe_chunk_whisper("test_chunk.wav", language="en", task="transcribe")
            self.assertEqual(text, "Fallback text from standard Whisper.")

    def test_05_nepali_language_and_task_modes(self):
        """Verify task='translate' and task='transcribe' routing for Nepali."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True

        dummy_ne_transcribe = [DummySegment(0.0, 1.0, "नमस्ते")]
        dummy_ne_translate = [DummySegment(0.0, 1.0, "Hello")]

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            # 1. Nepali transcription in Devanagari
            mock_model.transcribe.return_value = (iter(dummy_ne_transcribe), MagicMock())
            text_ne = transcribe_chunk("test.wav", language="nepali")
            mock_model.transcribe.assert_called_with("test.wav", task="transcribe", language="ne")
            self.assertEqual(text_ne, "नमस्ते")

            # 2. Nepali translation to English
            mock_model.transcribe.return_value = (iter(dummy_ne_translate), MagicMock())
            text_en = transcribe_chunk("test.wav", language="nepali-to-english")
            mock_model.transcribe.assert_called_with("test.wav", task="translate", language="ne")
            self.assertEqual(text_en, "Hello")

    def test_06_chunk_timestamp_offset_calculation(self):
        """Verify that chunk offsets accurately accumulate across consecutive chunks."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True

        def fake_transcribe(chunk_path, **kwargs):
            if "chunk_0" in chunk_path:
                return (iter([DummySegment(0.0, 10.0, "Chunk 0 part")]), MagicMock())
            elif "chunk_1" in chunk_path:
                return (iter([DummySegment(5.0, 15.0, "Chunk 1 part")]), MagicMock())
            return (iter([]), MagicMock())

        mock_model.transcribe.side_effect = fake_transcribe

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            # 180s per chunk
            chunks = ["path_chunk_0.wav", "path_chunk_1.wav"]
            res = transcribe_all(chunks, language="english", return_original=True, chunk_duration_seconds=180.0)

            segments = res["segments"]
            self.assertEqual(len(segments), 2)
            # Chunk 0 offset is 0.0 -> segment start=0.0, end=10.0
            self.assertEqual(segments[0]["start"], 0.0)
            self.assertEqual(segments[0]["end"], 10.0)
            # Chunk 1 offset is 180.0 -> segment start=185.0, end=195.0
            self.assertEqual(segments[1]["start"], 185.0)
            self.assertEqual(segments[1]["end"], 195.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
