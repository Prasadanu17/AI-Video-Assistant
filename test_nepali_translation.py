import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.transcriber import (
    resolve_language_mode,
    transcribe_chunk_whisper,
    transcribe_chunk,
    transcribe_all,
    translate_nepali_to_english,
    transcribe_nepali,
    transcribe_nepali_dual,
    validate_whisper_model,
)


class TestNepaliTranslationUnit(unittest.TestCase):
    """
    Unit tests for Nepali transcription, Nepali-to-English translation,
    English mode, Hinglish routing, chunk ordering, and error safety.
    """

    def test_01_resolve_language_modes(self):
        """Verify that UI labels and aliases resolve to the exact expected engine, code, and task."""
        # 1. Nepali transcription
        cfg_ne = resolve_language_mode("Nepali transcription")
        self.assertEqual(cfg_ne["engine"], "whisper")
        self.assertEqual(cfg_ne["whisper_code"], "ne")
        self.assertEqual(cfg_ne["task"], "transcribe")

        # 2. Nepali speech -> English text
        cfg_ne_en = resolve_language_mode("Nepali speech → English text")
        self.assertEqual(cfg_ne_en["engine"], "whisper")
        self.assertEqual(cfg_ne_en["whisper_code"], "ne")
        self.assertEqual(cfg_ne_en["task"], "translate")

        # 3. English transcription
        cfg_en = resolve_language_mode("English transcription")
        self.assertEqual(cfg_en["engine"], "whisper")
        self.assertEqual(cfg_en["whisper_code"], "en")
        self.assertEqual(cfg_en["task"], "transcribe")

        # 4. Hinglish (Sarvam)
        cfg_hi = resolve_language_mode("Hinglish transcription using the existing Sarvam integration")
        self.assertEqual(cfg_hi["engine"], "sarvam")
        self.assertIsNone(cfg_hi["whisper_code"])

        # 5. Automatic language detection
        cfg_auto = resolve_language_mode("Automatic language detection")
        self.assertEqual(cfg_auto["engine"], "whisper")
        self.assertIsNone(cfg_auto["whisper_code"])
        self.assertEqual(cfg_auto["task"], "transcribe")

    def test_02_nepali_transcription_parameters(self):
        """Verify that Nepali transcription passes language='ne' and task='transcribe' to Whisper."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True
        mock_model.transcribe.return_value = {"text": "नमस्ते नेपाल"}

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            text = transcribe_chunk_whisper("dummy_chunk.wav", language="ne", task="transcribe")

            mock_model.transcribe.assert_called_once_with(
                "dummy_chunk.wav",
                task="transcribe",
                language="ne",
            )
            self.assertEqual(text, "नमस्ते नेपाल")

    def test_03_nepali_to_english_translation_parameters(self):
        """Verify that Nepali-to-English translation passes language='ne' and task='translate'."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True
        mock_model.transcribe.return_value = {"text": "Hello Nepal"}

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            text = transcribe_chunk_whisper("dummy_chunk.wav", language="ne", task="translate")

            mock_model.transcribe.assert_called_once_with(
                "dummy_chunk.wav",
                task="translate",
                language="ne",
            )
            self.assertEqual(text, "Hello Nepal")

    def test_04_english_transcription_parameters(self):
        """Verify that English transcription passes language='en' and task='transcribe'."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True
        mock_model.transcribe.return_value = {"text": "Good morning"}

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            text = transcribe_chunk_whisper("dummy_chunk.wav", language="en", task="transcribe")

            mock_model.transcribe.assert_called_once_with(
                "dummy_chunk.wav",
                task="transcribe",
                language="en",
            )
            self.assertEqual(text, "Good morning")

    def test_05_hinglish_routes_to_sarvam(self):
        """Verify that Hinglish routing bypasses Whisper and invokes Sarvam."""
        with patch("core.transcriber.transcribe_chunk_sarvam", return_value="Translated Sarvam text") as mock_sv:
            text = transcribe_chunk("dummy_chunk.wav", language="hinglish")
            mock_sv.assert_called_once_with("dummy_chunk.wav")
            self.assertEqual(text, "Translated Sarvam text")

    def test_06_chunk_ordering_and_assembly(self):
        """Verify that all chunks are processed in strictly ascending order and joined cleanly."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True

        def fake_transcribe(chunk_path, **kwargs):
            if "chunk_0" in chunk_path:
                return {"text": "First sentence.", "segments": [{"start": 0.0, "end": 2.0, "text": "First sentence."}]}
            elif "chunk_1" in chunk_path:
                return {"text": "Second sentence.", "segments": [{"start": 1.0, "end": 3.0, "text": "Second sentence."}]}
            elif "chunk_2" in chunk_path:
                return {"text": "Third sentence.", "segments": [{"start": 0.5, "end": 2.5, "text": "Third sentence."}]}
            return {"text": ""}

        mock_model.transcribe.side_effect = fake_transcribe

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            chunks = ["path_chunk_0.wav", "path_chunk_1.wav", "path_chunk_2.wav"]
            result = transcribe_all(chunks, language="Nepali speech → English text")

            expected = "First sentence. Second sentence. Third sentence."
            self.assertEqual(result, expected)

    def test_07_empty_and_failed_chunks_handled_safely(self):
        """Verify that empty (0 bytes) or corrupted chunks do not abort or drop valid chunks."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True

        call_counts = {"count": 0}

        def fake_transcribe(chunk_path, **kwargs):
            call_counts["count"] += 1
            if "chunk_corrupt" in chunk_path:
                raise RuntimeError("Whisper decoding failure on corrupted frame")
            return {"text": f"Valid text for {chunk_path}."}

        mock_model.transcribe.side_effect = fake_transcribe

        def fake_getsize(path):
            if "chunk_empty" in path:
                return 0  # 0-byte file
            return 4096

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", side_effect=fake_getsize):

            chunks = ["chunk_0_valid.wav", "chunk_empty.wav", "chunk_corrupt.wav", "chunk_1_valid.wav"]
            result = transcribe_all(chunks, language="nepali-to-english")

            # chunk_empty should be skipped (0 bytes)
            # chunk_corrupt should log error safely without aborting
            # valid chunks 0 and 1 must be present
            self.assertIn("Valid text for chunk_0_valid.wav.", result)
            self.assertIn("Valid text for chunk_1_valid.wav.", result)
            self.assertNotIn("chunk_empty", result)

    def test_08_dual_nepali_mode(self):
        """Verify that return_original=True generates both translated English and original Devanagari text."""
        mock_model = MagicMock()
        mock_model.is_multilingual = True

        def fake_transcribe(chunk_path, **kwargs):
            if kwargs.get("task") == "translate":
                return {"text": "This is translated English text."}
            elif kwargs.get("task") == "transcribe" and kwargs.get("language") == "ne":
                return {"text": "यो मूल नेपाली पाठ हो।"}
            return {"text": ""}

        mock_model.transcribe.side_effect = fake_transcribe

        with patch("core.transcriber.load_model", return_value=mock_model), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=5000):

            res = transcribe_nepali_dual(["test_chunk_0.wav"])
            self.assertIsInstance(res, dict)
            self.assertEqual(res["transcript"], "This is translated English text.")
            self.assertEqual(res["original_transcript"], "यो मूल नेपाली पाठ हो।")
            self.assertTrue(res["is_translated"])

    def test_09_multilingual_model_guard(self):
        """Verify that an English-only model (.en) is rejected when non-English or translation is requested."""
        english_only_model = MagicMock()
        english_only_model.is_multilingual = False

        # Should fail for Nepali
        with self.assertRaises(ValueError):
            validate_whisper_model(english_only_model, language="ne", task="transcribe")

        # Should fail for translate task
        with self.assertRaises(ValueError):
            validate_whisper_model(english_only_model, language="ne", task="translate")

        # Multilingual model should pass
        multi_model = MagicMock()
        multi_model.is_multilingual = True
        try:
            validate_whisper_model(multi_model, language="ne", task="translate")
        except ValueError:
            self.fail("validate_whisper_model raised ValueError unexpectedly on multilingual model!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
