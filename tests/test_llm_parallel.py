import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.extractor import extract_all_insights


class TestLLMParallelOptimization(unittest.TestCase):
    """
    Focused test suite for Phase 2 LLM parallelization:
    output compatibility, bounded concurrency, partial failure isolation,
    and progress callback reporting.
    """

    def test_01_output_compatibility(self):
        """Verify that extract_all_insights returns all expected keys and types."""
        with patch("core.summarizer.generate_title", return_value="Test Meeting Title"), \
             patch("core.summarizer.summarize", return_value="Test Meeting Summary"), \
             patch("core.extractor.extract_action_items", return_value="1. Action item A"), \
             patch("core.extractor.extract_key_decisions", return_value="1. Decision A"), \
             patch("core.extractor.extract_questions", return_value="1. Question A"):

            res = extract_all_insights("dummy transcript text", max_workers=3)

            # Check all canonical keys
            self.assertIn("title", res)
            self.assertIn("summary", res)
            self.assertIn("action_items", res)
            self.assertIn("key_decisions", res)
            self.assertIn("open_questions", res)
            self.assertIn("stage_timings", res)
            self.assertIn("total_llm_time", res)

            # Check values
            self.assertEqual(res["title"], "Test Meeting Title")
            self.assertEqual(res["summary"], "Test Meeting Summary")
            self.assertEqual(res["action_items"], "1. Action item A")
            self.assertEqual(res["key_decisions"], "1. Decision A")
            self.assertEqual(res["open_questions"], "1. Question A")

            # Check timings dictionary
            timings = res["stage_timings"]
            self.assertIn("Title Generation", timings)
            self.assertIn("Summary Generation", timings)
            self.assertIn("Action Items", timings)
            self.assertIn("Key Decisions", timings)
            self.assertIn("Open Questions", timings)

    def test_02_concurrency_execution(self):
        """Verify that independent tasks execute concurrently across threads."""
        active_threads = set()

        def slow_mock(name):
            def _fn(*args, **kwargs):
                import threading
                active_threads.add(threading.current_thread().name)
                time.sleep(0.05)
                return f"{name} result"
            return _fn

        with patch("core.summarizer.generate_title", side_effect=slow_mock("title")), \
             patch("core.summarizer.summarize", side_effect=slow_mock("summary")), \
             patch("core.extractor.extract_action_items", side_effect=slow_mock("actions")), \
             patch("core.extractor.extract_key_decisions", side_effect=slow_mock("decisions")), \
             patch("core.extractor.extract_questions", side_effect=slow_mock("questions")):

            t0 = time.perf_counter()
            res = extract_all_insights("dummy transcript text", max_workers=3)
            elapsed = time.perf_counter() - t0

            # If serial (5 * 0.05s = 0.25s). Concurrent with 3 workers takes ~0.10s.
            self.assertLess(elapsed, 0.22, f"Execution took {elapsed:.2f}s, expected concurrent speedup")
            self.assertGreater(len(active_threads), 1, "Expected multiple threads to be utilized")
            self.assertEqual(res["title"], "title result")

    def test_03_partial_failure_isolation(self):
        """Verify that a failure in one LLM task does not abort or corrupt other tasks."""
        def failing_summary(*args, **kwargs):
            raise RuntimeError("Mistral API 500 Internal Server Error")

        with patch("core.summarizer.generate_title", return_value="Resilient Title"), \
             patch("core.summarizer.summarize", side_effect=failing_summary), \
             patch("core.extractor.extract_action_items", return_value="Action item preserved"), \
             patch("core.extractor.extract_key_decisions", return_value="Decision preserved"), \
             patch("core.extractor.extract_questions", return_value="Question preserved"):

            res = extract_all_insights("dummy transcript text", max_workers=3)

            # Successful tasks are completely preserved
            self.assertEqual(res["title"], "Resilient Title")
            self.assertEqual(res["action_items"], "Action item preserved")
            self.assertEqual(res["key_decisions"], "Decision preserved")
            self.assertEqual(res["open_questions"], "Question preserved")

            # Failed task is isolated gracefully
            self.assertIn("Failed to generate Summary Generation", res["summary"])

    def test_04_progress_callback(self):
        """Verify that progress callback is invoked for each completed task."""
        progress_events = []

        def callback(curr, total, name):
            progress_events.append((curr, total, name))

        with patch("core.summarizer.generate_title", return_value="Title"), \
             patch("core.summarizer.summarize", return_value="Summary"), \
             patch("core.extractor.extract_action_items", return_value="Actions"), \
             patch("core.extractor.extract_key_decisions", return_value="Decisions"), \
             patch("core.extractor.extract_questions", return_value="Questions"):

            extract_all_insights("dummy text", max_workers=3, progress_callback=callback)

            self.assertEqual(len(progress_events), 5)
            # Final call must be 5/5
            self.assertEqual(progress_events[-1][0], 5)
            self.assertEqual(progress_events[-1][1], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
