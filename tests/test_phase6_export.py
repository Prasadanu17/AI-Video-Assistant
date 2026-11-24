"""
Tests for Phase 6 (Multi-Format Export: DOCX, JSON, PDF, Markdown).
"""

import json
import unittest

from core.report_generator import (
    generate_docx_report,
    generate_json_report,
    generate_markdown_report,
    generate_pdf_report,
)


class TestPhase6MultiFormatExport(unittest.TestCase):

    def setUp(self):
        self.sample_data = {
            "title": "Quarterly Technical Review",
            "summary": "The team reviewed quarterly engineering deliverables and decided on architecture goals.",
            "action_items": [
                {
                    "task": "Migrate data pipeline to PostgreSQL",
                    "assignee": "John Doe",
                    "priority": "High",
                    "deadline": "Friday 5 PM",
                    "timestamp": "[05:10 - 05:30]",
                }
            ],
            "key_decisions": [
                {
                    "decision": "Adopt PostgreSQL for primary persistent storage",
                    "rationale": "High concurrency support",
                    "decision_makers": "Team Consensus",
                    "timestamp": "[04:20 - 04:45]",
                }
            ],
            "minutes_of_meeting": {
                "meeting_type": "Quarterly Review",
                "chair_or_facilitator": "Sarah Connor",
                "attendees_mentioned": ["John Doe", "Sarah Connor"],
                "formal_minutes_text": "Meeting opened at 10:00 AM. Discussed architecture migration. Adjourned at 10:45 AM.",
            },
            "transcript": "Hello everyone. Welcome to our quarterly review. We will migrate to PostgreSQL.",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "Speaker 1", "text": "Hello everyone. Welcome to our quarterly review."},
                {"start": 5.0, "end": 10.0, "speaker": "Speaker 2", "text": "We will migrate to PostgreSQL."},
            ],
            "metadata": {
                "duration": "45.0s",
                "language": "English",
                "audio_source": "Live Recording",
                "session_id": "test_sess_123",
                "date": "2026-11-24 20:00",
            },
        }

    def test_01_generate_docx_report(self):
        """Verify Word Document (.docx) report generation produces valid bytes."""
        docx_bytes = generate_docx_report(self.sample_data)
        self.assertIsInstance(docx_bytes, bytes)
        self.assertTrue(len(docx_bytes) > 1000)
        # Verify ZIP header for DOCX files
        self.assertTrue(docx_bytes.startswith(b"PK\x03\x04"))

    def test_02_generate_json_report(self):
        """Verify structured JSON export generates valid, parsable JSON."""
        json_str = generate_json_report(self.sample_data)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["title"], "Quarterly Technical Review")
        self.assertEqual(len(parsed["action_items"]), 1)
        self.assertEqual(parsed["action_items"][0]["assignee"], "John Doe")
        self.assertEqual(len(parsed["key_decisions"]), 1)

    def test_03_generate_markdown_report(self):
        """Verify Markdown export contains formatted headings and tables."""
        md_text = generate_markdown_report(self.sample_data)
        self.assertIn("# Quarterly Technical Review", md_text)
        self.assertIn("## 1. Executive Summary", md_text)
        self.assertIn("## 2. Action Items", md_text)
        self.assertIn("Migrate data pipeline to PostgreSQL", md_text)
        self.assertIn("Adopt PostgreSQL for primary persistent storage", md_text)

    def test_04_generate_pdf_report(self):
        """Verify PDF report generation produces valid PDF bytes."""
        pdf_bytes = generate_pdf_report(self.sample_data)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
