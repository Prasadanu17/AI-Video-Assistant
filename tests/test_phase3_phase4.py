"""
Tests for Phase 3 (Hierarchical Intelligence) & Phase 4 (Speaker Diarization).
"""

import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from core.meeting_intelligence import (
    ComprehensiveMeetingReport,
    MeetingTopic,
    ConfirmedDecision,
    ActionItemDetail,
    extract_hierarchical_intelligence,
    verify_and_align_citation,
)
from core.speaker_diarizer import (
    extract_acoustic_fingerprint,
    diarize_segments,
    apply_speaker_renames,
)


class TestPhase3HierarchicalIntelligence(unittest.TestCase):

    def test_01_comprehensive_report_schema_validation(self):
        """Verify ComprehensiveMeetingReport instantiation and Pydantic validation."""
        report = ComprehensiveMeetingReport(
            title="Sprint Planning & Backend Architecture",
            meeting_overview="Overview of sprint goals and database migration strategy.",
            executive_summary="Team agreed to migrate to Postgres and deploy by Friday.",
            detailed_meeting_summary="In the first half, John explained the Postgres migration. In the second half, Sarah approved.",
            topics_discussed=[
                MeetingTopic(
                    topic_title="Database Migration",
                    topic_description="Discussion on migrating from SQLite to Postgres.",
                    subtopics=["Connection pooling", "Index optimization"],
                    important_points=["Postgres supports higher concurrent transactions."],
                    suggestions=["Use connection pooling library."],
                    start_time="00:00",
                    end_time="05:30",
                )
            ],
            key_decisions=[
                ConfirmedDecision(
                    decision="Migrate primary database to PostgreSQL",
                    rationale="High concurrent transaction volume requires Postgres.",
                    decision_makers="Team Consensus",
                    timestamp="[04:20 - 04:35]",
                    evidence_quote="Let's proceed with PostgreSQL.",
                )
            ],
            action_items=[
                ActionItemDetail(
                    task="Set up Postgres RDS staging environment",
                    assignee="John",
                    deadline="Friday 5 PM",
                    priority="High",
                    timestamp="[05:10 - 05:25]",
                    evidence_quote="John will provision the database by Friday.",
                )
            ],
            important_announcements=["Next week's office hours moved to Tuesday."],
            pending_tasks=["Review frontend pull request #42."],
            next_steps=["Begin schema migration script."],
            next_meeting_details="Monday at 10:00 AM",
            minutes_of_meeting={
                "meeting_type": "Technical Sprint Planning",
                "chair_or_facilitator": "Sarah",
                "attendees_mentioned": ["John", "Sarah"],
                "formal_minutes_text": "Meeting opened at 10:00 AM. Discussed database migration. Agreed on PostgreSQL.",
            },
        )
        self.assertEqual(report.title, "Sprint Planning & Backend Architecture")
        self.assertEqual(len(report.topics_discussed), 1)
        self.assertEqual(report.topics_discussed[0].topic_title, "Database Migration")
        self.assertEqual(len(report.key_decisions), 1)
        self.assertEqual(report.action_items[0].assignee, "John")

    def test_02_empty_transcript_handling(self):
        """Verify graceful empty transcript handling without LLM invocation."""
        report = extract_hierarchical_intelligence("")
        self.assertEqual(report.title, "Empty Meeting Transcript")
        self.assertEqual(len(report.topics_discussed), 0)
        self.assertEqual(len(report.action_items), 0)

    def test_03_evidence_and_citation_verification(self):
        """Verify verify_and_align_citation accurately maps claims to verified segment timestamps."""
        segments = [
            {"start": 10.0, "end": 20.0, "text": "We need to fix the deployment pipeline immediately."},
            {"start": 25.0, "end": 35.0, "text": "John please take care of the database backup."},
        ]
        # Should match segment 2 for John's task
        ts, quote = verify_and_align_citation(
            claim_text="database backup",
            evidence_quote="take care of the database backup",
            candidate_timestamp="00:26",
            segments=segments,
        )
        self.assertEqual(ts, "00:25 - 00:35")
        self.assertIn("database backup", quote)


class TestPhase4SpeakerDiarization(unittest.TestCase):

    def test_01_acoustic_fingerprint_extraction(self):
        """Verify acoustic feature fingerprint extraction on simulated audio."""
        sr = 16000
        # 1-second audio tone
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        fp = extract_acoustic_fingerprint(audio, sample_rate=sr)
        self.assertIsNotNone(fp)
        self.assertIsInstance(fp, np.ndarray)
        self.assertTrue(len(fp) > 10)
        # Check normalized length ~ 1.0
        self.assertAlmostEqual(float(np.linalg.norm(fp)), 1.0, places=3)

    def test_02_speaker_clustering_differentiation(self):
        """Verify diarize_segments distinguishes between two distinctly pitched speakers."""
        sr = 16000
        # Generate 4-second audio: 2 seconds low tone (Speaker A 180Hz), 2 seconds high tone (Speaker B 600Hz)
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        audio_a = 0.6 * np.sin(2 * np.pi * 180 * t)
        audio_b = 0.6 * np.sin(2 * np.pi * 600 * t)
        combined_audio = np.concatenate([audio_a, audio_b])

        segments = [
            {"start": 0.0, "end": 2.0, "text": "Welcome to our status update."},
            {"start": 2.0, "end": 4.0, "text": "Thank you, here are the metrics."},
        ]

        result_segs = diarize_segments(combined_audio, segments, sample_rate=sr, distance_threshold=0.03)
        self.assertEqual(len(result_segs), 2)
        spk_1 = result_segs[0]["speaker"]
        spk_2 = result_segs[1]["speaker"]
        self.assertTrue(spk_1.startswith("Speaker"))
        self.assertTrue(spk_2.startswith("Speaker"))
        self.assertNotEqual(spk_1, spk_2)

    def test_03_speaker_renaming(self):
        """Verify apply_speaker_renames updates speaker names accurately."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello everyone.", "speaker": "Speaker 1"},
            {"start": 2.0, "end": 4.0, "text": "Hi team.", "speaker": "Speaker 2"},
        ]
        mapping = {"Speaker 1": "Dr. Sharma", "Speaker 2": "Sarah"}
        renamed = apply_speaker_renames(segments, mapping)
        self.assertEqual(renamed[0]["speaker"], "Dr. Sharma")
        self.assertEqual(renamed[0]["original_speaker_label"], "Speaker 1")
        self.assertEqual(renamed[1]["speaker"], "Sarah")


if __name__ == "__main__":
    unittest.main()
