"""
Phase 7 Test Suite: Meeting Intelligence, Verified Citations, and Executive Reporting.

Covers:
1. Structured ActionItem, MeetingDecision, and MeetingIntelligence data models.
2. Verified citation mapping, rejecting fabricated, missing, and out-of-range timestamps.
3. Multilingual PDF output (English, Hindi, Nepali/Devanagari) and multi-page wrapping.
4. Robust recovery from malformed LLM output and empty transcripts.
5. Executive Markdown report generation.
6. Backward compatibility of legacy functions (extract_action_items, build_report, etc.).
7. Live-to-post intelligence handoff and session isolation preservation.
"""

import io
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from core.extractor import (
    ActionItem,
    MeetingDecision,
    ExecutiveSummary,
    MeetingIntelligence,
    find_best_segment_citation,
    extract_structured_action_items,
    extract_structured_decisions,
    extract_meeting_intelligence,
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    extract_unresolved_questions,
    extra_action_items,
)
from core.report_generator import (
    generate_pdf_report,
    generate_markdown_report,
    sanitize_text_for_core_font,
)
from core.meeting_qa import Citation
from app import build_report, process_live_intelligence_ui


class TestMeetingIntelligenceModels(unittest.TestCase):
    """Test 01: Verify structured intelligence models and explicit defaults."""

    def test_01_models_and_explicit_defaults(self):
        # ActionItem defaults: never invent missing owners/deadlines
        item = ActionItem(task="Deploy staging environment")
        self.assertEqual(item.task, "Deploy staging environment")
        self.assertEqual(item.owner, "Unknown")
        self.assertEqual(item.priority, "Medium")
        self.assertEqual(item.deadline, "Not specified")
        self.assertIsNone(item.citation)
        self.assertEqual(item.status, "Open")

        # MeetingDecision defaults
        dec = MeetingDecision(decision="Switch to weekly sprints")
        self.assertEqual(dec.decision, "Switch to weekly sprints")
        self.assertEqual(dec.context_rationale, "Not specified")
        self.assertEqual(dec.consensus_owner, "Unknown")
        self.assertIsNone(dec.citation)

        # ExecutiveSummary & MeetingIntelligence
        summary = ExecutiveSummary(title="Sprint Planning", overview="Discussed roadmap.")
        intel = MeetingIntelligence(
            session_id="sess_123",
            summary=summary,
            action_items=[item],
            decisions=[dec],
            raw_transcript="We will switch to weekly sprints.",
            duration_seconds=120.0,
            language_mode="English",
            audio_source="Microphone + Loopback",
        )
        self.assertEqual(intel.session_id, "sess_123")
        self.assertEqual(len(intel.action_items), 1)
        self.assertEqual(len(intel.decisions), 1)


class TestCitationValidationAndGrounding(unittest.TestCase):
    """Test 02: Validate citations against actual segments; reject fabricated & out-of-range timestamps."""

    def setUp(self):
        self.segments = [
            {"start": 10.0, "end": 25.0, "text": "Sarah will finalize the backend API documentation by Friday."},
            {"start": 30.0, "end": 45.0, "text": "We agreed to adopt PostgreSQL for our primary persistent storage."},
            {"start": 50.0, "end": 65.0, "text": "Who is going to investigate the Redis cluster latency spike?"},
        ]
        self.max_duration = 70.0

    def test_02_valid_candidate_timestamp_matched(self):
        # Claim with accurate timestamp candidate
        citation = find_best_segment_citation(
            claim_text="Finalize backend API documentation",
            segments=self.segments,
            candidate_timestamp="[00:10 - 00:25]",
            max_duration=self.max_duration,
        )
        self.assertIsNotNone(citation)
        self.assertEqual(citation.start, 10.0)
        self.assertEqual(citation.end, 25.0)
        self.assertIn("Sarah will finalize", citation.excerpt)

    def test_03_out_of_range_timestamp_rejected(self):
        # Model hallucinated [05:00] when recording is only 70s
        citation = find_best_segment_citation(
            claim_text="Prepare the marketing launch slides",
            segments=self.segments,
            candidate_timestamp="[05:00 - 05:20]",
            max_duration=self.max_duration,
        )
        # Should reject the fabricated timestamp, and since text is absent from segments, return None
        self.assertIsNone(citation)

    def test_04_missing_candidate_timestamp_mapped_to_content(self):
        # No candidate timestamp provided, but claim matches segment 2
        citation = find_best_segment_citation(
            claim_text="Adopt PostgreSQL for primary persistent storage",
            segments=self.segments,
            candidate_timestamp="",
            max_duration=self.max_duration,
        )
        self.assertIsNotNone(citation)
        self.assertEqual(citation.start, 30.0)
        self.assertEqual(citation.end, 45.0)
        self.assertIn("PostgreSQL", citation.excerpt)

    def test_05_unsupported_claim_returns_none(self):
        # Claim is completely absent from transcript
        citation = find_best_segment_citation(
            claim_text="Build an iOS native Swift client for wearable devices",
            segments=self.segments,
            candidate_timestamp="",
            max_duration=self.max_duration,
        )
        self.assertIsNone(citation)


class TestStructuredExtraction(unittest.TestCase):
    """Test 03-05: Structured extraction, recovery from malformed output, empty transcripts."""

    def test_06_structured_action_items_and_decisions(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="""[
              {
                "task": "Migrate database to PostgreSQL",
                "owner": "David",
                "priority": "High",
                "deadline": "End of Q3",
                "timestamp": "[00:30]"
              },
              {
                "task": "Draft customer survey",
                "owner": null,
                "priority": null,
                "deadline": "TBD",
                "timestamp": "[00:10]"
              }
            ]"""
        )

        segments = [
            {"start": 10.0, "end": 20.0, "text": "We should draft a customer survey."},
            {"start": 30.0, "end": 45.0, "text": "David will migrate the database to PostgreSQL by end of Q3."},
        ]

        items = extract_structured_action_items(
            transcript="Full meeting transcript",
            segments=segments,
            max_duration=60.0,
            llm=mock_llm,
        )

        self.assertEqual(len(items), 2)
        # Item 1
        self.assertEqual(items[0].task, "Migrate database to PostgreSQL")
        self.assertEqual(items[0].owner, "David")
        self.assertEqual(items[0].priority, "High")
        self.assertEqual(items[0].deadline, "End of Q3")
        self.assertIsNotNone(items[0].citation)
        self.assertEqual(items[0].citation.start, 30.0)

        # Item 2: defaults properly applied for null/TBD
        self.assertEqual(items[1].task, "Draft customer survey")
        self.assertEqual(items[1].owner, "Unknown")
        self.assertEqual(items[1].priority, "Medium")
        self.assertEqual(items[1].deadline, "Not specified")
        self.assertIsNotNone(items[1].citation)

    def test_07_malformed_llm_output_recovery(self):
        mock_llm = MagicMock()
        # LLM returned numbered markdown lines instead of valid JSON
        mock_llm.invoke.return_value = AIMessage(
            content="""Here are the items:
            1. Update the security certificates
            2. Run load tests on staging server
            """
        )

        items = extract_structured_action_items(
            transcript="Sample transcript text",
            segments=[],
            llm=mock_llm,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].task, "Update the security certificates")
        self.assertEqual(items[0].owner, "Unknown")
        self.assertEqual(items[0].deadline, "Not specified")

    def test_08_empty_transcript_handling(self):
        # Empty transcript must return empty list without LLM calls
        items = extract_structured_action_items("", segments=[])
        self.assertEqual(items, [])

        decisions = extract_structured_decisions("   \n\t  ", segments=[])
        self.assertEqual(decisions, [])


class TestReportGeneration(unittest.TestCase):
    """Test 06-08: Multilingual PDF generation (English, Hindi, Nepali), multi-page wrapping, Markdown export."""

    def test_09_multilingual_pdf_export_english_hindi_nepali(self):
        # Multilingual content with Devanagari script
        intelligence_data = {
            "title": "वार्षिक योजना तथा त्रैमासिक समीक्षा (Annual Planning & Review)",
            "summary": "यस बैठकमा टोलीले नयाँ उत्पादन सुरुवात र बजेट विनियोजनबारे छलफल गर्यो। We discussed roadmap milestones and Q4 budget allocation.",
            "action_items": [
                ActionItem(
                    task="अनुसन्धान प्रतिवेदन तयार गर्ने (Prepare research report)",
                    owner="रमेश (Ramesh)",
                    priority="High",
                    deadline="शुक्रबार (Friday)",
                    citation=Citation(start=12.0, end=28.0, timestamp_str="00:12 - 00:28", excerpt="रमेशले शुक्रबारसम्म प्रतिवेदन बुझाउनेछ।"),
                ),
                ActionItem(
                    task="Deploy OAuth2 authentication service",
                    owner="Alex",
                    priority="Medium",
                    deadline="Next sprint",
                    citation=Citation(start=45.0, end=60.0, timestamp_str="00:45 - 01:00", excerpt="Alex will handle the OAuth2 deploy."),
                ),
            ],
            "key_decisions": [
                MeetingDecision(
                    decision="क्लाउड पूर्वाधारमा अपग्रेड गर्ने निर्णय (Approved cloud infrastructure upgrade)",
                    context_rationale="कार्यसम्पादन र स्केलेबिलिटी बढाउन (To boost scalability)",
                    consensus_owner="सर्वसम्मत (Unanimous)",
                    citation=Citation(start=65.0, end=80.0, timestamp_str="01:05 - 01:20", excerpt="सबैले क्लाउड अपग्रेड स्वीकार गरे।"),
                ),
            ],
            "open_questions": [
                "कुन क्लाउड प्रदायक छनौट गर्ने? (Which cloud provider to select?)",
                "What is the timeline for data migration?",
            ],
            "transcript": "Full transcript with multiple lines.\nपहिलो चरणमा योजना बनाइयो।\nSecond step was implementation.",
            "segments": [
                {"start": 12.0, "end": 28.0, "text": "रमेशले शुक्रबारसम्म प्रतिवेदन बुझाउनेछ।"},
                {"start": 45.0, "end": 60.0, "text": "Alex will handle the OAuth2 deploy."},
                {"start": 65.0, "end": 80.0, "text": "सबैले क्लाउड अपग्रेड स्वीकार गरे।"},
            ],
            "metadata": {
                "date": "2026-10-25 15:30",
                "duration": "01:20",
                "language": "Nepali speech → English text",
                "audio_source": "Microphone + System Loopback",
            },
        }

        pdf_bytes = generate_pdf_report(intelligence_data)
        self.assertIsInstance(pdf_bytes, bytes)
        # Verify valid PDF header
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        # Verify substantial non-empty file generated
        self.assertGreater(len(pdf_bytes), 3000)

    def test_10_multipage_pdf_wrapping(self):
        # Create a lengthy transcript with 80+ lines to force multi-page generation
        long_segments = [
            {"start": float(i * 10), "end": float(i * 10 + 8), "text": f"Meeting discussion segment #{i}: Reviewing key operational metrics and architectural dependencies across distributed microservices."}
            for i in range(80)
        ]

        data = {
            "title": "Extended Architecture Review (Multi-Page Test)",
            "summary": "In-depth review spanning multiple pages.",
            "action_items": [
                ActionItem(task=f"Action item #{i}", owner="Engineer", priority="Medium", deadline="Sprint end")
                for i in range(10)
            ],
            "key_decisions": [
                MeetingDecision(decision=f"Decision #{i}", context_rationale="Architecture consensus")
                for i in range(6)
            ],
            "open_questions": ["Question 1?", "Question 2?"],
            "transcript": "\n".join([s["text"] for s in long_segments]),
            "segments": long_segments,
            "metadata": {"duration": "13:20", "language": "English", "audio_source": "Meeting Audio"},
        }

        pdf_bytes = generate_pdf_report(data)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 15000)

    def test_11_markdown_report_formatting(self):
        data = {
            "title": "Quarterly Sprint Retrospective",
            "summary": "Team analyzed sprint velocity and deployment blockers.",
            "action_items": [
                ActionItem(
                    task="Configure automated canary deployments",
                    owner="Maya",
                    priority="High",
                    deadline="Oct 31",
                    citation=Citation(start=10.0, end=20.0, timestamp_str="00:10 - 00:20", excerpt="Maya will configure canary."),
                ),
            ],
            "key_decisions": [
                MeetingDecision(
                    decision="Enforce 80% test coverage gate on PRs",
                    context_rationale="Reduce production regressions",
                    consensus_owner="Tech Lead",
                    citation=Citation(start=30.0, end=40.0, timestamp_str="00:30 - 00:40", excerpt="Coverage gate is now 80%."),
                ),
            ],
            "open_questions": ["How do we handle legacy untyped modules?"],
            "transcript": "[00:10 - 00:20] Maya will configure canary.\n[00:30 - 00:40] Coverage gate is now 80%.",
            "segments": [
                {"start": 10.0, "end": 20.0, "text": "Maya will configure canary."},
                {"start": 30.0, "end": 40.0, "text": "Coverage gate is now 80%."},
            ],
            "metadata": {"date": "2026-10-25", "duration": "00:40", "language": "English", "audio_source": "Microphone"},
        }

        md = generate_markdown_report(data)
        self.assertIn("# Quarterly Sprint Retrospective", md)
        self.assertIn("| Metric | Value |", md)
        self.assertIn("## 1. Executive Summary", md)
        self.assertIn("## 2. Action Items", md)
        self.assertIn("| 1 | Configure automated canary deployments | Maya | High | Oct 31 | `00:10 - 00:20` |", md)
        self.assertIn("## 3. Key Decisions", md)
        self.assertIn("| 1 | Enforce 80% test coverage gate on PRs | Reduce production regressions | Tech Lead | `00:30 - 00:40` |", md)
        self.assertIn("## 5. Complete Meeting Transcript", md)


class TestBackwardCompatibilityAndIntegration(unittest.TestCase):
    """Test 09-10: Legacy signatures preservation, live-to-post transition, session isolation."""

    def test_12_legacy_extraction_signatures_preserved(self):
        # Ensure exact return type and signature of legacy extraction functions
        with patch("core.extractor.invoke_with_retry") as mock_invoke:
            mock_invoke.return_value = "1. Legacy task - Owner"
            res_actions = extract_action_items("sample transcript")
            self.assertIsInstance(res_actions, str)
            self.assertEqual(res_actions, "1. Legacy task - Owner")

            # Check alias
            res_alias = extra_action_items("sample transcript")
            self.assertEqual(res_alias, "1. Legacy task - Owner")

            mock_invoke.return_value = "1. Legacy decision"
            res_dec = extract_key_decisions("sample transcript")
            self.assertIsInstance(res_dec, str)
            self.assertEqual(res_dec, "1. Legacy decision")

            mock_invoke.return_value = "1. Legacy question?"
            res_q = extract_questions("sample transcript")
            self.assertIsInstance(res_q, str)

            res_unres = extract_unresolved_questions("sample transcript")
            self.assertIsInstance(res_unres, str)

        # Ensure legacy build_report works as before
        legacy_result = {
            "title": "Legacy Meeting",
            "summary": "Legacy summary text",
            "action_items": "1. Task A",
            "key_decisions": "1. Decision A",
            "open_questions": "1. Question A",
            "transcript": "Full text transcript",
            "display_language": "English",
        }
        legacy_md = build_report(legacy_result)
        self.assertIsInstance(legacy_md, str)
        self.assertIn("# Legacy Meeting", legacy_md)
        self.assertIn("## Action items\n1. Task A", legacy_md)

    def test_13_live_to_post_handoff_and_session_safety(self):
        # Test process_live_intelligence_ui with live segments
        live_segments = [
            {"start": 0.0, "end": 15.0, "text": "Welcome everyone to the product synchronization."},
            {"start": 16.0, "end": 35.0, "text": "We will ship the beta release on Wednesday. John is leading QA."},
        ]
        live_text = "Welcome everyone to the product synchronization. We will ship the beta release on Wednesday. John is leading QA."

        with patch("app.extract_all_insights") as mock_insights, \
             patch("app.extract_structured_action_items") as mock_act, \
             patch("app.extract_structured_decisions") as mock_dec, \
             patch("app.build_rag_chain") as mock_rag:

            mock_insights.return_value = {
                "title": "Product Synchronization",
                "summary": "Ship beta release Wednesday.",
                "action_items": "1. John lead QA",
                "key_decisions": "1. Ship beta Wednesday",
                "open_questions": "",
                "stage_timings": {"LLM": 1.2},
            }
            mock_act.return_value = [ActionItem(task="Lead QA", owner="John", priority="High")]
            mock_dec.return_value = [MeetingDecision(decision="Ship beta on Wednesday")]
            mock_rag.return_value = MagicMock()

            # Execute live-to-post intelligence handoff
            result = process_live_intelligence_ui(
                transcript=live_text,
                segments=live_segments,
                language="English",
                audio_source="Live Recording (both)",
                duration_seconds=35.0,
                session_id="live_sess_test_99",
            )

            # Assert complete result dictionary structure
            self.assertEqual(result["session_id"], "live_sess_test_99")
            self.assertEqual(result["title"], "Product Synchronization")
            self.assertEqual(len(result["structured_action_items"]), 1)
            self.assertEqual(len(result["structured_key_decisions"]), 1)
            self.assertEqual(len(result["segments"]), 2)
            self.assertIsNotNone(result["pdf_bytes"])
            self.assertTrue(result["pdf_bytes"].startswith(b"%PDF-"))

            # Confirm build_rag_chain was called with matching session_id and segments
            mock_rag.assert_called_once_with(
                live_text,
                session_id="live_sess_test_99",
                segments=live_segments,
            )

    def test_14_live_recording_language_resolution_and_initialization(self):
        """Regression test: verify language is defined and accepted when starting live recording."""
        from app import LANGUAGE_OPTIONS
        from core.live_transcriber import LiveTranscriber
        from core.transcriber import resolve_language_mode

        # Verify all supported language options can initialize LiveTranscriber without NameError
        for lang_opt in LANGUAGE_OPTIONS:
            transcriber = LiveTranscriber(language=lang_opt)
            self.assertEqual(transcriber.language, lang_opt)
            mode_info = resolve_language_mode(transcriber.language)
            self.assertIn("whisper_code", mode_info)
            self.assertIn("display_name", mode_info)

    def test_15_live_transcript_snapshot_and_recorder_elapsed_compatibility(self):
        """Regression test: verify LiveTranscript snapshot keys and AudioRecorder elapsed time compatibility."""
        from core.live_transcriber import LiveTranscript, LiveTranscriptSegment
        from core.audio_recorder import AudioRecorder

        transcript = LiveTranscript()
        transcript.add_segment(LiveTranscriptSegment(index=1, start=0.0, end=4.5, text="Snapshot compatibility test", language="en"))
        snapshot = transcript.get_snapshot()

        # Both 'full_text' and 'text' must exist to prevent KeyError in UI handoffs
        self.assertIn("full_text", snapshot)
        self.assertIn("text", snapshot)
        self.assertEqual(snapshot["full_text"], snapshot["text"])
        self.assertEqual(snapshot["text"], "Snapshot compatibility test")

        # AudioRecorder must support both get_elapsed_seconds() and get_elapsed_time()
        rec = AudioRecorder(mock_mode=True)
        self.assertEqual(rec.get_elapsed_seconds(), 0.0)
        self.assertEqual(rec.get_elapsed_time(), 0.0)
        self.assertEqual(rec.get_elapsed_seconds(), rec.get_elapsed_time())



