"""
Comprehensive Test Suite for Phase 6: Live and Post-Meeting Q&A Engine.

Tests:
1. Live Q&A with empty transcript (honest immediate response without LLM waste).
2. Live Q&A cutoff awareness:
   - Early query before topic is discussed -> 'not_yet_mentioned'.
   - Later query after topic is discussed -> 'confirmed' with exact [MM:SS] citation.
3. Live Q&A future topic rejection: strictly refuses to speculate or claim whole-meeting knowledge.
4. Timestamp and transcript excerpt accuracy.
5. Live Q&A ambiguous/incomplete evidence handling.
6. Post-meeting Q&A with confirmed answer and multi-chunk evidence synthesis.
7. Post-meeting Q&A with unmentioned topic -> 'not_found_completed'.
8. Cross-session isolation: query for session B cannot access session A transcript chunks.
9. Chat history & follow-up question grounding without hallucinating unmentioned facts.
10. Live Q&A concurrency during active audio capture (audio recorder not blocked by Q&A).
"""

import os
import sys
import time
import numpy as np
import pytest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.meeting_qa import (
    ask_live_meeting_question,
    ask_post_meeting_question,
    AnswerState,
    MeetingQAResponse,
    format_timestamp,
    parse_timestamp_range,
    extract_citations_from_answer,
)
from core.live_transcriber import LiveTranscriber, LiveAudioChunker, LiveTranscriptSegment
from core.audio_recorder import AudioRecorder, AudioSource
from core.vector_store import build_vector_store, get_retriever, SENTINEL_NO_MATCH_SESSION
from core.rag_engine import create_rag_runnable


class TestLiveMeetingQA:
    """Unit and simulated integration tests for real-time in-meeting Q&A."""

    def test_01_empty_transcript_handling(self):
        """When no speech has been captured, immediately return honest not_yet_mentioned response."""
        empty_snapshot = {"segments": [], "full_text": "", "is_busy": False}
        resp = ask_live_meeting_question(empty_snapshot, "What is our Q3 budget?")

        assert resp.state == AnswerState.NOT_YET_MENTIONED
        assert "no speech has been captured" in resp.answer.lower()
        assert resp.cutoff_seconds == 0.0
        assert resp.cutoff_timestamp == "00:00"

    def test_02_temporal_cutoff_awareness(self):
        """
        Verify that a question asked at 00:30 (before budget is discussed at 01:10)
        returns 'not_yet_mentioned', and when asked at 01:30 returns 'confirmed'.
        """
        early_segments = [
            {"index": 0, "start": 0.0, "end": 15.0, "text": "Welcome to the team sync, we have five items on the agenda."},
            {"index": 1, "start": 15.0, "end": 30.0, "text": "Let's first review the UI design deliverables from last sprint."},
        ]
        early_snapshot = {"segments": early_segments, "full_text": " ".join(s["text"] for s in early_segments)}

        # Mock LLM response for early snapshot
        mock_llm_early = MagicMock()
        mock_llm_early.invoke.return_value = MagicMock(
            content="[STATUS: NOT_YET_MENTIONED] Based on the meeting transcript up to 00:30, the budget has not yet been discussed."
        )

        resp_early = ask_live_meeting_question(
            early_snapshot,
            "What is the allocated budget for marketing?",
            llm=mock_llm_early,
        )
        assert resp_early.state == AnswerState.NOT_YET_MENTIONED
        assert resp_early.cutoff_timestamp == "00:30"
        assert "not yet been discussed" in resp_early.answer.lower()

        # Later snapshot containing the budget discussion at 01:10 - 01:25
        later_segments = early_segments + [
            {"index": 2, "start": 30.0, "end": 65.0, "text": "The frontend components are complete and deployed."},
            {"index": 3, "start": 65.0, "end": 85.0, "text": "Moving to finances, we have allocated twenty thousand dollars for marketing campaigns."},
        ]
        later_snapshot = {"segments": later_segments, "full_text": " ".join(s["text"] for s in later_segments)}

        mock_llm_later = MagicMock()
        mock_llm_later.invoke.return_value = MagicMock(
            content="[STATUS: CONFIRMED] The marketing budget is $20,000 [01:05 - 01:25], as the speaker stated 'allocated twenty thousand dollars for marketing campaigns'."
        )

        resp_later = ask_live_meeting_question(
            later_snapshot,
            "What is the allocated budget for marketing?",
            llm=mock_llm_later,
        )
        assert resp_later.state == AnswerState.CONFIRMED
        assert resp_later.cutoff_timestamp == "01:25"
        assert "$20,000" in resp_later.answer
        assert len(resp_later.citations) > 0
        assert resp_later.citations[0].timestamp_str == "01:05 - 01:25"

    def test_03_future_topic_rejection_no_hallucination(self):
        """Strictly refuses to speculate on topics not yet covered in active recording."""
        segments = [
            {"index": 0, "start": 0.0, "end": 10.0, "text": "Let us discuss the hiring timeline."},
            {"index": 1, "start": 10.0, "end": 20.0, "text": "We need two backend engineers by next month."},
        ]
        snapshot = {"segments": segments, "full_text": " ".join(s["text"] for s in segments)}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="[STATUS: NOT_YET_MENTIONED] Based on the meeting transcript up to 00:20, the product launch date has not yet been discussed."
        )

        resp = ask_live_meeting_question(
            snapshot,
            "When is the official product launch?",
            llm=mock_llm,
        )
        assert resp.state == AnswerState.NOT_YET_MENTIONED
        assert "not yet been discussed" in resp.answer.lower()
        # Verify it did not hallucinate a launch date
        assert "2026" not in resp.answer and "launch" in resp.answer.lower()

    def test_04_ambiguous_evidence_handling(self):
        """When discussion is partial or unresolved, clearly reports ambiguous_incomplete status."""
        segments = [
            {"index": 0, "start": 0.0, "end": 25.0, "text": "We talked about using AWS or GCP, but we haven't decided yet and need another meeting."},
        ]
        snapshot = {"segments": segments, "full_text": segments[0]["text"]}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="[STATUS: AMBIGUOUS_INCOMPLETE] The team discussed AWS and GCP [00:00 - 00:25], but explicitly stated they haven't decided yet and need another meeting."
        )

        resp = ask_live_meeting_question(
            snapshot,
            "Which cloud provider did the team choose?",
            llm=mock_llm,
        )
        assert resp.state == AnswerState.AMBIGUOUS_INCOMPLETE
        assert "haven't decided yet" in resp.answer.lower()


class TestPostMeetingQA:
    """Unit and integration tests for post-meeting RAG synthesis, citations, and session isolation."""

    def test_05_post_meeting_confirmed_answer(self):
        """Verified answer synthesized across retrieved chunks with timestamps and excerpts."""
        doc1 = MagicMock(
            page_content="The team reviewed the deployment pipeline and confirmed Kubernetes is deployed.",
            metadata={"session_id": "sess_post_1", "start_time": 40.0, "end_time": 60.0, "timestamp_label": "[00:40 - 01:00]"}
        )
        retriever = MagicMock()
        retriever.invoke.return_value = [doc1]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="[STATUS: CONFIRMED] The deployment pipeline uses Kubernetes [00:40 - 01:00], as confirmed in the review."
        )

        resp = ask_post_meeting_question(
            rag_chain=None,
            question="What deployment technology was chosen?",
            session_id="sess_post_1",
            retriever=retriever,
            llm=mock_llm,
        )
        assert resp.state == AnswerState.CONFIRMED
        assert "Kubernetes" in resp.answer
        assert len(resp.citations) == 1
        assert resp.citations[0].timestamp_str == "00:40 - 01:00"

    def test_06_post_meeting_not_found(self):
        """When topic was not discussed in completed meeting, returns not_found_completed."""
        retriever = MagicMock()
        retriever.invoke.return_value = []

        resp = ask_post_meeting_question(
            rag_chain=None,
            question="What was said about quantum computing?",
            session_id="sess_post_1",
            retriever=retriever,
        )
        assert resp.state == AnswerState.NOT_FOUND_COMPLETED
        assert "could not find this information" in resp.answer.lower()

    def test_07_cross_session_protection(self):
        """If a chunk from Session A is returned during a query for Session B, access is blocked."""
        leak_doc = MagicMock(
            page_content="Secret financial numbers from Session A: Revenue was $5M.",
            metadata={"session_id": "session_A"}
        )
        retriever = MagicMock()
        retriever.invoke.return_value = [leak_doc]

        resp = ask_post_meeting_question(
            rag_chain=None,
            question="What was the revenue?",
            session_id="session_B",  # Querying for Session B!
            retriever=retriever,
        )
        assert "Security violation" in resp.answer
        assert "5M" not in resp.answer


class TestMeetingQAIntegration:
    """Integration tests verifying live recording and Q&A concurrency without thread blocking."""

    def test_08_live_qa_concurrent_with_recording(self):
        """Verify Q&A query does not interrupt or block live recording audio chunking."""
        transcriber = LiveTranscriber(mock_mode=True)
        chunker = LiveAudioChunker(transcriber, chunk_seconds=1.0, overlap_seconds=0.2)
        transcriber.start()

        sr = 16000
        pcm_block = np.zeros(sr, dtype=np.float32)

        # Feed 2 chunks
        chunker.feed_samples(pcm_block)
        chunker.feed_samples(pcm_block)
        time.sleep(0.1)

        snapshot = transcriber.transcript.get_snapshot()

        # Execute Live Q&A query while worker continues running
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="[STATUS: NOT_YET_MENTIONED] Topic has not yet been discussed."
        )

        t0 = time.perf_counter()
        resp = ask_live_meeting_question(snapshot, "Any blocker discussed?", llm=mock_llm)
        t_qa = time.perf_counter() - t0

        # Live Q&A should return without blocking worker
        assert resp.state == AnswerState.NOT_YET_MENTIONED
        assert t_qa < 1.0
        assert transcriber.is_running is True

        transcriber.stop(timeout=5.0)
        transcriber.reset()

    def test_09_chat_history_followup_handling(self):
        """Verify conversational chat history provides context for follow-up questions."""
        segments = [
            {"index": 0, "start": 0.0, "end": 10.0, "text": "Alice will prepare the presentation slides for the client."},
            {"index": 1, "start": 10.0, "end": 20.0, "text": "Bob will review the security checklist."},
        ]
        snapshot = {"segments": segments, "full_text": " ".join(s["text"] for s in segments)}

        history = [
            {"role": "user", "content": "Who is working on the slides?"},
            {"role": "assistant", "content": "Alice will prepare the presentation slides [00:00 - 00:10]."},
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="[STATUS: CONFIRMED] Bob will review the security checklist [00:10 - 00:20]."
        )

        resp = ask_live_meeting_question(
            snapshot,
            "What about Bob?",  # Contextual follow-up question
            chat_history=history,
            llm=mock_llm,
        )
        assert resp.state == AnswerState.CONFIRMED
        assert "Bob" in resp.answer
        assert len(resp.citations) == 1
        assert resp.citations[0].timestamp_str == "00:10 - 00:20"

    def test_10_timestamp_formatting_helpers(self):
        """Verify timestamp formatting and parsing utilities."""
        assert format_timestamp(0.0) == "00:00"
        assert format_timestamp(65.4) == "01:05"
        assert format_timestamp(3600.0) == "60:00"

        s1, e1 = parse_timestamp_range("[01:23 - 02:45]")
        assert s1 == 83.0
        assert e1 == 165.0

        s2, e2 = parse_timestamp_range("05:12")
        assert s2 == 312.0
        assert e2 == 312.0

    def test_11_hallucinated_citation_rejected(self):
        """If LLM generates a citation for a timestamp not present in the context, it is rejected."""
        context = "[00:00 - 00:10] Team members introduced themselves."
        answer = "The product launch is set for October [05:45]."  # 05:45 never occurred in context!

        citations = extract_citations_from_answer(answer, context, max_timestamp=10.0)
        assert len(citations) == 0  # Fabricated citation must be rejected!

    def test_12_future_timestamp_beyond_live_cutoff_rejected(self):
        """Citation with timestamp exceeding live cutoff must be discarded."""
        context = "[00:00 - 00:30] We are reviewing last week's metrics."
        answer = "The server migration will happen tonight [02:15]."

        citations = extract_citations_from_answer(answer, context, max_timestamp=30.0)
        assert len(citations) == 0

    def test_13_missing_session_id_blocked_in_post_meeting(self):
        """Calling ask_post_meeting_question with missing or empty session_id blocks retrieval."""
        retriever = MagicMock()
        resp = ask_post_meeting_question(
            rag_chain=None,
            question="What is the roadmap?",
            session_id="",  # Missing session_id!
            retriever=retriever,
        )
        assert "Session ID missing or invalid" in resp.answer
        assert retriever.invoke.call_count == 0  # No query executed!
