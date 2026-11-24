"""
Hierarchical Meeting Intelligence Module (Phase 3).

Defines structured Pydantic schemas and hierarchical extraction engines
capable of analyzing entire meetings (short, medium, and multi-hour long meetings)
with chronological context preservation, cross-chunk topic consolidation,
and rigorous transcript citation verification.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.meeting_qa import Citation, format_timestamp, parse_timestamp_range


# --------------------------------------------------------------------------
# Structured Pydantic Schemas for Comprehensive Meeting Intelligence
# --------------------------------------------------------------------------

class TopicDiscussionPoint(BaseModel):
    point: str = Field(description="Key point or argument discussed under this topic")
    speaker: str = Field(default="Unknown", description="Speaker name or label (e.g. Speaker 1) who stated it")
    timestamp: str = Field(default="Not specified", description="Timestamp e.g. [02:15] or [02:15 - 02:45]")


class QuestionAnswerItem(BaseModel):
    question: str = Field(description="Question raised during the meeting")
    asked_by: str = Field(default="Unknown", description="Person who asked the question")
    answer: str = Field(default="Not mentioned in the meeting", description="Answer provided, or 'Not mentioned in the meeting'")
    answered_by: str = Field(default="Unknown", description="Person who answered")
    timestamp: str = Field(default="Not specified", description="Timestamp of the Q&A exchange")


class ProblemSolutionItem(BaseModel):
    problem: str = Field(description="Problem, challenge, or roadblock identified")
    proposed_solution: str = Field(default="Not mentioned in the meeting", description="Proposed solution, workaround, or mitigation")
    status: str = Field(default="Open", description="Status e.g. 'Open', 'Resolved', 'Under Investigation'")
    timestamp: str = Field(default="Not specified", description="Timestamp when discussed")


class ConfirmedDecision(BaseModel):
    decision: str = Field(description="Specific, agreed-upon decision (not a mere suggestion or discussion)")
    rationale: str = Field(default="Not specified", description="Underlying reason or context for this decision")
    decision_makers: Any = Field(default="Unknown", description="Participants or consensus owner who agreed")
    timestamp: str = Field(default="Not specified", description="Timestamp range where decision was made")
    evidence_quote: str = Field(default="", description="Exact transcript quote proving this decision")

    def __init__(self, **data):
        dm = data.get("decision_makers")
        if isinstance(dm, list):
            data["decision_makers"] = ", ".join(str(x) for x in dm)
        super().__init__(**data)


class ActionItemDetail(BaseModel):
    task: str = Field(description="Concrete action item or task to be executed")
    assignee: str = Field(default="Unknown", description="Explicit person or team responsible, or 'Unknown'")
    deadline: str = Field(default="Not specified", description="Explicit due date/time or 'Not specified'")
    priority: str = Field(default="Medium", description="'High', 'Medium', or 'Low' if explicitly stated; else 'Medium'")
    timestamp: str = Field(default="Not specified", description="Timestamp where this task was assigned")
    evidence_quote: str = Field(default="", description="Transcript quote supporting this assignment")


class MeetingTopic(BaseModel):
    topic_title: str = Field(description="Descriptive title of the discussion topic")
    topic_description: str = Field(description="Detailed overview of what was discussed under this topic")
    subtopics: List[str] = Field(default_factory=list, description="Subtopics or related branches")
    important_points: List[str] = Field(default_factory=list, description="Important statements and arguments")
    suggestions: List[str] = Field(default_factory=list, description="Ideas or suggestions proposed by participants")
    questions_and_answers: List[QuestionAnswerItem] = Field(default_factory=list, description="Questions and answers for this topic")
    problems_and_solutions: List[ProblemSolutionItem] = Field(default_factory=list, description="Roadblocks and solutions")
    decisions: List[ConfirmedDecision] = Field(default_factory=list, description="Decisions relevant to this topic")
    action_items: List[ActionItemDetail] = Field(default_factory=list, description="Action items originating from this topic")
    start_time: str = Field(default="00:00", description="Start timestamp of this topic discussion")
    end_time: str = Field(default="00:00", description="End timestamp of this topic discussion")


class MeetingMinutesSection(BaseModel):
    meeting_type: str = Field(default="Organizational / Technical Meeting", description="Type of meeting e.g. Academic, Sprint, Business")
    chair_or_facilitator: str = Field(default="Unknown", description="Facilitator or lead, or 'Unknown'")
    attendees_mentioned: List[str] = Field(default_factory=list, description="Explicitly mentioned participant names/roles")
    formal_minutes_text: str = Field(description="Professional, formal Minutes of Meeting (MoM) in academic/business standard")


class ComprehensiveMeetingReport(BaseModel):
    title: str = Field(description="Professional meeting title reflecting the actual content")
    meeting_overview: str = Field(description="Clear explanation of the meeting purpose, context, and key outcomes")
    executive_summary: str = Field(description="Concise executive summary covering critical takeaways")
    detailed_meeting_summary: str = Field(description="Thorough, chronological summary sufficient for an absentee to understand everything discussed")
    topics_discussed: List[MeetingTopic] = Field(default_factory=list, description="All meaningful topics discussed during the meeting")
    key_decisions: List[ConfirmedDecision] = Field(default_factory=list, description="Global list of confirmed decisions")
    action_items: List[ActionItemDetail] = Field(default_factory=list, description="Global list of actionable items with assignees & deadlines")
    questions_and_answers: List[QuestionAnswerItem] = Field(default_factory=list, description="Global list of questions and answers")
    problems_and_solutions: List[ProblemSolutionItem] = Field(default_factory=list, description="Global list of problems and solutions")
    important_announcements: List[str] = Field(default_factory=list, description="Official announcements, organizational updates, or schedule notices")
    pending_tasks: List[str] = Field(default_factory=list, description="Unresolved topics, open questions, or tasks lacking clarity")
    next_steps: List[str] = Field(default_factory=list, description="Future planned milestones and immediate follow-ups")
    next_meeting_details: Any = Field(default="Not mentioned in the meeting", description="Date, time, or agenda of next meeting if explicitly stated; else 'Not mentioned in the meeting'")
    minutes_of_meeting: MeetingMinutesSection = Field(description="Formal Minutes of Meeting")

    def __init__(self, **data):
        nm = data.get("next_meeting_details")
        if isinstance(nm, dict):
            parts = [f"{k}: {v}" for k, v in nm.items() if v]
            data["next_meeting_details"] = "; ".join(parts)
        elif isinstance(nm, list):
            data["next_meeting_details"] = "; ".join(str(x) for x in nm)
        super().__init__(**data)


# --------------------------------------------------------------------------
# LLM Initialization & Utilities
# --------------------------------------------------------------------------

def get_mistral_llm(temperature: float = 0.2):
    """Retrieve ChatMistralAI instance with exponential backoff configuration."""
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is missing. Please add it to your .env file or environment.")
    model = os.getenv("MISTRAL_MODEL", "open-mistral-nemo").strip()
    return ChatMistralAI(
        model=model,
        mistral_api_key=api_key,
        temperature=temperature,
        max_retries=5,
    )


def invoke_chain_with_backoff(chain, inputs, max_retries: int = 5, initial_delay: float = 3.0):
    """Execute runnable chain with exponential backoff on HTTP 429."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str or "rate_limited" in err_str:
                if attempt < max_retries - 1:
                    print(f"[!] Rate limit (429) encountered. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise


# --------------------------------------------------------------------------
# Evidence Verification & Hallucination Prevention Helpers
# --------------------------------------------------------------------------

def verify_and_align_citation(
    claim_text: str,
    evidence_quote: str,
    candidate_timestamp: str,
    segments: Optional[List[dict]],
    max_duration: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Validates that candidate evidence actually exists in the transcript segments
    and that candidate timestamps are valid and within bounds.
    Returns: (verified_timestamp_str, verified_evidence_quote)
    """
    if not segments:
        ts = candidate_timestamp if candidate_timestamp and candidate_timestamp != "Not specified" else "00:00"
        return ts, evidence_quote

    clean_quote = (evidence_quote or "").strip().lower()
    clean_claim = (claim_text or "").strip().lower()

    # 1. First search for direct quote match in segments
    if clean_quote and len(clean_quote) > 8:
        for s in segments:
            seg_txt = s.get("text", "").strip()
            if clean_quote in seg_txt.lower() or seg_txt.lower() in clean_quote:
                s_start = s.get("start", 0.0)
                s_end = s.get("end", 0.0)
                return f"{format_timestamp(s_start)} - {format_timestamp(s_end)}", seg_txt

    # 2. Check candidate timestamp
    if candidate_timestamp and candidate_timestamp != "Not specified":
        start_sec, end_sec = parse_timestamp_range(candidate_timestamp)
        if max_duration is None or start_sec <= (max_duration + 2.0):
            for s in segments:
                s_start = s.get("start", 0.0)
                s_end = s.get("end", 0.0)
                if abs(s_start - start_sec) <= 3.0:
                    seg_txt = s.get("text", "").strip()
                    return f"{format_timestamp(s_start)} - {format_timestamp(s_end)}", seg_txt

    # 3. Fuzzy lexical word overlap search against segments
    claim_words = set(re.findall(r"\w{3,}", clean_claim))
    if claim_words:
        best_seg = None
        best_overlap = 0
        for s in segments:
            seg_txt = s.get("text", "").strip()
            seg_words = set(re.findall(r"\w{3,}", seg_txt.lower()))
            overlap = len(claim_words & seg_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_seg = s
        if best_seg and best_overlap >= 2:
            s_start = best_seg.get("start", 0.0)
            s_end = best_seg.get("end", 0.0)
            return f"{format_timestamp(s_start)} - {format_timestamp(s_end)}", best_seg.get("text", "").strip()

    # Fallback if no specific segment was validated
    ts = candidate_timestamp if (candidate_timestamp and candidate_timestamp != "Not specified") else "00:00"
    return ts, evidence_quote or "Not specified in transcript"


# --------------------------------------------------------------------------
# Single-Pass & Hierarchical Extraction Engine
# --------------------------------------------------------------------------

SYSTEM_INTELLIGENCE_PROMPT = """You are a Principal AI Meeting Secretary and Senior Intelligence Analyst comparable to Otter.ai and Fireflies.ai.
Analyze the provided meeting transcript with meticulous depth, accuracy, and chronological structure.

CRITICAL INSTRUCTIONS & ACCURACY RULES:
1. COMPLETE MEETING UNDERSTANDING: Analyze the entire meeting conversation from beginning to end. Never invent facts, participant names, dates, numbers, or deadlines.
2. ABSENCE OF INFORMATION: If a field is not explicitly stated in the transcript, strictly output "Not mentioned in the meeting" or "Unknown". Do NOT speculate.
3. DISTINGUISH DECISIONS VS SUGGESTIONS: Only record confirmed, agreed-upon decisions under 'key_decisions'. Put preliminary ideas or suggestions under participant 'suggestions' within the relevant topic.
4. ACTION ITEMS: Every action item must have a clear task, explicit assignee (or "Unknown"), deadline (or "Not specified"), priority, timestamp, and supporting transcript quote.
5. TOPIC BREAKDOWN: Break down every distinct topic discussed. Provide topic title, detailed description, discussion points, suggestions, questions, problems, solutions, and timestamps.
6. MINUTES OF MEETING: Formulate formal, publication-ready Minutes of Meeting (MoM) following organizational and academic executive standards.
7. TRANSLATED/MULTILINGUAL ALIGNMENT: The transcript provided is in English (or aligned English translation of Hindi/Nepali discussions). Preserve all technical terms, names, and metrics faithfully.

You MUST respond ONLY with valid JSON matching the exact schema below without any markdown fences, explanations, or commentary.

JSON Schema:
{
  "title": "Meeting Title",
  "meeting_overview": "Comprehensive overview of purpose, context, and key results.",
  "executive_summary": "Concise executive summary covering critical takeaways.",
  "detailed_meeting_summary": "Detailed narrative of all discussions from beginning, middle, to end.",
  "topics_discussed": [
    {
      "topic_title": "Topic Title",
      "topic_description": "Thorough description of what was discussed.",
      "subtopics": ["Subtopic 1", "Subtopic 2"],
      "important_points": ["Key statement or argument"],
      "suggestions": ["Participant suggestion"],
      "questions_and_answers": [
        {
          "question": "Question asked?",
          "asked_by": "Speaker 1 or Name",
          "answer": "Answer given or Not mentioned in the meeting",
          "answered_by": "Speaker 2 or Name",
          "timestamp": "[MM:SS]"
        }
      ],
      "problems_and_solutions": [
        {
          "problem": "Problem described",
          "proposed_solution": "Solution or Not mentioned in the meeting",
          "status": "Open / Resolved",
          "timestamp": "[MM:SS]"
        }
      ],
      "decisions": [
        {
          "decision": "Agreed decision",
          "rationale": "Reason for decision",
          "decision_makers": "Participants who agreed",
          "timestamp": "[MM:SS]",
          "evidence_quote": "Exact quote from transcript"
        }
      ],
      "action_items": [
        {
          "task": "Task to complete",
          "assignee": "Person or Unknown",
          "deadline": "Deadline or Not specified",
          "priority": "High / Medium / Low",
          "timestamp": "[MM:SS]",
          "evidence_quote": "Exact quote from transcript"
        }
      ],
      "start_time": "00:00",
      "end_time": "05:00"
    }
  ],
  "key_decisions": [
    {
      "decision": "Agreed decision",
      "rationale": "Reason for decision",
      "decision_makers": "Participants who agreed",
      "timestamp": "[MM:SS]",
      "evidence_quote": "Exact quote"
    }
  ],
  "action_items": [
    {
      "task": "Action item task",
      "assignee": "Assignee or Unknown",
      "deadline": "Deadline or Not specified",
      "priority": "High / Medium / Low",
      "timestamp": "[MM:SS]",
      "evidence_quote": "Quote"
    }
  ],
  "questions_and_answers": [
    {
      "question": "Question",
      "asked_by": "Name or Unknown",
      "answer": "Answer",
      "answered_by": "Name or Unknown",
      "timestamp": "[MM:SS]"
    }
  ],
  "problems_and_solutions": [
    {
      "problem": "Problem",
      "proposed_solution": "Solution",
      "status": "Open",
      "timestamp": "[MM:SS]"
    }
  ],
  "important_announcements": ["Announcement 1"],
  "pending_tasks": ["Unresolved topic or follow-up"],
  "next_steps": ["Next milestone or planned follow-up"],
  "next_meeting_details": "Date/Time/Agenda or Not mentioned in the meeting",
  "minutes_of_meeting": {
    "meeting_type": "Organizational / Technical Meeting",
    "chair_or_facilitator": "Facilitator or Unknown",
    "attendees_mentioned": ["Participant 1", "Participant 2"],
    "formal_minutes_text": "Professional formal minutes..."
  }
}
"""


def _clean_json_response(raw_text: str) -> str:
    """Clean markdown code fences and extraneous text from JSON response."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_hierarchical_intelligence(
    transcript: str,
    segments: Optional[List[dict]] = None,
    duration_seconds: float = 0.0,
    llm: Any = None,
) -> ComprehensiveMeetingReport:
    """
    Extract comprehensive meeting intelligence using a unified Pydantic schema.
    For short-to-medium meetings (< 45,000 characters), executes a high-fidelity single pass.
    For long meetings (>= 45,000 characters), utilizes chronological topic chunking and map-reduce consolidation.
    """
    if not transcript or not transcript.strip():
        # Return empty structured report
        return ComprehensiveMeetingReport(
            title="Empty Meeting Transcript",
            meeting_overview="No audible speech or transcript was recorded.",
            executive_summary="No content available.",
            detailed_meeting_summary="No discussion was captured.",
            topics_discussed=[],
            key_decisions=[],
            action_items=[],
            questions_and_answers=[],
            problems_and_solutions=[],
            important_announcements=[],
            pending_tasks=[],
            next_steps=[],
            next_meeting_details="Not mentioned in the meeting",
            minutes_of_meeting=MeetingMinutesSection(
                meeting_type="General Meeting",
                chair_or_facilitator="Unknown",
                attendees_mentioned=[],
                formal_minutes_text="No discussion recorded.",
            ),
        )

    if llm is None:
        llm = get_mistral_llm()

    # Format transcript with timestamps if available
    annotated_transcript = transcript
    if segments and len(segments) > 0:
        lines = []
        for s in segments:
            t_str = f"[{format_timestamp(s.get('start', 0.0))} - {format_timestamp(s.get('end', 0.0))}]"
            spk = s.get("speaker")
            spk_str = f" {spk}:" if spk else ""
            txt = s.get("english_text") or s.get("text", "")
            lines.append(f"{t_str}{spk_str} {txt}")
        annotated_transcript = "\n".join(lines)

    # Context window check: ~45k characters (~11k tokens) fits cleanly within Mistral Nemo context
    from langchain_core.messages import SystemMessage, HumanMessage
    if len(annotated_transcript) < 45000:
        messages = [
            SystemMessage(content=SYSTEM_INTELLIGENCE_PROMPT),
            HumanMessage(content=f"Here is the meeting transcript:\n\n{annotated_transcript}\n\nGenerate the complete JSON meeting intelligence."),
        ]
        raw_res = invoke_chain_with_backoff(llm, messages)
        raw_text = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
        clean_json = _clean_json_response(raw_text)

        try:
            data = json.loads(clean_json, strict=False)
        except Exception as parse_err:
            print(f"[!] JSON parsing error in single pass: {parse_err}. Attempting recovery...")
            # Attempt basic regex recovery for JSON object
            match = re.search(r"(\{.*\})", clean_json, re.DOTALL)
            if match:
                data = json.loads(match.group(1), strict=False)
            else:
                raise parse_err

        report = ComprehensiveMeetingReport(**data)
    else:
        # Hierarchical Long-Meeting Chunking & Consolidation
        print(f"[*] Processing long meeting transcript ({len(annotated_transcript)} chars) via hierarchical chunking...")
        report = _hierarchical_long_meeting_extraction(annotated_transcript, segments, duration_seconds, llm)

    # Post-process: Verify all decisions and action items against transcript segments
    if segments:
        for item in report.action_items:
            item.timestamp, item.evidence_quote = verify_and_align_citation(
                claim_text=item.task,
                evidence_quote=item.evidence_quote,
                candidate_timestamp=item.timestamp,
                segments=segments,
                max_duration=duration_seconds,
            )
        for dec in report.key_decisions:
            dec.timestamp, dec.evidence_quote = verify_and_align_citation(
                claim_text=dec.decision,
                evidence_quote=dec.evidence_quote,
                candidate_timestamp=dec.timestamp,
                segments=segments,
                max_duration=duration_seconds,
            )

    return report


def _hierarchical_long_meeting_extraction(
    annotated_transcript: str,
    segments: Optional[List[dict]],
    duration_seconds: float,
    llm: Any,
) -> ComprehensiveMeetingReport:
    """
    Divide long transcripts into chronological chunks, extract section intelligence from each,
    and consolidate into a unified ComprehensiveMeetingReport without losing beginning/middle/end info.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=30000, chunk_overlap=2000)
    chunks = splitter.split_text(annotated_transcript)

    partial_reports: List[dict] = []
    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INTELLIGENCE_PROMPT + "\nNote: This is one chronological section of a long meeting. Extract topics, decisions, actions, and details from this section."),
        ("human", "Meeting Section Transcript:\n\n{text}\n\nGenerate JSON intelligence for this section."),
    ])
    chain = chunk_prompt | llm | StrOutputParser()

    for i, ch in enumerate(chunks):
        print(f"[*] Processing long meeting section {i + 1}/{len(chunks)}...")
        messages = [
            SystemMessage(content=SYSTEM_INTELLIGENCE_PROMPT + "\nNote: This is one chronological section of a long meeting. Extract topics, decisions, actions, and details from this section."),
            HumanMessage(content=f"Meeting Section Transcript:\n\n{ch}\n\nGenerate JSON intelligence for this section."),
        ]
        raw_res = invoke_chain_with_backoff(llm, messages)
        raw_text = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
        try:
            p_data = json.loads(_clean_json_response(raw_text))
            partial_reports.append(p_data)
        except Exception as e:
            print(f"[!] Warning: Chunk {i + 1} extraction notice: {e}")

    # Consolidate partial reports into final ComprehensiveMeetingReport
    if not partial_reports:
        return extract_hierarchical_intelligence(annotated_transcript[:40000], segments, duration_seconds, llm)

    cons_messages = [
        SystemMessage(content="You are a Chief Meeting Editor. Combine the following JSON sections of a long meeting into a single unified JSON ComprehensiveMeetingReport. Consolidate repeating topics, remove duplicates, maintain chronological integrity, and ensure complete coverage."),
        HumanMessage(content=f"Partial meeting reports:\n\n{json.dumps(partial_reports)}\n\nProduce the consolidated final JSON meeting report matching the schema."),
    ]
    raw_final = invoke_chain_with_backoff(llm, cons_messages)
    raw_final_text = raw_final.content if hasattr(raw_final, "content") else str(raw_final)
    final_data = json.loads(_clean_json_response(raw_final_text))
    return ComprehensiveMeetingReport(**final_data)
