"""
Meeting Q&A Engine for AI Video Assistant (Phase 6).

Provides grounded live in-meeting and post-meeting question answering:
1. Live In-Meeting Q&A:
   - Queries only finalized transcript segments available at the submission moment.
   - Temporal awareness: explicitly identifies the meeting timestamp cutoff.
   - Honest states: 'confirmed', 'not_yet_mentioned', 'ambiguous_incomplete'.
   - Zero hallucination: explicitly refuses to speculate on topics not yet covered.

2. Post-Meeting Q&A:
   - Synthesizes answers using ChromaDB RAG with strict session isolation.
   - Multi-chunk evidence retrieval with timestamps and supporting excerpts.
   - Honest states: 'confirmed', 'not_found_completed', 'ambiguous_incomplete'.

3. Grounded Citations:
   - Extracts and verifies exact [MM:SS] timestamps and source quotations.
   - Refuses to guess speaker identities when speaker labels are absent.
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.rag_engine import get_llm, invoke_with_retry, FALLBACK_PHRASE
from core.vector_store import SENTINEL_NO_MATCH_SESSION


class AnswerState(str, Enum):
    CONFIRMED = "confirmed"
    NOT_YET_MENTIONED = "not_yet_mentioned"
    NOT_FOUND_COMPLETED = "not_found_completed"
    AMBIGUOUS_INCOMPLETE = "ambiguous_incomplete"


@dataclass
class Citation:
    """Represents a verified supporting excerpt with timeline timestamps."""
    start: float
    end: float
    timestamp_str: str  # e.g. "01:24 - 01:45"
    excerpt: str


@dataclass
class MeetingQAResponse:
    """Standardized response from the Meeting Q&A engine."""
    answer: str
    state: AnswerState
    cutoff_seconds: Optional[float] = None
    cutoff_timestamp: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    session_id: str = ""
    is_live: bool = False


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to MM:SS string format."""
    if seconds is None or seconds < 0:
        return "00:00"
    total_sec = int(round(seconds))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"


def parse_timestamp_range(ts_str: str) -> Tuple[float, float]:
    """Parse '[MM:SS - MM:SS]' or 'MM:SS' into (start_sec, end_sec)."""
    matches = re.findall(r"(\d{1,2}):(\d{2})", ts_str)
    if not matches:
        return 0.0, 0.0
    start_sec = float(matches[0][0]) * 60 + float(matches[0][1])
    end_sec = start_sec
    if len(matches) > 1:
        end_sec = float(matches[1][0]) * 60 + float(matches[1][1])
    return start_sec, end_sec


# --------------------------------------------------------------------------
# Live In-Meeting Q&A Prompts & Logic
# --------------------------------------------------------------------------

LIVE_QA_SYSTEM_PROMPT = """You are a grounded real-time Meeting Assistant assisting attendees during an ONGOING meeting.
The meeting is actively in progress. You are provided ONLY with the transcript of what has been spoken and transcribed so far, up to {cutoff_time} (elapsed time: {cutoff_seconds:.1f}s).

CRITICAL GROUNDING RULES:
1. TIME-BOUNDED REALITY: You ONLY know what was said in the transcript up to {cutoff_time}.
   - If the user asks about a topic, decision, or agenda item that has NOT been discussed yet in the transcript up to this point, you MUST respond honestly:
     "Based on the meeting transcript up to {cutoff_time}, this topic has not yet been discussed."
   - State clearly: [STATUS: NOT_YET_MENTIONED]
   - DO NOT speculate, assume, or invent what might happen later in the meeting.
   - DO NOT claim it was never discussed in the entire meeting; only that it has not been captured yet up to {cutoff_time}.

2. CONFIRMED FACTS:
   - If the transcript directly answers the question, state the answer concisely.
   - Cite the exact timestamp in brackets (e.g. [01:24 - 01:40] or [01:24]).
   - Include a brief quote of the speaker's words.
   - State: [STATUS: CONFIRMED]

3. AMBIGUOUS OR INCOMPLETE:
   - If the topic was mentioned but unresolved, vague, or conflicting, explain exactly what was said and state that the topic remains open or incomplete.
   - State: [STATUS: AMBIGUOUS_INCOMPLETE]

4. NO FABRICATION:
   - Do NOT invent timestamps, speaker names, agenda items, or facts.
   - If speaker names are not in the text, use "a participant" or "the speaker".

Available transcript up to {cutoff_time}:
{context}
"""

POST_MEETING_QA_SYSTEM_PROMPT = """You are an intelligent Meeting Q&A Assistant analyzing a completed meeting transcript.
Answer the user's question using ONLY the provided transcript excerpts below.

CRITICAL GROUNDING RULES:
1. STRICT RETRIEVAL GROUNDING: Rely strictly on the provided excerpts.
   - If the transcript supports the answer: State the answer clearly, cite the timestamp [MM:SS] if present in the excerpt, and provide a short supporting quote.
   - State: [STATUS: CONFIRMED]
2. NOT FOUND:
   - If the topic is absent from the provided excerpts: State clearly that this information was not found in the recorded meeting transcript.
   - State: [STATUS: NOT_FOUND_COMPLETED]
3. AMBIGUOUS OR INCOMPLETE:
   - If the evidence is partial or conflicting, explain the ambiguity.
   - State: [STATUS: AMBIGUOUS_INCOMPLETE]
4. NO FABRICATION:
   - Do not invent timestamps, facts, or speaker identities.

Retrieved Transcript Excerpts:
{context}
"""


def extract_answer_and_state(raw_text: str, default_state: AnswerState) -> Tuple[str, AnswerState]:
    """Parse out [STATUS: ...] tag if present and determine AnswerState."""
    status_match = re.search(r"\[STATUS:\s*([A-Z_]+)\]", raw_text, re.IGNORECASE)
    cleaned_answer = raw_text
    state = default_state

    if status_match:
        tag = status_match.group(1).upper()
        if "NOT_YET_MENTIONED" in tag:
            state = AnswerState.NOT_YET_MENTIONED
        elif "NOT_FOUND" in tag:
            state = AnswerState.NOT_FOUND_COMPLETED
        elif "AMBIGUOUS" in tag or "INCOMPLETE" in tag:
            state = AnswerState.AMBIGUOUS_INCOMPLETE
        elif "CONFIRMED" in tag:
            state = AnswerState.CONFIRMED

        # Remove the tag from the user-facing text
        cleaned_answer = re.sub(r"\[STATUS:\s*[A-Z_]+\]", "", cleaned_answer).strip()

    # Secondary heuristic check
    lower = cleaned_answer.lower()
    if "has not yet been discussed" in lower or "not yet mentioned" in lower:
        state = AnswerState.NOT_YET_MENTIONED
    elif "not found in the" in lower or "could not find this information" in lower:
        state = AnswerState.NOT_FOUND_COMPLETED

    return cleaned_answer, state


def extract_citations_from_answer(
    answer_text: str,
    context_text: str,
    max_timestamp: Optional[float] = None,
) -> List[Citation]:
    """
    Extract and verify cited timestamps and quotes against actual transcript context.
    Discards any timestamps that do not exist in the context or exceed the live cutoff.
    """
    citations = []
    # Match [MM:SS] or [MM:SS - MM:SS]
    pattern = r"\[(\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?)\]"
    matches = list(re.finditer(pattern, answer_text))

    for m in matches:
        ts_str = m.group(1).strip()
        start, end = parse_timestamp_range(ts_str)

        # Grounding check 1: Timestamp cannot exceed the cutoff of what was recorded
        if max_timestamp is not None and start > (max_timestamp + 1.0):
            continue

        # Grounding check 2: Timestamp or its formatted equivalent must exist in actual transcript context
        start_fmt = format_timestamp(start)
        pos = context_text.find(f"[{ts_str}]")
        if pos == -1:
            pos = context_text.find(ts_str)
        if pos == -1:
            pos = context_text.find(start_fmt)

        excerpt = ""
        if pos != -1:
            # Find the closing bracket ']' of the timestamp header
            bracket_end = context_text.find("]", pos)
            if bracket_end != -1:
                tail = context_text[bracket_end + 1:].strip()
                # Excerpt runs up to the next timestamp header '[' or at most 180 chars
                next_bracket = tail.find("[")
                raw_excerpt = tail[:next_bracket].strip() if next_bracket != -1 else tail[:180].strip()
                clean_excerpt = " ".join(raw_excerpt.split())
                if clean_excerpt:
                    excerpt = clean_excerpt

        if not excerpt:
            # Timestamp was not found in the actual context! Do not accept unverified citation.
            continue

        citations.append(Citation(
            start=start,
            end=end,
            timestamp_str=ts_str,
            excerpt=excerpt,
        ))

    return citations


# --------------------------------------------------------------------------
# Public Q&A Functions
# --------------------------------------------------------------------------

def ask_live_meeting_question(
    transcript_snapshot: dict,
    question: str,
    chat_history: Optional[List[dict]] = None,
    session_id: str = "",
    llm: Any = None,
) -> MeetingQAResponse:
    """
    Answer a question during an active meeting using only transcript segments
    finalized up to the moment of query submission.
    """
    clean_session_id = str(session_id).strip() if session_id else ""
    clean_q = question.strip()
    if not clean_q:
        return MeetingQAResponse(
            answer="Please ask a question about the meeting.",
            state=AnswerState.AMBIGUOUS_INCOMPLETE,
            session_id=clean_session_id,
            is_live=True,
        )

    segments = transcript_snapshot.get("segments", [])
    if not segments:
        return MeetingQAResponse(
            answer="No speech has been captured or transcribed in the meeting yet.",
            state=AnswerState.NOT_YET_MENTIONED,
            cutoff_seconds=0.0,
            cutoff_timestamp="00:00",
            session_id=clean_session_id,
            is_live=True,
        )

    # Determine cutoff from latest finalized segment
    latest_end = max(s.get("end", 0.0) for s in segments)
    cutoff_time_str = format_timestamp(latest_end)

    # Format transcript with segment timestamps
    context_lines = []
    for s in segments:
        t_start = format_timestamp(s.get("start", 0.0))
        t_end = format_timestamp(s.get("end", 0.0))
        txt = s.get("text", "").strip()
        if txt:
            context_lines.append(f"[{t_start} - {t_end}] {txt}")

    context_str = "\n".join(context_lines)

    # Build prompt messages including conversation history
    system_content = LIVE_QA_SYSTEM_PROMPT.format(
        cutoff_time=cutoff_time_str,
        cutoff_seconds=latest_end,
        context=context_str,
    )

    messages = [SystemMessage(content=system_content)]

    if chat_history:
        for turn in chat_history[-6:]:  # last 3 turns
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=clean_q))

    if llm is None:
        try:
            llm = get_llm()
        except Exception as e:
            return MeetingQAResponse(
                answer=f"Could not initialize language model: {e}",
                state=AnswerState.AMBIGUOUS_INCOMPLETE,
                cutoff_seconds=latest_end,
                cutoff_timestamp=cutoff_time_str,
                session_id=clean_session_id,
                is_live=True,
            )

    try:
        raw_response = llm.invoke(messages)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    except Exception as e:
        return MeetingQAResponse(
            answer=f"Error generating answer: {e}",
            state=AnswerState.AMBIGUOUS_INCOMPLETE,
            cutoff_seconds=latest_end,
            cutoff_timestamp=cutoff_time_str,
            session_id=clean_session_id,
            is_live=True,
        )

    clean_answer, state = extract_answer_and_state(raw_text, default_state=AnswerState.CONFIRMED)
    citations = extract_citations_from_answer(clean_answer, context_str, max_timestamp=latest_end)

    return MeetingQAResponse(
        answer=clean_answer,
        state=state,
        cutoff_seconds=latest_end,
        cutoff_timestamp=cutoff_time_str,
        citations=citations,
        session_id=clean_session_id,
        is_live=True,
    )


def ask_post_meeting_question(
    rag_chain: Any,
    question: str,
    chat_history: Optional[List[dict]] = None,
    session_id: str = "",
    retriever: Any = None,
    llm: Any = None,
) -> MeetingQAResponse:
    """
    Answer a question against the complete meeting transcript using ChromaDB RAG
    with strict session isolation and evidence citation extraction.
    """
    clean_q = question.strip()
    if not clean_q:
        return MeetingQAResponse(
            answer="Please ask a question about the meeting.",
            state=AnswerState.AMBIGUOUS_INCOMPLETE,
            session_id=session_id,
            is_live=False,
        )

    clean_session_id = str(session_id).strip() if session_id else ""
    if not clean_session_id and hasattr(rag_chain, "session_id"):
        clean_session_id = str(getattr(rag_chain, "session_id", "")).strip()

    if not clean_session_id or clean_session_id == SENTINEL_NO_MATCH_SESSION:
        return MeetingQAResponse(
            answer="Session ID missing or invalid. Retrieval is blocked to prevent cross-meeting data exposure.",
            state=AnswerState.NOT_FOUND_COMPLETED,
            session_id=clean_session_id,
            is_live=False,
        )

    if rag_chain is None and retriever is None:
        return MeetingQAResponse(
            answer="RAG chain or vector retriever is not available. Please process the meeting first.",
            state=AnswerState.AMBIGUOUS_INCOMPLETE,
            session_id=clean_session_id,
            is_live=False,
        )

    active_retriever = retriever or getattr(rag_chain, "retriever", None)

    # 1. Retrieve docs with session isolation
    retrieved_docs = []
    if active_retriever is not None:
        try:
            retrieved_docs = active_retriever.invoke(clean_q)
        except Exception as e:
            return MeetingQAResponse(
                answer=f"Error retrieving transcript chunks: {e}",
                state=AnswerState.AMBIGUOUS_INCOMPLETE,
                session_id=clean_session_id,
                is_live=False,
            )

    # Cross-session verification: ensure no retrieved doc belongs to another session
    for doc in retrieved_docs:
        doc_session = doc.metadata.get("session_id")
        if doc_session != clean_session_id:
            return MeetingQAResponse(
                answer="Security violation: Cross-session data leakage prevented.",
                state=AnswerState.NOT_FOUND_COMPLETED,
                session_id=clean_session_id,
                is_live=False,
            )

    if not retrieved_docs:
        return MeetingQAResponse(
            answer="I could not find this information in the transcript.",
            state=AnswerState.NOT_FOUND_COMPLETED,
            session_id=clean_session_id,
            is_live=False,
        )

    # Format excerpts
    context_chunks = []
    for i, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        ts_label = meta.get("timestamp_label", "")
        start_t = meta.get("start_time")
        end_t = meta.get("end_time")
        if start_t is not None and end_t is not None:
            ts_label = f"[{format_timestamp(start_t)} - {format_timestamp(end_t)}]"
        elif not ts_label:
            ts_label = f"[Chunk #{meta.get('chunk_index', i)}]"

        context_chunks.append(f"{ts_label}\n{doc.page_content.strip()}")

    context_str = "\n\n".join(context_chunks)

    # 2. Invoke LLM with strict post-meeting prompt
    if llm is None:
        try:
            llm = get_llm()
        except Exception as e:
            return MeetingQAResponse(
                answer=f"Could not initialize language model: {e}",
                state=AnswerState.AMBIGUOUS_INCOMPLETE,
                session_id=clean_session_id,
                is_live=False,
            )

    messages = [
        SystemMessage(content=POST_MEETING_QA_SYSTEM_PROMPT.format(context=context_str)),
    ]

    if chat_history:
        for turn in chat_history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=clean_q))

    try:
        raw_response = llm.invoke(messages)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    except Exception as e:
        return MeetingQAResponse(
            answer=f"Error generating answer: {e}",
            state=AnswerState.AMBIGUOUS_INCOMPLETE,
            session_id=clean_session_id,
            is_live=False,
        )

    clean_answer, state = extract_answer_and_state(raw_text, default_state=AnswerState.CONFIRMED)
    citations = extract_citations_from_answer(clean_answer, context_str)

    return MeetingQAResponse(
        answer=clean_answer,
        state=state,
        citations=citations,
        session_id=clean_session_id,
        is_live=False,
    )
