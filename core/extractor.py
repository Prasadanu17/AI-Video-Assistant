# Actionable items, decisions, questions (Phase 7 Meeting Intelligence)

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.meeting_qa import Citation, format_timestamp, parse_timestamp_range


# --------------------------------------------------------------------------
# Structured Intelligence Models
# --------------------------------------------------------------------------

@dataclass
class ActionItem:
    """Represents a structured action item with owner, priority, deadline, and citation."""
    task: str
    owner: str = "Unknown"
    priority: str = "Medium"  # High, Medium, Low
    deadline: str = "Not specified"
    citation: Optional[Citation] = None
    status: str = "Open"


@dataclass
class MeetingDecision:
    """Represents a key meeting decision with context rationale and consensus."""
    decision: str
    context_rationale: str = "Not specified"
    consensus_owner: str = "Unknown"
    citation: Optional[Citation] = None


@dataclass
class ExecutiveSummary:
    """Structured executive summary of the meeting."""
    title: str = "Meeting Notes"
    overview: str = ""
    key_highlights: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)


@dataclass
class MeetingIntelligence:
    """Comprehensive structured intelligence report for a meeting."""
    session_id: str
    summary: ExecutiveSummary
    action_items: List[ActionItem] = field(default_factory=list)
    decisions: List[MeetingDecision] = field(default_factory=list)
    raw_transcript: str = ""
    segments: List[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    language_mode: str = "English"
    audio_source: str = "Recording"


# --------------------------------------------------------------------------
# LLM Initialization & Retries
# --------------------------------------------------------------------------

def get_llm():
    model = os.getenv("MISTRAL_MODEL", "open-mistral-nemo").strip()
    return ChatMistralAI(
        model=model,
        mistral_api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
        temperature=0.2,
        max_retries=5,
    )


def invoke_with_retry(chain, inputs, max_retries: int = 5, initial_delay: float = 3.0):
    """
    Invoke a runnable chain with exponential backoff on HTTP 429 (Rate Limit Exceeded).
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit exceeded" in err_str or "rate_limited" in err_str:
                if attempt < max_retries - 1:
                    print(
                        f"[!] Mistral rate limit (429) hit. Pausing for {delay:.1f}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise


def build_chain(system_prompt: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------
# Citation Mapping & Verification Helpers
# --------------------------------------------------------------------------

def find_best_segment_citation(
    claim_text: str,
    segments: Optional[List[dict]],
    candidate_timestamp: Optional[str] = None,
    max_duration: Optional[float] = None,
) -> Optional[Citation]:
    """
    Validate and map an action item or decision to an actual finalized segment.
    Prefer mapping extracted claims to existing segments over accepting model-generated timestamps.
    Reject unsupported, fabricated, or out-of-range timestamps.
    """
    if not segments:
        return None

    clean_claim = claim_text.strip().lower()
    if not clean_claim:
        return None

    # Step 1: Candidate timestamp validation
    if candidate_timestamp:
        start_sec, end_sec = parse_timestamp_range(candidate_timestamp)
        # Check out-of-bounds relative to max duration
        if max_duration is not None and start_sec > (max_duration + 1.5):
            # Candidate is fabricated or beyond cutoff
            pass
        else:
            for s in segments:
                s_start = s.get("start", 0.0)
                s_end = s.get("end", 0.0)
                # Overlap check with small tolerance
                if not (end_sec < s_start - 2.0 or start_sec > s_end + 2.0):
                    seg_txt = s.get("text", "").strip()
                    claim_words = set(re.findall(r"\w{3,}", clean_claim))
                    seg_words = set(re.findall(r"\w{3,}", seg_txt.lower()))
                    # Either text overlap exists or timestamp matches closely (<= 1.5s)
                    if (claim_words & seg_words) or abs(s_start - start_sec) <= 1.5:
                        return Citation(
                            start=s_start,
                            end=s_end,
                            timestamp_str=f"{format_timestamp(s_start)} - {format_timestamp(s_end)}",
                            excerpt=seg_txt,
                        )

    # Step 2: Content-based segment mapping
    claim_words = set(re.findall(r"\w{3,}", clean_claim))
    if not claim_words:
        return None

    best_seg = None
    best_overlap = 0

    for s in segments:
        s_start = s.get("start", 0.0)
        if max_duration is not None and s_start > (max_duration + 1.5):
            continue
        seg_txt = s.get("text", "").strip()
        if not seg_txt:
            continue
        seg_words = set(re.findall(r"\w{3,}", seg_txt.lower()))
        overlap = len(claim_words & seg_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_seg = s

    min_required = 2 if len(claim_words) >= 3 else 1
    if best_seg is not None and best_overlap >= min_required:
        s_start = best_seg.get("start", 0.0)
        s_end = best_seg.get("end", 0.0)
        return Citation(
            start=s_start,
            end=s_end,
            timestamp_str=f"{format_timestamp(s_start)} - {format_timestamp(s_end)}",
            excerpt=best_seg.get("text", "").strip(),
        )

    return None


# --------------------------------------------------------------------------
# Structured Extraction Functions (Phase 7)
# --------------------------------------------------------------------------

STRUCTURED_ACTION_ITEMS_PROMPT = """You are an expert meeting analyst. Extract all actionable tasks from the transcript.
CRITICAL GROUNDING RULES:
1. Do NOT invent owners, deadlines, priority, or tasks.
2. If the owner is not explicitly stated in the transcript, set owner to "Unknown".
3. If the deadline is not explicitly stated, set deadline to "Not specified".
4. Set priority to "High", "Medium", or "Low" based only on explicit urgency words; default to "Medium".
5. Include the timestamp [MM:SS] if mentioned in the transcript.
6. If no action items are present, output an empty JSON list [].

Output strictly in JSON format as a list of objects:
[
  {
    "task": "Task description",
    "owner": "Owner name or Unknown",
    "priority": "High/Medium/Low",
    "deadline": "Deadline or Not specified",
    "timestamp": "[MM:SS]"
  }
]
"""

STRUCTURED_DECISIONS_PROMPT = """You are an expert meeting analyst. Extract all key decisions made during the meeting.
CRITICAL GROUNDING RULES:
1. Do NOT invent decisions, rationale, or consensus.
2. If context/rationale is not explicitly stated, set to "Not specified".
3. If consensus owner is not explicitly stated, set to "Unknown".
4. Include the timestamp [MM:SS] if mentioned in the transcript.
5. If no decisions were agreed on, output an empty JSON list [].

Output strictly in JSON format as a list of objects:
[
  {
    "decision": "Decision agreed upon",
    "context_rationale": "Rationale or Not specified",
    "consensus_owner": "Consensus owner or Unknown",
    "timestamp": "[MM:SS]"
  }
]
"""


def extract_structured_action_items(
    transcript: str,
    segments: Optional[List[dict]] = None,
    max_duration: Optional[float] = None,
    llm: Any = None,
) -> List[ActionItem]:
    """
    Extract structured ActionItem objects with verified citations against transcript segments.
    """
    if not transcript or not transcript.strip():
        return []

    if llm is None:
        llm = get_llm()

    messages = [
        SystemMessage(content=STRUCTURED_ACTION_ITEMS_PROMPT),
        HumanMessage(content=transcript),
    ]

    try:
        raw_response = llm.invoke(messages)
        raw_res = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    except Exception as e:
        print(f"[!] Structured action item extraction failed: {e}")
        return []

    # Parse JSON or fallback blocks
    items_data = []
    try:
        # Match JSON block
        json_match = re.search(r"\[\s*\{.*\}\s*\]", raw_res, re.DOTALL)
        if json_match:
            items_data = json.loads(json_match.group(0))
        elif raw_res.strip().startswith("[") and raw_res.strip().endswith("]"):
            items_data = json.loads(raw_res.strip())
    except Exception:
        items_data = []

    # Fallback heuristic parser if model output was not valid JSON
    if not items_data and "no action items" not in raw_res.lower():
        # Match lines like: 1. Task - Owner - Deadline
        lines = [line.strip() for line in raw_res.split("\n") if line.strip()]
        for line in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                task_txt = m.group(1).strip()
                items_data.append({
                    "task": task_txt,
                    "owner": "Unknown",
                    "priority": "Medium",
                    "deadline": "Not specified",
                    "timestamp": "",
                })

    structured_items = []
    for d in items_data:
        task = str(d.get("task", "")).strip()
        if not task:
            continue
        owner = str(d.get("owner", "Unknown")).strip()
        if not owner or owner.lower() in ("none", "null", "n/a", "tbd"):
            owner = "Unknown"

        priority = str(d.get("priority", "Medium")).strip().capitalize()
        if priority not in ("High", "Medium", "Low"):
            priority = "Medium"

        deadline = str(d.get("deadline", "Not specified")).strip()
        if not deadline or deadline.lower() in ("none", "null", "n/a", "tbd"):
            deadline = "Not specified"

        ts_candidate = str(d.get("timestamp", "")).strip()

        # Map and verify citation
        citation = find_best_segment_citation(
            claim_text=task,
            segments=segments,
            candidate_timestamp=ts_candidate,
            max_duration=max_duration,
        )

        structured_items.append(ActionItem(
            task=task,
            owner=owner,
            priority=priority,
            deadline=deadline,
            citation=citation,
            status="Open",
        ))

    return structured_items


def extract_structured_decisions(
    transcript: str,
    segments: Optional[List[dict]] = None,
    max_duration: Optional[float] = None,
    llm: Any = None,
) -> List[MeetingDecision]:
    """
    Extract structured MeetingDecision objects with verified citations against transcript segments.
    """
    if not transcript or not transcript.strip():
        return []

    if llm is None:
        llm = get_llm()

    messages = [
        SystemMessage(content=STRUCTURED_DECISIONS_PROMPT),
        HumanMessage(content=transcript),
    ]

    try:
        raw_response = llm.invoke(messages)
        raw_res = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    except Exception as e:
        print(f"[!] Structured decision extraction failed: {e}")
        return []

    dec_data = []
    try:
        json_match = re.search(r"\[\s*\{.*\}\s*\]", raw_res, re.DOTALL)
        if json_match:
            dec_data = json.loads(json_match.group(0))
        elif raw_res.strip().startswith("[") and raw_res.strip().endswith("]"):
            dec_data = json.loads(raw_res.strip())
    except Exception:
        dec_data = []

    # Fallback heuristic parser
    if not dec_data and "no key decisions" not in raw_res.lower() and "no decisions" not in raw_res.lower():
        lines = [line.strip() for line in raw_res.split("\n") if line.strip()]
        for line in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                dec_data.append({
                    "decision": m.group(1).strip(),
                    "context_rationale": "Not specified",
                    "consensus_owner": "Unknown",
                    "timestamp": "",
                })

    structured_decisions = []
    for d in dec_data:
        decision = str(d.get("decision", "")).strip()
        if not decision:
            continue
        rationale = str(d.get("context_rationale", "Not specified")).strip()
        if not rationale or rationale.lower() in ("none", "null", "n/a"):
            rationale = "Not specified"

        owner = str(d.get("consensus_owner", "Unknown")).strip()
        if not owner or owner.lower() in ("none", "null", "n/a"):
            owner = "Unknown"

        ts_candidate = str(d.get("timestamp", "")).strip()

        citation = find_best_segment_citation(
            claim_text=decision,
            segments=segments,
            candidate_timestamp=ts_candidate,
            max_duration=max_duration,
        )

        structured_decisions.append(MeetingDecision(
            decision=decision,
            context_rationale=rationale,
            consensus_owner=owner,
            citation=citation,
        ))

    return structured_decisions


def extract_meeting_intelligence(
    transcript: str,
    segments: Optional[List[dict]] = None,
    metadata: Optional[dict] = None,
    max_duration: Optional[float] = None,
    llm: Any = None,
) -> MeetingIntelligence:
    """
    Generate complete structured meeting intelligence from transcript and metadata.
    """
    from core.summarizer import generate_title, summarize

    meta = metadata or {}
    session_id = meta.get("session_id", "session_default")
    lang_mode = meta.get("language", "English")
    source = meta.get("audio_source", "Recording")
    dur_sec = meta.get("duration_seconds", 0.0)
    if max_duration is None and dur_sec > 0:
        max_duration = dur_sec

    # Run extractions
    title = generate_title(transcript) if transcript.strip() else "Untitled Meeting"
    summary_text = summarize(transcript) if transcript.strip() else "No transcript content available."
    action_items = extract_structured_action_items(transcript, segments=segments, max_duration=max_duration, llm=llm)
    decisions = extract_structured_decisions(transcript, segments=segments, max_duration=max_duration, llm=llm)
    questions_raw = extract_questions(transcript) if transcript.strip() else ""

    q_list = []
    if questions_raw and "no open questions" not in questions_raw.lower():
        for line in questions_raw.split("\n"):
            line = line.strip()
            if line:
                m = re.match(r"^\d+[\.\)]\s*(.+)", line)
                q_list.append(m.group(1).strip() if m else line)

    exec_summary = ExecutiveSummary(
        title=title,
        overview=summary_text,
        key_highlights=[],
        unresolved_questions=q_list,
    )

    return MeetingIntelligence(
        session_id=session_id,
        summary=exec_summary,
        action_items=action_items,
        decisions=decisions,
        raw_transcript=transcript,
        segments=segments or [],
        duration_seconds=dur_sec,
        language_mode=lang_mode,
        audio_source=source,
    )


# --------------------------------------------------------------------------
# Backward-Compatible Legacy Functions (Exact Signatures Preserved)
# --------------------------------------------------------------------------

def extract_action_items(transcript: str) -> str:
    """Extract action items formatted as a numbered markdown string."""
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


# Aliases for backward compatibility
extra_action_items = extract_action_items


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions formatted as a numbered markdown string."""
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extract open questions formatted as a numbered markdown string."""
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


# Backward compatibility alias
extract_unresolved_questions = extract_questions


def extract_all_insights(
    transcript: str,
    max_workers: int = None,
    progress_callback = None,
) -> dict:
    """
    Execute all independent LLM insight extractions concurrently with bounded concurrency.
    Preserves exact backward compatibility with Phase 2/3 parallel runner.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from core.summarizer import generate_title, summarize

    if max_workers is None:
        try:
            max_workers = int(os.getenv("LLM_MAX_WORKERS", "3"))
        except (ValueError, TypeError):
            max_workers = 3

    workers = max(1, min(max_workers, 5))

    task_definitions = [
        ("title", "Title Generation", lambda: generate_title(transcript)),
        ("summary", "Summary Generation", lambda: summarize(transcript)),
        ("action_items", "Action Items", lambda: extract_action_items(transcript)),
        ("key_decisions", "Key Decisions", lambda: extract_key_decisions(transcript)),
        ("open_questions", "Open Questions", lambda: extract_questions(transcript)),
    ]

    total_tasks = len(task_definitions)
    results = {}
    stage_timings = {}
    t_start = time.perf_counter()

    def _execute_task(field_key, display_name, fn):
        t0 = time.perf_counter()
        try:
            val = fn()
            dur = time.perf_counter() - t0
            return field_key, display_name, val, dur, None
        except Exception as exc:
            dur = time.perf_counter() - t0
            return field_key, display_name, f"Failed to generate {display_name}: {exc}", dur, exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_execute_task, field, name, func)
            for field, name, func in task_definitions
        ]

        completed_count = 0
        for future in as_completed(futures):
            field_key, display_name, val, dur, exc = future.result()
            results[field_key] = val
            stage_timings[display_name] = dur
            completed_count += 1

            if exc:
                print(f"[!] Warning: Task '{display_name}' encountered an error: {exc}")

            if progress_callback:
                try:
                    progress_callback(completed_count, total_tasks, display_name)
                except Exception:
                    pass

    total_elapsed = time.perf_counter() - t_start

    return {
        "title": results.get("title", ""),
        "summary": results.get("summary", ""),
        "action_items": results.get("action_items", ""),
        "key_decisions": results.get("key_decisions", ""),
        "open_questions": results.get("open_questions", ""),
        "stage_timings": stage_timings,
        "total_llm_time": total_elapsed,
    }