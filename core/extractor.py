# Actionable items, decisions, questions

import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


# Alias for backward compatibility
extra_action_items = extract_action_items


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


def extract_all_insights(
    transcript: str,
    max_workers: int = None,
    progress_callback = None,
) -> dict:
    """
    Execute all independent LLM insight extractions concurrently with bounded concurrency.
    
    Tasks executed:
      - Title Generation
      - Summary Generation
      - Action Items Extraction
      - Key Decisions Extraction
      - Open Questions Extraction

    Args:
        transcript: Full text of the meeting/video transcript.
        max_workers: Maximum concurrent LLM requests (defaults to LLM_MAX_WORKERS env or 3).
        progress_callback: Optional callable(completed_count, total_count, task_name) for progress tracking.

    Returns:
        dict with keys: title, summary, action_items, key_decisions, open_questions, stage_timings.
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