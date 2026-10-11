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
    time.sleep(1.0)
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
    time.sleep(1.0)
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})


def extract_questions(transcript: str) -> str:
    time.sleep(1.0)
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return invoke_with_retry(chain, {"text": transcript})