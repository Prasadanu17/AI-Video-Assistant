import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    model = os.getenv("MISTRAL_MODEL", "open-mistral-nemo").strip()
    return ChatMistralAI(
        model=model,
        mistral_api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
        temperature=0.3,
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


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."],
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """
    Generate a summary of the meeting/video transcript.
    If the transcript fits within the context window (<40,000 chars),
    it summarizes directly in 1 API call to avoid rate limit issues.
    For longer transcripts, it uses chunking with rate-limit pacing.
    """
    llm = get_llm()

    # Direct summary for texts under 40k chars (~10k tokens)
    if len(transcript) < 40000:
        direct_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert meeting summarizer. Provide a concise, professional, "
                "well-structured summary in bullet points highlighting the main topics and conclusions.",
            ),
            ("human", "Summarize this transcript:\n\n{text}"),
        ])
        direct_chain = direct_prompt | llm | StrOutputParser()
        return invoke_with_retry(direct_chain, {"text": transcript})

    # For very long transcripts, map-reduce with pauses between chunks
    chunks = split_transcript(transcript)
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely:"),
        ("human", "Summarize this transcript: {text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(2)  # Cooldown between requests
        chunk_summaries.append(invoke_with_retry(map_chain, {"text": chunk}))

    combined = "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])
    combined_chain = combined_prompt | llm | StrOutputParser()
    time.sleep(2)
    return invoke_with_retry(combined_chain, {"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a short professional meeting title "
            "(max 8 words). Only return the title, nothing else.",
        ),
        ("human", "{text}"),
    ])

    title_chain = title_prompt | llm | StrOutputParser()
    return invoke_with_retry(title_chain, {"text": transcript[:2000]})
