import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
    DEFAULT_K,
    COLLECTION_NAME,
)

RAG_SYSTEM_PROMPT = """You are an intelligent video transcript assistant. Your task is to answer the user's question accurately using ONLY the provided transcript context below.

Guidelines:
1. SEMANTIC MATCHING: Understand paraphrasing, synonyms, and context. The transcript may explain the concept using colloquial or conversational language (for example, "painted door test" / "fake door test", "token economy", "inference costs", or GitHub Copilot moving from executes to token usage). Connect concepts based on their meaning in the discussion rather than requiring word-for-word matches.
2. TRUTHFUL & SUPPORTED: Rely strictly on what is stated or directly implied in the transcript context. Do not invent facts, numbers, company announcements, or outside details not found in the context.
3. PARTIAL ANSWERS: If the context answers part of the question but not all of it, answer the supported part clearly and explicitly state what is not mentioned in the transcript.
4. DISTINGUISH OPINION VS FACT: If a speaker expresses an opinion, speculation, or prediction (e.g. "I think...", "cracks are starting to show", "they will find a way"), represent it as the speaker's viewpoint, not an established industry fact.
5. STRICT FALLBACK: If and ONLY IF the transcript context contains no relevant information at all to answer the question (for example, completely unrelated general knowledge topics like geography or trivia not mentioned in the video), reply with EXACTLY:
"I could not find this information in the transcript."
Do NOT use this fallback if there is relevant discussion or context that addresses the question.
6. CONCISE & DIRECT: Be direct, structured, and concise in your response.

Transcript context:
{context}"""


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is missing. "
            "Please add it to your .env file or environment variables."
        )

    model = os.getenv("MISTRAL_MODEL", "open-mistral-nemo").strip()

    return ChatMistralAI(
        model=model,
        mistral_api_key=api_key.strip(),
        temperature=0.2,
        max_retries=5,
    )


def invoke_with_retry(chain, inputs, max_retries: int = 5, initial_delay: float = 2.0):
    """
    Invoke a chain with exponential backoff on HTTP 429 (Rate Limit Exceeded).
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


def format_docs(docs: list) -> str:
    """Format retrieved documents with chunk index labels for clear LLM grounding."""
    if not docs:
        return "No transcript context available."

    formatted = []
    for i, doc in enumerate(docs):
        chunk_idx = doc.metadata.get("chunk_index", i)
        formatted.append(
            f"[Transcript Excerpt {i + 1} (Chunk #{chunk_idx})]:\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


def create_rag_runnable(retriever, llm=None):
    """Assemble the LCEL RAG chain from retriever and LLM."""
    if llm is None:
        llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_rag_chain(
    transcript: str,
    k: int = DEFAULT_K,
    collection_name: str = COLLECTION_NAME,
    verbose: bool = False,
):
    """
    Build a fresh vector store from the transcript and return an LCEL RAG chain.
    Ensures clean collection isolation so old transcripts are never mixed.
    """
    vector_store = build_vector_store(
        transcript=transcript,
        collection_name=collection_name,
        clear_existing=True,
    )
    retriever = get_retriever(vector_store, k=k)
    llm = get_llm()

    return create_rag_runnable(retriever, llm)


def load_rag_chain(
    k: int = DEFAULT_K,
    collection_name: str = COLLECTION_NAME,
    verbose: bool = False,
):
    """Load an existing Chroma collection and return an LCEL RAG chain."""
    vector_store = load_vector_store(collection_name=collection_name)
    retriever = get_retriever(vector_store, k=k)
    llm = get_llm()

    return create_rag_runnable(retriever, llm)


def get_retrieved_chunks(retriever, question: str) -> list:
    """Diagnostic helper to inspect chunks retrieved for a given query."""
    return retriever.invoke(question)


def ask_question(rag_chain, question: str, retriever=None, verbose: bool = False) -> str:
    """
    Ask a question against the RAG chain with optional diagnostic logging.
    """
    print(f"\nQuestion: {question}")

    # Diagnostic logging if verbose is requested or DEBUG_RAG environment variable is set
    debug_mode = verbose or os.getenv("DEBUG_RAG", "0") in ("1", "true", "True")
    if debug_mode and retriever is not None:
        docs = retriever.invoke(question)
        print(f"[Debug] Retrieved {len(docs)} chunk(s):")
        for i, d in enumerate(docs):
            idx = d.metadata.get("chunk_index", "?")
            preview = d.page_content.replace("\n", " ")[:120]
            print(f"  - Chunk #{idx}: {preview}...")

    answer = invoke_with_retry(rag_chain, question)
    print(f"\nAnswer:\n{answer}\n")
    return answer

