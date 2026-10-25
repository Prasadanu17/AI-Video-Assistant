import os
import sys
import uuid
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from core.vector_store import build_vector_store, get_retriever, get_vector_store
from core.rag_engine import build_rag_chain, ask_question, FALLBACK_PHRASE

TEST_COLLECTION = "test_session_isolation_suite"


def test_01_retrieval_correct_video():
    """Verify that retrieval strictly returns chunks from the specified video session."""
    session_a = f"sess_alpha_{uuid.uuid4().hex[:6]}"
    text_a = (
        "Project Apollo was an American spaceflight program by NASA that succeeded "
        "in landing the first humans on the Moon in 1969 with Commander Neil Armstrong "
        "and Lunar Module Pilot Buzz Aldrin."
    )

    vs = build_vector_store(
        transcript=text_a,
        collection_name=TEST_COLLECTION,
        session_id=session_a,
        clear_existing=True,
    )

    retriever_a = get_retriever(vs, session_id=session_a, k=3)
    docs = retriever_a.invoke("Who landed on the Moon in Project Apollo?")

    assert len(docs) > 0, "Expected at least one document retrieved"
    for doc in docs:
        assert doc.metadata.get("session_id") == session_a, (
            f"Chunk session_id '{doc.metadata.get('session_id')}' != expected '{session_a}'"
        )
        assert "Neil Armstrong" in doc.page_content


def test_02_overlapping_similar_terms_and_no_cross_retrieval():
    """
    Verify that when two different videos share identical keywords and topics,
    retrieval for session A never returns session B chunks, and vice versa.
    """
    session_a = f"video_pricing_cloud_{uuid.uuid4().hex[:6]}"
    session_b = f"video_pricing_onprem_{uuid.uuid4().hex[:6]}"

    # Video A: Cloud pricing model
    text_a = (
        "Our enterprise pricing strategy is based on a monthly cloud subscription. "
        "The tier costs 500 dollars per month, includes unlimited AWS cloud storage, "
        "and provides a dedicated Slack channel with 99.9% cloud uptime SLA."
    )

    # Video B: On-premise pricing model with identical keywords ("enterprise pricing strategy")
    text_b = (
        "Our enterprise pricing strategy is based on an annual on-premise license. "
        "The tier costs 25000 dollars per year, requires local hardware servers, "
        "and provides telephone pager-duty escalation with on-site engineer visits."
    )

    # Index video A
    vs = build_vector_store(
        transcript=text_a,
        collection_name=TEST_COLLECTION,
        session_id=session_a,
        clear_existing=True,
    )

    # Index video B into the SAME collection (clear_existing=True should only purge session B)
    build_vector_store(
        transcript=text_b,
        collection_name=TEST_COLLECTION,
        session_id=session_b,
        clear_existing=True,
    )

    # Retriever A
    retriever_a = get_retriever(vs, session_id=session_a, k=5)
    docs_a = retriever_a.invoke("What is our enterprise pricing strategy and cost?")

    assert len(docs_a) > 0, "Session A should return relevant chunks"
    for doc in docs_a:
        assert doc.metadata.get("session_id") == session_a
        assert "500 dollars" in doc.page_content
        assert "25000 dollars" not in doc.page_content
        assert "pager-duty" not in doc.page_content

    # Retriever B
    retriever_b = get_retriever(vs, session_id=session_b, k=5)
    docs_b = retriever_b.invoke("What is our enterprise pricing strategy and cost?")

    assert len(docs_b) > 0, "Session B should return relevant chunks"
    for doc in docs_b:
        assert doc.metadata.get("session_id") == session_b
        assert "25000 dollars" in doc.page_content
        assert "500 dollars" not in doc.page_content
        assert "Slack channel" not in doc.page_content


def test_03_empty_or_missing_session_id():
    """Verify system handles empty, None, or whitespace-only session IDs safely."""
    # 1. build_vector_store with None session_id generates a fallback session without error
    vs = build_vector_store(
        transcript="Test content with missing session identifier fallback handling.",
        collection_name=TEST_COLLECTION,
        session_id=None,
        clear_existing=False,
    )
    assert vs is not None

    # 2. get_retriever with None or empty session_id does not crash
    retriever_none = get_retriever(vs, session_id=None, k=2)
    docs_none = retriever_none.invoke("Test content")
    assert isinstance(docs_none, list)

    retriever_empty = get_retriever(vs, session_id="", k=2)
    docs_empty = retriever_empty.invoke("Test content")
    assert isinstance(docs_empty, list)


def test_04_repeated_processing_stale_index_handling():
    """
    Verify that re-processing the same session ID purges old chunks of that session
    without affecting chunks from other concurrent sessions.
    """
    session_target = f"sess_update_{uuid.uuid4().hex[:6]}"
    session_other = f"sess_preserve_{uuid.uuid4().hex[:6]}"

    # 1. Ingest initial version of target session
    vs = build_vector_store(
        transcript="Initial preliminary draft: Q3 revenue was reported as 5 million dollars.",
        collection_name=TEST_COLLECTION,
        session_id=session_target,
        clear_existing=True,
    )

    # 2. Ingest other session
    build_vector_store(
        transcript="Confidential personnel report: Dr. Aris Thorne was appointed Chief Technology Officer.",
        collection_name=TEST_COLLECTION,
        session_id=session_other,
        clear_existing=True,
    )

    # 3. Re-process target session with updated transcript
    build_vector_store(
        transcript="Final audited draft: Q3 revenue was officially corrected to 9 million dollars.",
        collection_name=TEST_COLLECTION,
        session_id=session_target,
        clear_existing=True,  # Purges only session_target
    )

    # 4. Query target session: should have 9 million, NOT 5 million
    retriever_target = get_retriever(vs, session_id=session_target, k=5)
    docs_target = retriever_target.invoke("What was the Q3 revenue?")
    combined_target_text = " ".join(d.page_content for d in docs_target)

    assert "9 million dollars" in combined_target_text
    assert "5 million dollars" not in combined_target_text

    # 5. Query other session: should be completely intact
    retriever_other = get_retriever(vs, session_id=session_other, k=5)
    docs_other = retriever_other.invoke("Who was appointed Chief Technology Officer?")
    combined_other_text = " ".join(d.page_content for d in docs_other)

    assert "Dr. Aris Thorne" in combined_other_text


def test_05_rag_chain_session_metadata_and_retriever():
    """Verify that build_rag_chain attaches the session_id and isolates retrieval."""
    session_id = f"sess_chain_{uuid.uuid4().hex[:6]}"
    transcript = (
        "Quantum computing utilizes qubits that can exist in superposition states, "
        "enabling exponential parallelism for specific factoring algorithms like Shor's."
    )

    chain = build_rag_chain(
        transcript=transcript,
        session_id=session_id,
        collection_name=TEST_COLLECTION,
    )

    assert hasattr(chain, "session_id")
    assert chain.session_id == session_id
    assert hasattr(chain, "retriever")

    # In-domain question
    answer_in = chain.invoke("What does quantum computing utilize?").strip()
    assert "qubit" in answer_in.lower() or "superposition" in answer_in.lower()


def test_06_rag_fallback_preservation():
    """Verify that the strict fallback phrase is preserved when answers are missing."""
    session_id = f"sess_fallback_{uuid.uuid4().hex[:6]}"
    transcript = "The Mediterranean diet is rich in olive oil, fresh fruits, vegetables, and whole grains."

    chain = build_rag_chain(
        transcript=transcript,
        session_id=session_id,
        collection_name=TEST_COLLECTION,
    )

    # Query about completely unrelated knowledge
    out_of_domain_query = "What is the speed of light in vacuum meters per second?"
    answer = chain.invoke(out_of_domain_query).strip()

    assert FALLBACK_PHRASE.lower() in answer.lower(), (
        f"Expected fallback phrase '{FALLBACK_PHRASE}', but got: {answer}"
    )
