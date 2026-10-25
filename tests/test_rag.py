import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.vector_store import build_vector_store, get_retriever
from core.rag_engine import create_rag_runnable, ask_question, get_llm

FALLBACK_PHRASE = "I could not find this information in the transcript."

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the AI token economy?",
        "should_find": True,
        "expected_topics": ["token", "cost", "inference", "economy"],
    },
    {
        "id": 2,
        "question": "Why are AI companies changing their pricing strategies?",
        "should_find": True,
        "expected_topics": ["money", "cost", "lose", "billion", "inference"],
    },
    {
        "id": 3,
        "question": "What is Anthropic's painted door test?",
        "should_find": True,
        "expected_topics": ["painted door", "fake door", "Claude", "$20", "$100", "pricing"],
    },
    {
        "id": 4,
        "question": "How has GitHub Copilot changed its pricing model?",
        "should_find": True,
        "expected_topics": ["Copilot", "executes", "token", "model", "cost"],
    },
    {
        "id": 5,
        "question": "What financial challenges do OpenAI and Anthropic face?",
        "should_find": True,
        "expected_topics": ["billion", "investment", "burn", "lose", "costs", "inference"],
    },
    {
        "id": 6,
        "question": "Why does Google have an advantage in AI investment?",
        "should_find": True,
        "expected_topics": ["Google", "money", "billion", "profitable", "revenue"],
    },
    {
        "id": 7,
        "question": "What future pricing models are discussed in the transcript?",
        "should_find": True,
        "expected_topics": ["token", "free", "usage", "price", "charging"],
    },
    {
        "id": 8,
        "question": "What is the speaker's main concern about free AI tools?",
        "should_find": True,
        "expected_topics": ["free", "usage", "cracks", "going down", "viable"],
    },
    {
        "id": 9,
        "question": "What is the transcript's main conclusion?",
        "should_find": True,
        "expected_topics": ["AI", "viable", "cracks", "free", "future"],
    },
    {
        "id": 10,
        "question": "What is the capital of Japan?",
        "should_find": False,  # Should trigger the fallback!
        "expected_topics": [],
    },
]


def run_rag_tests():
    transcript_file = os.path.join(PROJECT_ROOT, "transcript.txt")
    if not os.path.exists(transcript_file):
        transcript_file = "transcript.txt"
    if not os.path.exists(transcript_file):
        print(f"[!] Error: {transcript_file} not found.")
        sys.exit(1)

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    print(f"[+] Loaded transcript ({len(transcript)} chars, {len(transcript.split())} words).")
    print("\n[+] Building isolated test vector store (collection='test_rag_eval')...")

    # Build isolated collection specifically for test
    test_collection = "test_rag_eval"
    vector_store = build_vector_store(
        transcript=transcript,
        collection_name=test_collection,
        chunk_size=1000,
        chunk_overlap=200,
        clear_existing=True,
    )

    retriever = get_retriever(vector_store, k=5)
    llm = get_llm()
    rag_chain = create_rag_runnable(retriever, llm)

    print("\n" + "=" * 70)
    print("STARTING RAG EVALUATION SUITE (10 QUESTIONS)")
    print("=" * 70)

    results = []

    for item in TEST_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        should_find = item["should_find"]

        print(f"\n--- [Q{qid}] {q} ---")

        # 1. Retrieve chunks
        docs = retriever.invoke(q)
        chunk_indices = [d.metadata.get("chunk_index", -1) for d in docs]
        print(f"Retrieved {len(docs)} chunk(s): {chunk_indices}")

        # 2. Invoke RAG
        time.sleep(1.0)  # Rate limit cooldown
        answer = rag_chain.invoke(q).strip()
        print(f"Answer: {answer[:250]}..." if len(answer) > 250 else f"Answer: {answer}")

        # 3. Verify correctness
        is_fallback = FALLBACK_PHRASE.lower() in answer.lower()

        if should_find:
            # Questions 1-9 should NOT trigger fallback
            passed = not is_fallback
            reason = "Answered from transcript" if passed else "Incorrect fallback triggered"
        else:
            # Question 10 MUST trigger fallback
            passed = is_fallback
            reason = "Correctly fell back (unsupported info)" if passed else "Hallucinated unsupported info"

        status_str = "PASS" if passed else "FAIL"
        print(f"Status: [{status_str}] - {reason}")

        results.append({
            "id": qid,
            "question": q,
            "status": status_str,
            "passed": passed,
            "reason": reason,
            "retrieved_chunks": chunk_indices,
            "answer_preview": answer[:120].replace("\n", " "),
        })

    # Summary table
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'ID':<3} | {'Status':<5} | {'Question':<45} | {'Chunks'}")
    print("-" * 70)

    all_passed = True
    for r in results:
        chunks_str = str(r["retrieved_chunks"])
        print(f"{r['id']:<3} | {r['status']:<5} | {r['question'][:45]:<45} | {chunks_str}")
        if not r["passed"]:
            all_passed = False

    print("=" * 70)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Total: {passed_count}/{len(results)} passed.")

    return all_passed


def test_rag_evaluation_suite():
    """Pytest test case executing the complete 10-question RAG verification suite."""
    assert run_rag_tests() is True


if __name__ == "__main__":
    success = run_rag_tests()
    sys.exit(0 if success else 1)
