import os
import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meetings_transcripts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_K = 5
SENTINEL_NO_MATCH_SESSION = "__NO_MATCH_MISSING_SESSION__"

_embeddings_instance = None


def get_embeddings():
    """Cache and return the HuggingFace embeddings model instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


def build_vector_store(
    transcript: str,
    session_id: str = None,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = CHROMA_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    clear_existing: bool = True,
    metadata_extra: dict = None,
    segments: list = None,
) -> Chroma:
    """
    Split the transcript into semantically rich chunks and index in ChromaDB.
    Enforces video/session isolation by tagging each chunk with `session_id`.
    When clear_existing=True and session_id is provided, purges stale chunks
    for this session without deleting other indexed sessions.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty. Cannot build vector store.")

    clean_session_id = str(session_id).strip() if session_id else None
    if not clean_session_id:
        clean_session_id = f"sess_{uuid.uuid4().hex[:8]}"

    print(f"[VectorStore] Splitting transcript into chunks (size={chunk_size}, overlap={chunk_overlap}, session={clean_session_id})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    chunks = splitter.split_text(transcript.strip())

    if not chunks:
        raise ValueError("No text chunks generated from transcript.")

    docs = []
    base_meta = metadata_extra or {}
    for i, chunk in enumerate(chunks):
        meta = {
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_char_count": len(chunk),
            "session_id": clean_session_id,
            **base_meta,
        }
        # If segments with start/end are provided, calculate timeline window for this chunk
        if segments:
            matching_starts = [
                s["start"] for s in segments
                if s.get("text", "").strip() and s.get("text", "").strip()[:20].lower() in chunk.lower()
            ]
            matching_ends = [
                s["end"] for s in segments
                if s.get("text", "").strip() and s.get("text", "").strip()[-20:].lower() in chunk.lower()
            ]
            if matching_starts and matching_ends:
                meta["start_time"] = min(matching_starts)
                meta["end_time"] = max(matching_ends)
                meta["timestamp_label"] = f"[{int(meta['start_time']//60):02d}:{int(meta['start_time']%60):02d} - {int(meta['end_time']//60):02d}:{int(meta['end_time']%60):02d}]"

        docs.append(Document(page_content=chunk, metadata=meta))

    embeddings = get_embeddings()

    # Isolate session or collection
    if clear_existing and os.path.exists(persist_directory):
        try:
            # Targeted purge: remove any stale chunks belonging to this session
            existing_vs = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            existing_vs.delete(where={"session_id": clean_session_id})
            print(f"[VectorStore] Cleared stale chunks for session '{clean_session_id}' in '{collection_name}'.")
        except Exception:
            # If collection does not exist yet or delete is not needed, proceed
            pass

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # Attach session_id to vector store instance for active context resolution
    try:
        object.__setattr__(vector_store, "session_id", clean_session_id)
    except Exception:
        pass

    print(f"[VectorStore] Successfully indexed '{collection_name}' with {len(docs)} chunks (session='{clean_session_id}').")
    return vector_store


def load_vector_store(
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = CHROMA_DIR,
) -> Chroma:
    """Load an existing Chroma vector store collection."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


get_vector_store = load_vector_store


def get_retriever(
    vector_store: Chroma,
    session_id: str = None,
    k: int = DEFAULT_K,
    raise_on_missing: bool = False,
):
    """
    Return a similarity search retriever configured with k chunks,
    strictly filtered by session_id to guarantee video/session isolation.

    If session_id is not explicitly provided, attempts to resolve it
    from the active processing context (vector_store.session_id).
    If still missing, empty, or invalid:
      - If raise_on_missing=True, raises ValueError.
      - Otherwise, applies a non-matching sentinel filter (session_id='__NO_MATCH_MISSING_SESSION__')
        guaranteeing a clean no-results response ([]), and NEVER an unfiltered search.
    """
    if session_id is None:
        # Parameter was omitted: attempt to resolve from active processing context
        context_id = getattr(vector_store, "session_id", None)
        clean_id = str(context_id).strip() if context_id else None
    else:
        # Caller provided an explicit session_id (which might be empty or invalid)
        clean_id = str(session_id).strip()

    if not clean_id:
        if raise_on_missing:
            raise ValueError(
                "A valid session_id is required for retrieval. "
                "Unfiltered retrieval across multiple video sessions is prohibited to prevent cross-video data leakage."
            )
        # Never silently fall back to an unfiltered search: apply sentinel filter for 0-result isolation
        clean_id = SENTINEL_NO_MATCH_SESSION

    search_kwargs = {
        "k": k,
        "filter": {"session_id": clean_id},
    }

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

