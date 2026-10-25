import os
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
    print(f"[VectorStore] Splitting transcript into chunks (size={chunk_size}, overlap={chunk_overlap}, session={clean_session_id or 'none'})...")

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
            **base_meta,
        }
        if clean_session_id:
            meta["session_id"] = clean_session_id
        docs.append(Document(page_content=chunk, metadata=meta))

    embeddings = get_embeddings()

    # Isolate session or collection
    if clear_existing and os.path.exists(persist_directory):
        if clean_session_id:
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
        else:
            try:
                client = chromadb.PersistentClient(path=persist_directory)
                existing_colls = [c.name for c in client.list_collections()]
                if collection_name in existing_colls:
                    client.delete_collection(name=collection_name)
                    print(f"[VectorStore] Cleared stale collection '{collection_name}' for clean isolation.")
            except Exception as e:
                print(f"[VectorStore] Notice during collection reset: {e}")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    print(f"[VectorStore] Successfully indexed '{collection_name}' with {len(docs)} chunks (session='{clean_session_id or 'none'}').")
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
):
    """
    Return a similarity search retriever configured with k chunks,
    strictly filtered by session_id when provided.
    """
    search_kwargs = {"k": k}
    if session_id:
        clean_id = str(session_id).strip()
        if clean_id:
            search_kwargs["filter"] = {"session_id": clean_id}

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

