"""
rag_pipeline.py
---------------
Medical Knowledge Retrieval using RAG (Retrieval-Augmented Generation).

What this file does:
  1. Loads .txt medical documents from the /medical_docs folder
  2. Converts them into vector embeddings using a local sentence-transformer model
  3. Stores those embeddings in ChromaDB (a local vector database)
  4. When a user asks a question, it finds the most relevant documents
     and returns them so the response generator can build an answer.

Models used  : sentence-transformers/all-MiniLM-L6-v2  (runs 100% locally)
Vector store : ChromaDB                                  (stored on disk, no server needed)

No paid APIs. No internet needed after first model download.
"""

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ──────────────────────────────────────────────────────────────
# 1.  Configuration — edit these paths if needed
# ──────────────────────────────────────────────────────────────

# Folder that contains your .txt medical documents
MEDICAL_DOCS_FOLDER = Path(__file__).parent / "medical_docs"

# Folder where ChromaDB will save its data on disk
CHROMA_DB_FOLDER = Path(__file__).parent / "chroma_db"

# Name of the collection inside ChromaDB
COLLECTION_NAME = "medical_knowledge"

# HuggingFace embedding model (downloaded once, cached locally)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ──────────────────────────────────────────────────────────────
# 2.  Load the embedding model once (shared across all calls)
# ──────────────────────────────────────────────────────────────

print("[rag_pipeline] Loading embedding model …")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("[rag_pipeline] Embedding model loaded successfully.")


# ──────────────────────────────────────────────────────────────
# 3.  Initialise ChromaDB (persistent — data survives restarts)
# ──────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER))

# Get-or-create the collection (safe to call multiple times)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# ──────────────────────────────────────────────────────────────
# 4.  Public functions
# ──────────────────────────────────────────────────────────────

def load_documents(folder_path: str | Path = MEDICAL_DOCS_FOLDER) -> list[dict]:
    """
    Read every .txt file inside *folder_path* and return a list of documents.

    Each document is a dict with:
        {
            "id"      : filename without extension  e.g. "paracetamol"
            "content" : full text content of the file
            "source"  : original filename           e.g. "paracetamol.txt"
        }

    Example
    -------
    >>> docs = load_documents()
    >>> print(docs[0]["id"])
    'paracetamol'
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"[rag_pipeline] medical_docs folder not found at: {folder_path}\n"
            "Please create the folder and add your .txt files."
        )

    documents = []

    for txt_file in sorted(folder_path.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()

        if not content:
            print(f"[rag_pipeline] WARNING: {txt_file.name} is empty — skipping.")
            continue

        documents.append({
            "id":      txt_file.stem,          # e.g. "paracetamol"
            "content": content,
            "source":  txt_file.name,          # e.g. "paracetamol.txt"
        })
        print(f"[rag_pipeline] Loaded: {txt_file.name}  ({len(content)} chars)")

    print(f"[rag_pipeline] Total documents loaded: {len(documents)}")
    return documents


def build_vector_store(documents: list[dict]) -> None:
    """
    Embed each document and store it in ChromaDB.

    This function is IDEMPOTENT — running it multiple times will update
    existing entries rather than create duplicates (uses upsert).

    Parameters
    ----------
    documents : list[dict]
        Output from load_documents().

    Example
    -------
    >>> docs = load_documents()
    >>> build_vector_store(docs)
    """
    if not documents:
        print("[rag_pipeline] No documents to embed — vector store not updated.")
        return

    ids        = [doc["id"]      for doc in documents]
    contents   = [doc["content"] for doc in documents]
    metadatas  = [{"source": doc["source"]} for doc in documents]

    print(f"[rag_pipeline] Embedding {len(documents)} document(s) …")

    # Generate embeddings locally using sentence-transformers
    embeddings = embedding_model.encode(contents, show_progress_bar=True).tolist()

    # Upsert into ChromaDB (insert or update)
    collection.upsert(
        ids        = ids,
        documents  = contents,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    print(f"[rag_pipeline] Vector store updated. "
          f"Total documents in DB: {collection.count()}")


def retrieve_relevant_docs(question: str, k: int = 3) -> list[dict]:
    """
    Find the *k* most relevant documents for a given *question*.

    Uses cosine similarity between the question embedding and stored embeddings.

    Parameters
    ----------
    question : str
        The user's medical question.
    k : int
        Number of top documents to return (default 3).

    Returns
    -------
    list[dict]
        Each dict contains:
            {
                "id"      : document id   e.g. "paracetamol"
                "content" : document text
                "source"  : filename      e.g. "paracetamol.txt"
                "score"   : similarity distance (lower = more similar)
            }

    Example
    -------
    >>> results = retrieve_relevant_docs("What is the dosage for paracetamol?")
    >>> print(results[0]["id"])
    'paracetamol'
    """
    if collection.count() == 0:
        print("[rag_pipeline] WARNING: Vector store is empty. "
              "Run build_vectors.py first.")
        return []

    # Embed the question
    question_embedding = embedding_model.encode(question).tolist()

    # Query ChromaDB for top-k similar documents
    results = collection.query(
        query_embeddings = [question_embedding],
        n_results        = min(k, collection.count()),  # can't exceed total docs
        include          = ["documents", "metadatas", "distances"],
    )

    # Unpack and format results
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id":      results["ids"][0][i],
            "content": results["documents"][0][i],
            "source":  results["metadatas"][0][i]["source"],
            "score":   round(results["distances"][0][i], 4),
        })
        print(f"[rag_pipeline] Retrieved: {results['ids'][0][i]}  "
              f"(distance: {results['distances'][0][i]:.4f})")

    return retrieved


# ──────────────────────────────────────────────────────────────
# 5.  Quick self-test (run: python rag_pipeline.py)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  RAG Pipeline — Self Test")
    print("=" * 55)

    # Step 1: Load documents
    docs = load_documents()

    # Step 2: Build / update vector store
    build_vector_store(docs)

    # Step 3: Test retrieval with sample questions
    test_questions = [
        "What is the dosage for paracetamol?",
        "What are the side effects of amoxicillin?",
        "How do I manage high blood pressure?",
        "What should a diabetic patient avoid?",
    ]

    print("\n--- Retrieval Test ---")
    for question in test_questions:
        print(f"\nQuestion : {question}")
        results = retrieve_relevant_docs(question, k=2)
        for r in results:
            print(f"  → [{r['id']}]  score={r['score']}  "
                  f"preview: {r['content'][:80]}…")

    print("\n" + "=" * 55)