"""
build_vectors.py  (UPDATED — supports MedQuAD large dataset)
----------------
Builds the ChromaDB vector store from ALL medical documents
including MedQuAD subfolder documents.

Run this script:
  - Once after initial setup
  - After adding new documents to medical_docs/
  - After running parse_medquad.py

Usage:
  python build_vectors.py
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

MEDICAL_DOCS_FOLDER = Path(__file__).parent / "medical_docs"
CHROMA_DB_FOLDER    = Path(__file__).parent / "chroma_db"
COLLECTION_NAME     = "medical_knowledge"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
BATCH_SIZE          = 100   # process 100 docs at a time to avoid memory issues


# ──────────────────────────────────────────────────────────────
# Load model and ChromaDB
# ──────────────────────────────────────────────────────────────

print("[build] Loading embedding model…")
model = SentenceTransformer(EMBEDDING_MODEL)
print("[build] Model loaded.\n")

client     = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER))
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# ──────────────────────────────────────────────────────────────
# Load ALL .txt files recursively (including subfolders)
# ──────────────────────────────────────────────────────────────

def load_all_documents(folder: Path) -> list[dict]:
    """
    Recursively finds every .txt file inside folder AND all subfolders.
    This includes both the base docs and the medquad/ subfolder docs.
    """
    docs = []
    skipped = 0

    # rglob("*.txt") searches ALL subfolders recursively
    all_txt_files = sorted(folder.rglob("*.txt"))
    print(f"[build] Found {len(all_txt_files):,} total .txt files across all folders")

    for txt_file in all_txt_files:
        try:
            content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            skipped += 1
            continue

        if not content or len(content) < 50:
            skipped += 1
            continue

        rel_path = txt_file.relative_to(folder)
        doc_id   = str(rel_path.with_suffix(""))
        doc_id   = doc_id.replace("\\", "/").replace(" ", "_")
        doc_id   = doc_id[:200]

        docs.append({
            "id":      doc_id,
            "content": content[:8000],
            "source":  txt_file.name,
        })

    print(f"[build] Documents loaded  : {len(docs):,}")
    print(f"[build] Documents skipped : {skipped:,}\n")
    return docs


# ──────────────────────────────────────────────────────────────
# Embed and store in batches
# ──────────────────────────────────────────────────────────────

def build_in_batches(documents: list[dict]) -> None:
    """
    Embeds and upserts documents into ChromaDB in batches.
    Batching prevents RAM issues with thousands of documents.
    """
    total   = len(documents)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"[build] Embedding {total:,} documents in {batches} batches")
    print(f"[build] This may take 15-30 minutes for large datasets...\n")

    for i in range(0, total, BATCH_SIZE):
        batch     = documents[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        ids       = [doc["id"]      for doc in batch]
        contents  = [doc["content"] for doc in batch]
        metadatas = [{"source": doc["source"]} for doc in batch]

        embeddings = model.encode(
            contents,
            show_progress_bar = False,
            batch_size        = 32,
        ).tolist()

        collection.upsert(
            ids        = ids,
            documents  = contents,
            embeddings = embeddings,
            metadatas  = metadatas,
        )

        pct    = (batch_num / batches) * 100
        filled = int(pct / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        print(
            f"\r  [{bar}] {pct:5.1f}%  "
            f"batch {batch_num}/{batches}  "
            f"({min(i + BATCH_SIZE, total):,}/{total:,} docs)",
            end="", flush=True
        )

    print(f"\n\n[build] Done! {total:,} documents embedded.")
    print(f"[build] Total documents in ChromaDB: {collection.count():,}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Building Medical Knowledge Vector Store")
    print("=" * 55 + "\n")

    if not MEDICAL_DOCS_FOLDER.exists():
        print(f"ERROR: medical_docs folder not found at: {MEDICAL_DOCS_FOLDER}")
        exit(1)

    base_docs    = list(MEDICAL_DOCS_FOLDER.glob("*.txt"))
    medquad_path = MEDICAL_DOCS_FOLDER / "medquad"
    medquad_docs = list(medquad_path.rglob("*.txt")) if medquad_path.exists() else []

    print(f"[build] Base medical docs : {len(base_docs)}")
    print(f"[build] MedQuAD docs      : {len(medquad_docs):,}")
    print(f"[build] MedQuAD folder    : {'found' if medquad_path.exists() else 'NOT FOUND - run parse_medquad.py first'}")
    print()

    all_docs = load_all_documents(MEDICAL_DOCS_FOLDER)

    if not all_docs:
        print("ERROR: No documents found.")
        exit(1)

    build_in_batches(all_docs)

    print("\n" + "=" * 55)
    print("  Vector store is ready!")
    print("  Run: python main.py  to start the server")
    print("=" * 55 + "\n")