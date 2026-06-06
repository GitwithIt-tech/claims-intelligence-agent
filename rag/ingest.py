"""
rag/ingest.py
──────────────
Ingests real insurance PDF documents into ChromaDB vector store.
Chunks text, creates embeddings, persists to disk.

Run:
    python3 rag/ingest.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from config.setting import rag_settings

DOCS_DIR = ROOT / "data/raw"


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from PDF page by page.
    Returns list of dicts with text, source, page number.
    """
    doc    = fitz.open(str(pdf_path))
    pages  = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if len(text) < 50:
            continue
        pages.append({
            "text":   text,
            "source": pdf_path.name,
            "page":   page_num + 1,
        })
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def ingest_documents():
    print("Initialising ChromaDB...")
    Path(rag_settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=rag_settings.CHROMA_PERSIST_DIR)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=rag_settings.EMBEDDING_MODEL
    )

    # Fresh ingest — delete if exists
    try:
        client.delete_collection(rag_settings.COLLECTION_NAME)
        print("  Cleared existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=rag_settings.COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {DOCS_DIR}")
        return

    print(f"  Found {len(pdf_files)} PDF documents\n")

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        try:
            pages = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"    ✗ Failed to read: {e}")
            continue

        all_chunks = []
        all_ids    = []
        all_metas  = []

        for page in pages:
            sub_chunks = chunk_text(
                page["text"],
                chunk_size=rag_settings.CHUNK_SIZE,
                overlap=rag_settings.CHUNK_OVERLAP,
            )
            for i, chunk in enumerate(sub_chunks):
                chunk_id = f"{pdf_path.stem}_p{page['page']}_c{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metas.append({
                    "source": page["source"],
                    "page":   page["page"],
                    "chunk":  i,
                })

        # Upsert in batches of 50
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            collection.upsert(
                documents=all_chunks[i:i+batch_size],
                ids=all_ids[i:i+batch_size],
                metadatas=all_metas[i:i+batch_size],
            )

        print(f"    ✓ {len(all_chunks)} chunks ingested from {len(pages)} pages")
        total_chunks += len(all_chunks)

    print(f"\n✓ Total chunks in vector store: {total_chunks}")
    print(f"  Persisted to: {rag_settings.CHROMA_PERSIST_DIR}")
    print(f"  Collection: {rag_settings.COLLECTION_NAME}")


if __name__ == "__main__":
    ingest_documents()