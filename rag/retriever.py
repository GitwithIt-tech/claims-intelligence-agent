"""
rag/retriever.py
─────────────────
Queries ChromaDB and returns relevant policy chunks with citations.
Called by the RAG Agent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
from chromadb.utils import embedding_functions
from config.setting import rag_settings


def get_collection():
    client = chromadb.PersistentClient(path=rag_settings.CHROMA_PERSIST_DIR)
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=rag_settings.EMBEDDING_MODEL
    )
    return client.get_collection(
        name=rag_settings.COLLECTION_NAME,
        embedding_function=ef,
    )


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    Returns list of dicts: text, source, page, score.
    """
    if top_k is None:
        top_k = rag_settings.TOP_K_RETRIEVAL

    collection = get_collection()
    results    = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", 0),
            "score":  round(1 - dist, 4),
        })

    return chunks


def retrieve_with_context(query: str, top_k: int = None) -> str:
    """
    Returns formatted context string with citations.
    Ready to inject directly into an LLM prompt.
    """
    chunks = retrieve(query, top_k)

    if not chunks:
        return "No relevant policy documentation found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']} "
            f"(relevance: {chunk['score']:.2f})]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    test_queries = [
        "What is the fraud score threshold for referral to special investigations?",
        "How long does the company have to settle a vehicle claim?",
        "What happens when a claim goes to litigation?",
        "What are the requirements for high value property claims?",
    ]

    print("Testing RAG retrieval on real insurance documents...\n")
    for query in test_queries:
        print(f"Query: {query}")
        chunks = retrieve(query, top_k=2)
        if chunks:
            for c in chunks:
                print(f"  [{c['source']} p.{c['page']}] score={c['score']}")
                print(f"  {c['text'][:150]}...")
        else:
            print("  No results found")
        print()