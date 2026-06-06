"""
rag/rag_chain.py
─────────────────
Full RAG chain — retrieves policy context and 
generates cited answers using Groq LLM.
Called by the RAG Agent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config.setting import llm_settings
from rag.retriever import retrieve_with_context, retrieve

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=llm_settings.GROQ_API_KEY,
    model=llm_settings.GROQ_MODEL,
    temperature=llm_settings.TEMPERATURE,
    max_tokens=llm_settings.MAX_TOKENS,
)

# ── Prompt ─────────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert insurance claims analyst with deep knowledge 
of insurance policy terms, claims procedures, fraud detection, and litigation guidelines.

Answer the question using ONLY the policy documentation provided below.
Always cite your sources by mentioning the document name and page number.
If the documentation does not contain enough information to answer, say so clearly.
Be precise and professional — your answers are used by claims handlers and legal teams.

POLICY DOCUMENTATION:
{context}
"""),
    ("human", "{question}"),
])


def ask(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from vector store
    2. Build context with citations
    3. Generate answer with Groq LLM
    
    Returns dict with answer, sources, chunks used.
    """
    # Retrieve
    chunks  = retrieve(question, top_k=top_k)
    context = retrieve_with_context(question, top_k=top_k)

    # Generate
    chain    = RAG_PROMPT | llm
    response = chain.invoke({
        "context":  context,
        "question": question,
    })

    # Extract unique sources
    sources = list({
        f"{c['source']} (p.{c['page']})" for c in chunks
    })

    return {
        "answer":  response.content,
        "sources": sources,
        "chunks":  len(chunks),
    }


if __name__ == "__main__":
    test_questions = [
        "What fraud score threshold triggers a referral to the Special Investigations Unit?",
        "What is the time limit for submitting a vehicle insurance claim after an incident?",
        "How are high value property claims handled differently from standard claims?",
    ]

    print("Testing full RAG chain with Groq LLM...\n")
    for q in test_questions:
        print(f"Question: {q}")
        print("-" * 60)
        result = ask(q)
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"Chunks used: {result['chunks']}")
        print("=" * 60 + "\n")