"""
agents/rag_agent.py
RAG with intelligent fallback:
- Definition/concept questions → Groq knowledge directly
- Policy questions → search real PDFs first
- If PDF confidence low → fall back to Groq knowledge
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config.setting import llm_settings
from rag.retriever import retrieve_with_context, retrieve

llm = ChatGroq(
    api_key=llm_settings.GROQ_API_KEY,
    model=llm_settings.GROQ_MODEL,
    temperature=0.0,
    max_tokens=1024,
    timeout=60,
    max_retries=1,
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert insurance claims analyst.
Answer using ONLY the policy documentation below. Always cite the source document and page number.

Answer the question as fully as the documentation allows, even if it covers the topic only partially. If the documents explain a concept but omit a specific detail (e.g. an exact figure), give what IS stated and note plainly which part is not specified.

Reply with exactly NOT_IN_DOCUMENTS only when the documentation is entirely unrelated to the question and contains nothing useful about the topic.

POLICY DOCUMENTATION:
{context}
"""),
    ("human", "{question}"),
])

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior insurance professional with deep expertise in:
- UK insurance terminology, definitions, and concepts
- FCA regulation, ABI guidelines, Financial Ombudsman rules
- Claims procedures: motor, property, liability, health, travel
- Insurance fraud detection and litigation procedures
- Underwriting, risk assessment, excess, subrogation, indemnity, and all standard policy terms

When answering definitions or concepts, always provide:
1. A clear plain-English definition
2. How it applies in a real claims context
3. A practical example where helpful

Be concise, accurate, and professional.
If the question is completely unrelated to insurance, politely explain you specialise in insurance and claims."""),
    ("human", "{question}"),
])

DEFINITION_KEYWORDS = [
    "what is", "what's", "define", "definition of", "meaning of",
    "explain", "what does", "tell me about", "describe",
    "what are", "how does", "difference between", " vs ", "versus",
    "what do you mean", "what does it mean", "can you explain",
    "clarify", "elaborate", "break down",
]


def _is_definition_question(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in DEFINITION_KEYWORDS)


def _rag_confidence(chunks: list) -> float:
    # Top-chunk score, not average: you answer from the single best chunk,
    # so averaging in weaker neighbours dilutes a real signal. Calibrated
    # against all-MiniLM-L6-v2's score distribution on this corpus (0.33-0.60).
    if not chunks:
        return 0.0
    return max(c["score"] for c in chunks)


def _groq_knowledge_answer(question: str) -> dict:
    """Answer from Groq's training knowledge — like ChatGPT, goes beyond documents."""
    try:
        chain    = FALLBACK_PROMPT | llm
        response = chain.invoke({"question": question})
        return {
            "success":     True,
            "agent":       "rag",
            "answer":      response.content.strip(),
            "sources":     [],
            "source_type": "ai_knowledge",
        }
    except Exception as e:
        return {
            "success":     False,
            "agent":       "rag",
            "answer":      f"Could not answer: {str(e)}",
            "sources":     [],
            "source_type": "error",
        }


def run_rag_agent(question: str) -> dict:
    """
    Smart RAG with fallback:
    1. Definition/concept question → try PDFs, fall back to Groq knowledge
    2. Policy/procedure question  → search PDFs, fall back if low confidence
    3. Always returns a useful answer — never leaves user empty-handed
    """

    if _is_definition_question(question):
        # Try RAG first with relaxed threshold
        try:
            chunks     = retrieve(question, top_k=3)
            confidence = _rag_confidence(chunks)

            if confidence > 0.47:
                context  = retrieve_with_context(question, top_k=3)
                chain    = RAG_PROMPT | llm
                response = chain.invoke({"context": context, "question": question})
                answer   = response.content.strip()

                if "NOT_IN_DOCUMENTS" not in answer:
                    sources = list({f"{c['source']} (p.{c['page']})" for c in chunks})
                    return {
                        "success":     True,
                        "agent":       "rag",
                        "answer":      answer,
                        "sources":     sources,
                        "source_type": "policy_documents",
                        "confidence":  confidence,
                        "contexts":    [c["text"] for c in chunks],
                    }
        except Exception:
            pass

        # Fall back to Groq knowledge for definitions
        return _groq_knowledge_answer(question)

    # Standard RAG for policy/procedure questions
    try:
        chunks     = retrieve(question, top_k=5)
        confidence = _rag_confidence(chunks)
        context    = retrieve_with_context(question, top_k=5)

        chain    = RAG_PROMPT | llm
        response = chain.invoke({"context": context, "question": question})
        answer   = response.content.strip()

        if "NOT_IN_DOCUMENTS" in answer or confidence < 0.40:
            return _groq_knowledge_answer(question)

        sources = list({f"{c['source']} (p.{c['page']})" for c in chunks})
        return {
                        "success":     True,
                        "agent":       "rag",
                        "answer":      answer,
                        "sources":     sources,
                        "source_type": "policy_documents",
                        "confidence":  confidence,
                        "contexts":    [c["text"] for c in chunks],
                    }

    except Exception:
        return _groq_knowledge_answer(question)


if __name__ == "__main__":
    tests = [
        "What is subrogation in insurance?",
        "What does excess mean on a policy?",
        "What does the AXA policy say about claim time limits?",
        "What is indemnity?",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        r = run_rag_agent(q)
        print(f"Source: {r['source_type']}")
        print(f"Answer: {r['answer'][:200]}")
        print("-" * 50)