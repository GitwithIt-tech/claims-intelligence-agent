"""
agents/orchestrator.py
───────────────────────
LangGraph Orchestrator — the central brain.
Routes questions to the right agent(s) and synthesises answers.

Agents available:
  - sql_agent:  analytics queries on claims database
  - rag_agent:  policy document questions
  - ml_agent:   fraud/litigation scoring for a specific claim
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import re
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config.setting import llm_settings
from agents.sql_agent import query as sql_query
from agents.rag_agent import run_rag_agent
from agents.ml_agent  import run_ml_agent

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=llm_settings.GROQ_API_KEY,
    model=llm_settings.GROQ_MODEL,
    temperature=0.0,
    max_tokens=2048,
)

# ── State definition ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:    str
    route:       str
    claim_id:    str | None
    sql_result:  dict
    rag_result:  dict
    ml_result:   dict
    final_answer: str


# ── Router ────────────────────────────────────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a router for an insurance claims AI system.
Classify the user's question into ONE of these categories:

- sql      : questions about statistics, counts, trends, comparisons across many claims
             (e.g. "how many claims", "which region", "average fraud score", "top adjusters")
- rag      : questions about insurance policy rules, procedures, coverage, legal terms
             (e.g. "what does the policy say", "what is the procedure for", "time limit for")
- ml       : questions about scoring or analysis of a SPECIFIC claim by ID
             (e.g. "score claim CLM123", "what is the fraud risk for CLM456")
- both_sql_rag : questions needing both data AND policy context
             (e.g. "which claims are high risk AND what does policy say about them")

Reply with ONLY one word: sql, rag, ml, or both_sql_rag
"""),
    ("human", "{question}"),
])


def extract_claim_id(text: str) -> str | None:
    """Extract CLM ID from question if present."""
    match = re.search(r'CLM\d+', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def router_node(state: AgentState) -> AgentState:
    """Decide which agent(s) to call."""
    chain    = ROUTER_PROMPT | llm
    response = chain.invoke({"question": state["question"]})
    route    = response.content.strip().lower()

    # Fallback
    if route not in ["sql", "rag", "ml", "both_sql_rag"]:
        route = "sql"

    claim_id = extract_claim_id(state["question"])

    return {**state, "route": route, "claim_id": claim_id}


def sql_node(state: AgentState) -> AgentState:
    """Run SQL analytics agent."""
    result = sql_query(state["question"])
    return {**state, "sql_result": result}


def rag_node(state: AgentState) -> AgentState:
    """Run RAG policy agent."""
    result = run_rag_agent(state["question"])
    return {**state, "rag_result": result}


def ml_node(state: AgentState) -> AgentState:
    """Run ML scoring agent."""
    claim_id = state.get("claim_id")
    result   = run_ml_agent(claim_id=claim_id)
    return {**state, "ml_result": result}


# ── Synthesiser ───────────────────────────────────────────────────────────────
SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior insurance claims analyst.
Synthesise the results from one or more AI agents into a clear, 
professional response for a claims manager.
Be specific, include numbers and cite sources where available.
Format your response clearly with sections if multiple agents contributed.
"""),
    ("human", """Original question: {question}

Agent results:
{agent_results}

Provide a comprehensive, professional answer:"""),
])


def synthesiser_node(state: AgentState) -> AgentState:
    """Combine results from all agents into final answer."""
    parts = []

    if state.get("sql_result", {}).get("success"):
        parts.append(f"DATABASE ANALYSIS:\n{state['sql_result']['summary']}")

    if state.get("rag_result", {}).get("success"):
        sources = ", ".join(state["rag_result"].get("sources", []))
        parts.append(
            f"POLICY DOCUMENTATION:\n{state['rag_result']['answer']}"
            + (f"\nSources: {sources}" if sources else "")
        )

    if state.get("ml_result", {}).get("success"):
        parts.append(f"ML RISK SCORING:\n{state['ml_result']['answer']}")

    if not parts:
        final = "I was unable to find relevant information to answer your question."
    elif len(parts) == 1:
        final = parts[0]
    else:
        agent_results = "\n\n---\n\n".join(parts)
        chain    = SYNTH_PROMPT | llm
        response = chain.invoke({
            "question":     state["question"],
            "agent_results": agent_results,
        })
        final = response.content

    return {**state, "final_answer": final}


# ── Route condition ───────────────────────────────────────────────────────────
def route_condition(state: AgentState) -> Literal["sql", "rag", "ml", "both_sql_rag"]:
    return state["route"]


def both_sql_rag_node(state: AgentState) -> AgentState:
    """Call SQL first, then RAG with a simpler query to avoid timeout."""
    state = sql_node(state)
    # Simplify the question for RAG to reduce tokens
    simplified = "What are the litigation procedures and legal requirements?"
    rag_result = run_rag_agent(simplified)
    return {**state, "rag_result": rag_result}


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router",       router_node)
    graph.add_node("sql",          sql_node)
    graph.add_node("rag",          rag_node)
    graph.add_node("ml",           ml_node)
    graph.add_node("both_sql_rag", both_sql_rag_node)
    graph.add_node("synthesiser",  synthesiser_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_condition,
        {
            "sql":          "sql",
            "rag":          "rag",
            "ml":           "ml",
            "both_sql_rag": "both_sql_rag",
        }
    )

    graph.add_edge("sql",          "synthesiser")
    graph.add_edge("rag",          "synthesiser")
    graph.add_edge("ml",           "synthesiser")
    graph.add_edge("both_sql_rag", "synthesiser")
    graph.add_edge("synthesiser",  END)

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────
_graph = None

def ask(question: str) -> dict:
    """
    Main entry point. Ask any question — the orchestrator
    routes it to the right agent(s) automatically.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: AgentState = {
        "question":    question,
        "route":       "",
        "claim_id":    None,
        "sql_result":  {},
        "rag_result":  {},
        "ml_result":   {},
        "final_answer": "",
    }

    result = _graph.invoke(initial_state)
    return {
        "question":    question,
        "route":       result["route"],
        "answer":      result["final_answer"],
        "sql_result":  result.get("sql_result", {}),
        "rag_result":  result.get("rag_result", {}),
        "ml_result":   result.get("ml_result", {}),
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "How many claims are open by region?",
        "What does the policy say about submitting a claim after an incident?",
        "Score claim CLM0000042",
        "Which regions have highest litigation rates and what does policy say about litigation procedures?",
    ]

    print("Testing LangGraph Orchestrator...\n")
    print("=" * 60)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = ask(question)
        print(f"Routed to: {result['route'].upper()}")
        print("-" * 40)
        print(result["answer"])
        print("=" * 60)