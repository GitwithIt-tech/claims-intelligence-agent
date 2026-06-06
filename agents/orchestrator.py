"""
agents/orchestrator.py
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

llm = ChatGroq(
    api_key=llm_settings.GROQ_API_KEY,
    model=llm_settings.GROQ_MODEL,
    temperature=0.0,
    max_tokens=2048,
    timeout=60,
    max_retries=1,
)

class AgentState(TypedDict):
    question:     str
    route:        str
    claim_id:     str | None
    sql_result:   dict
    rag_result:   dict
    ml_result:    dict
    final_answer: str

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a router for an insurance claims AI system.
Classify the user question into ONE category:

sql      — statistics, counts, trends, comparisons across many claims
rag      — insurance policy rules, procedures, coverage, legal terms
ml       — scoring a SPECIFIC claim by its CLM ID
both_sql_rag — needs both data AND policy context together

Reply with ONLY one word: sql, rag, ml, or both_sql_rag"""),
    ("human", "{question}"),
])

def extract_claim_id(text: str) -> str | None:
    match = re.search(r'CLM\d+', text, re.IGNORECASE)
    return match.group(0).upper() if match else None

def router_node(state: AgentState) -> AgentState:
    try:
        chain    = ROUTER_PROMPT | llm
        response = chain.invoke({"question": state["question"]})
        route    = response.content.strip().lower()
        if route not in ["sql", "rag", "ml", "both_sql_rag"]:
            route = "sql"
    except Exception:
        route = "sql"
    claim_id = extract_claim_id(state["question"])
    return {**state, "route": route, "claim_id": claim_id}

def sql_node(state: AgentState) -> AgentState:
    try:
        from agents.sql_agent import query as sql_query
        result = sql_query(state["question"])
    except Exception as e:
        result = {"success": False, "summary": f"SQL error: {e}"}
    return {**state, "sql_result": result}

def rag_node(state: AgentState) -> AgentState:
    try:
        from agents.rag_agent import run_rag_agent
        result = run_rag_agent(state["question"])
    except Exception as e:
        result = {"success": False, "answer": f"RAG error: {e}", "sources": []}
    return {**state, "rag_result": result}

def ml_node(state: AgentState) -> AgentState:
    try:
        from agents.ml_agent import run_ml_agent
        result = run_ml_agent(claim_id=state.get("claim_id"))
    except Exception as e:
        result = {"success": False, "answer": f"ML error: {e}", "scores": {}}
    return {**state, "ml_result": result}

def both_sql_rag_node(state: AgentState) -> AgentState:
    state = sql_node(state)
    try:
        from agents.rag_agent import run_rag_agent
        rag_result = run_rag_agent("What are the litigation procedures and legal requirements?")
    except Exception as e:
        rag_result = {"success": False, "answer": f"RAG error: {e}", "sources": []}
    return {**state, "rag_result": rag_result}

SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior insurance claims analyst.
Synthesise results from multiple AI agents into one clear professional response.
Be specific, include numbers, cite sources. Max 200 words."""),
    ("human", "Question: {question}\n\nAgent results:\n{agent_results}\n\nProvide a professional combined answer:"),
])

def synthesiser_node(state: AgentState) -> AgentState:
    parts = []
    if state.get("sql_result", {}).get("success"):
        parts.append(f"DATABASE:\n{state['sql_result']['summary']}")
    if state.get("rag_result", {}).get("success"):
        sources = ", ".join(state["rag_result"].get("sources", []))
        parts.append(
            f"POLICY:\n{state['rag_result']['answer']}"
            + (f"\nSources: {sources}" if sources else "")
        )
    if state.get("ml_result", {}).get("success"):
        parts.append(f"ML SCORING:\n{state['ml_result']['answer']}")

    if not parts:
        final = "I was unable to find relevant information for your question."
    elif len(parts) == 1:
        final = parts[0]
    else:
        try:
            chain    = SYNTH_PROMPT | llm
            response = chain.invoke({
                "question":     state["question"],
                "agent_results": "\n\n---\n\n".join(parts),
            })
            final = response.content
        except Exception:
            final = "\n\n".join(parts)

    return {**state, "final_answer": final}

def route_condition(state: AgentState) -> Literal["sql", "rag", "ml", "both_sql_rag"]:
    return state["route"]

def ask(question: str) -> dict:
    """Build a fresh graph every call — never cache, prevents second-call hang."""

    # Build graph fresh inside this function
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
        {"sql":"sql","rag":"rag","ml":"ml","both_sql_rag":"both_sql_rag"}
    )
    graph.add_edge("sql",          "synthesiser")
    graph.add_edge("rag",          "synthesiser")
    graph.add_edge("ml",           "synthesiser")
    graph.add_edge("both_sql_rag", "synthesiser")
    graph.add_edge("synthesiser",  END)
    compiled = graph.compile()

    initial_state: AgentState = {
        "question":     question,
        "route":        "",
        "claim_id":     None,
        "sql_result":   {},
        "rag_result":   {},
        "ml_result":    {},
        "final_answer": "",
    }

    try:
        result = compiled.invoke(initial_state)
    except Exception as e:
        return {
            "question": question, "route": "sql",
            "answer":   f"Something went wrong: {str(e)}",
            "sql_result": {}, "rag_result": {}, "ml_result": {},
        }

    return {
        "question":   question,
        "route":      result["route"],
        "answer":     result["final_answer"],
        "sql_result": result.get("sql_result", {}),
        "rag_result": result.get("rag_result", {}),
        "ml_result":  result.get("ml_result", {}),
    }