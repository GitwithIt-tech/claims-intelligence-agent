"""
agents/sql_agent.py
────────────────────
SQL Analytics Agent — converts natural language questions
into SQL queries and runs them against the claims database.

Uses LangChain + Groq LLM for text-to-SQL generation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config.setting import llm_settings, db_settings

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=llm_settings.GROQ_API_KEY,
    model=llm_settings.GROQ_MODEL,
    temperature=0.0,
    max_tokens=1024,
)

# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine(db_settings.url)

# ── Schema context fed to LLM ─────────────────────────────────────────────────
DB_SCHEMA = """
You have access to a PostgreSQL database with these tables:

TABLE: claims
  claim_id          VARCHAR  -- unique ID e.g. CLM0000001
  policy_number     VARCHAR  -- linked policy
  claim_type        VARCHAR  -- 'vehicle', 'property', 'liability', 'health', 'travel'
  claim_date        DATE     -- date claim was submitted
  incident_date     DATE     -- date incident occurred
  claim_amount      NUMERIC  -- claimed amount in GBP
  settled_amount    NUMERIC  -- final settled amount (NULL if not closed)
  status            VARCHAR  -- 'open', 'closed', 'in_review', 'litigated'
  resolution_days   INTEGER  -- days taken to resolve
  adjuster_id       VARCHAR  -- e.g. ADJ0001
  region            VARCHAR  -- UK region e.g. 'London', 'Manchester'
  fraud_flag        BOOLEAN  -- true if flagged as fraud
  fraud_score       NUMERIC  -- fraud probability 0.0 to 1.0
  litigation_flag   BOOLEAN  -- true if litigation risk high
  litigation_score  NUMERIC  -- litigation probability 0.0 to 1.0
  notes             TEXT     -- claim notes

TABLE: policies
  policy_number     VARCHAR  -- unique ID
  policy_type       VARCHAR  -- 'vehicle', 'property', 'liability', 'health', 'travel'
  holder_name       VARCHAR  -- policyholder name
  holder_age        INTEGER  -- age of policyholder
  start_date        DATE
  end_date          DATE
  premium_amount    NUMERIC  -- annual premium in GBP
  coverage_limit    NUMERIC  -- maximum coverage in GBP
  region            VARCHAR
  risk_tier         VARCHAR  -- 'low', 'medium', 'high'

TABLE: adjusters
  adjuster_id       VARCHAR  -- unique ID e.g. ADJ0001
  adjuster_name     VARCHAR
  region            VARCHAR
  specialisation    VARCHAR  -- 'vehicle', 'property', 'liability', 'health', 'fraud'
  active            BOOLEAN
  claims_handled    INTEGER
"""

# ── Prompts ───────────────────────────────────────────────────────────────────
SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert PostgreSQL analyst working with insurance claims data.

{schema}

RULES:
- Write ONLY a single valid PostgreSQL SELECT query
- Never use DROP, DELETE, UPDATE, INSERT or any write operations
- Always add LIMIT 100 unless the question asks for aggregates
- Use proper SQL formatting
- Return ONLY the SQL query with no explanation, no markdown, no backticks
- For date comparisons use DATE type casting
- claim_amount and other numerics are stored as NUMERIC type
"""),
    ("human", "Write a SQL query to answer: {question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert insurance claims analyst.
You ran a SQL query and got results. Summarise the findings clearly and concisely
in 2-4 sentences for a claims manager. Focus on the key insight.
Be specific — include numbers, percentages, and trends from the data.
"""),
    ("human", """Question: {question}

SQL Query used:
{sql}

Query Results:
{results}

Provide a clear business summary of these findings:"""),
])


# ── Core functions ────────────────────────────────────────────────────────────

def generate_sql(question: str) -> str:
    """Convert natural language question to SQL query."""
    chain    = SQL_GENERATION_PROMPT | llm
    response = chain.invoke({
        "schema":   DB_SCHEMA,
        "question": question,
    })
    # Clean up any accidental markdown
    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def run_sql(sql: str) -> pd.DataFrame:
    """Execute SQL query safely — SELECT only."""
    sql_upper = sql.upper().strip()

    # Safety check — only allow SELECT
    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    # Block dangerous keywords
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    for keyword in blocked:
        if keyword in sql_upper:
            raise ValueError(f"Blocked keyword detected: {keyword}")

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def summarise_results(question: str, sql: str, df: pd.DataFrame) -> str:
    """Use LLM to summarise query results in plain English."""
    if df.empty:
        return "The query returned no results."

    # Limit rows sent to LLM to avoid token overflow
    sample = df.head(20).to_string(index=False)

    chain    = ANSWER_PROMPT | llm
    response = chain.invoke({
        "question": question,
        "sql":      sql,
        "results":  sample,
    })
    return response.content


def query(question: str) -> dict:
    """
    Full SQL agent pipeline:
    1. Generate SQL from question
    2. Execute against PostgreSQL
    3. Summarise results with LLM

    Returns dict with sql, dataframe, summary, row_count.
    """
    try:
        sql = generate_sql(question)
        df  = run_sql(sql)
        summary = summarise_results(question, sql, df)

        return {
            "success":   True,
            "question":  question,
            "sql":       sql,
            "dataframe": df,
            "summary":   summary,
            "row_count": len(df),
        }

    except Exception as e:
        return {
            "success":   False,
            "question":  question,
            "sql":       "",
            "dataframe": pd.DataFrame(),
            "summary":   f"Error: {str(e)}",
            "row_count": 0,
        }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "How many claims are there by claim type?",
        "Which regions have the highest average fraud score?",
        "What is the average resolution time in days for litigated claims vs non-litigated claims?",
        "Which adjusters have handled the most high fraud risk claims?",
        "What percentage of liability claims are flagged for litigation?",
    ]

    print("Testing SQL Analytics Agent...\n")
    print("=" * 60)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        result = query(question)

        if result["success"]:
            print(f"SQL:\n{result['sql']}\n")
            print(f"Results ({result['row_count']} rows):")
            print(result["dataframe"].to_string(index=False))
            print(f"\nSummary:\n{result['summary']}")
        else:
            print(f"Failed: {result['summary']}")

        print("=" * 60)