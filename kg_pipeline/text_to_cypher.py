"""
Step 4: Text-to-Cypher retrieval.

Pipeline for each question:
  1. GENERATE  -> LLM writes a Cypher query based on the question + our schema.
  2. VALIDATE  -> reject the query if it contains any write/destructive keyword.
                  (Critical safety step since the LLM has free-form query power.)
  3. EXECUTE   -> run the validated, read-only query against Neo4j.
  4. ANSWER    -> feed the raw graph results to a second LLM call that writes
                  a clear, farmer-friendly final answer.
"""
import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# The graph schema, described in plain text for the LLM.
# Keeping this accurate and in sync with extract_triples.py / load_to_neo4j.py
# is what makes text-to-Cypher reliable — the LLM can only write queries
# that match patterns it knows exist.
# ---------------------------------------------------------------------------
GRAPH_SCHEMA = """
Node labels, their 'name' property, and EXAMPLE values (use these to pick the
right label — e.g. "women farmers" is a SubGroup, NOT a Beneficiary):
  Scheme       — e.g. "Training to Farmers", "Distribution of Gypsum"
  Sponsor      — e.g. "State", "Centre"
  Beneficiary  — e.g. "Farmers" (the broad, general category only)
  BenefitType  — e.g. "Subsidy", "Grants"
  Department   — e.g. "Agriculture - Farmers Welfare Department"
  SchemeType   — e.g. "na" (often unused)
  Crop         — e.g. "Maize", "Paddy", "Pulses", "Groundnut"
  District     — e.g. "Coimbatore", "Madurai", "Salem"
  SubGroup     — e.g. "Women Farmers", "SC/ST Farmers", "Small Farmers"
                 (specific target sub-groups, NOT general "Farmers")

Relationships (always Scheme -> something):
  (Scheme)-[:SPONSORED_BY]->(Sponsor)
  (Scheme)-[:TARGETS_BENEFICIARY]->(Beneficiary)
  (Scheme)-[:HAS_BENEFIT_TYPE]->(BenefitType)
  (Scheme)-[:BELONGS_TO_DEPARTMENT]->(Department)
  (Scheme)-[:HAS_SCHEME_TYPE]->(SchemeType)
  (Scheme)-[:TARGETS_CROP]->(Crop)
  (Scheme)-[:TARGETS_DISTRICT]->(District)
  (Scheme)-[:TARGETS_SUBGROUP]->(SubGroup)
"""

# ---------------------------------------------------------------------------
# STAGE 1 — Generate Cypher from the question
# ---------------------------------------------------------------------------

CYPHER_PROMPT = ChatPromptTemplate.from_template("""
You write Cypher queries for a Neo4j database about Tamil Nadu agriculture schemes.

Graph schema:
{schema}

Rules:
- Write ONE read-only Cypher query (MATCH ... RETURN ...) that answers the question.
- NEVER use CREATE, MERGE, DELETE, SET, REMOVE, DROP, or CALL.
- IMPORTANT: Sponsor, Beneficiary, BenefitType, Department, SchemeType, Crop,
  District, and SubGroup nodes are NEVER connected to each other directly —
  they connect ONLY to Scheme. If a question mentions two of these (e.g. a
  crop AND a district), match them as TWO SEPARATE relationships from the
  SAME Scheme node, like:
  MATCH (s:Scheme)-[:TARGETS_CROP]->(c:Crop), (s)-[:TARGETS_DISTRICT]->(d:District)
- Always RETURN scheme names as s.name (alias the Scheme node as s).
- Use case-insensitive matching for property values, e.g.:
  WHERE toLower(d.name) = toLower("coimbatore")
- Return ONLY the Cypher query text. No markdown, no explanation, no ```.

Question: {question}
""")

# Keywords that should never appear in a query we're about to execute.
FORBIDDEN_KEYWORDS = [
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE",
    "DROP", "CALL", "LOAD CSV", "DETACH",
]


def generate_cypher(question: str) -> str:
    chain = CYPHER_PROMPT | llm
    response = chain.invoke({"schema": GRAPH_SCHEMA, "question": question})
    query = response.content.strip()

    # Defensive cleanup in case the LLM wraps it in a code fence anyway
    if query.startswith("```"):
        query = query.strip("`").replace("cypher", "", 1).strip()

    return query


def is_safe_query(query: str) -> bool:
    """Guardrail: reject the query if ANY destructive keyword appears,
    as a whole word (so 'Department' doesn't accidentally trip on nothing,
    but 'SET' as a real clause would)."""
    upper_query = query.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_query):
            return False
    return True


# ---------------------------------------------------------------------------
# STAGE 2 — Execute against Neo4j
# ---------------------------------------------------------------------------

def run_query(driver, query: str) -> list[dict]:
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]


# ---------------------------------------------------------------------------
# STAGE 3 — Turn raw graph results into a farmer-friendly answer
# ---------------------------------------------------------------------------

def format_results(results: list[dict]) -> str:
    """Turns raw Neo4j records into a clean, readable list for the LLM —
    instead of dumping Python dicts with keys like 's.name'."""
    if not results:
        return "No matching results were found in the graph."

    lines = []
    for record in results:
        # Join all values in the record (usually just one: the scheme name)
        values = [str(v) for v in record.values()]
        lines.append("- " + ", ".join(values))
    return "\n".join(lines)


ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant for Tamil Nadu farmers, answering questions about
government Agriculture schemes.

The list below was already retrieved by running a database query that
filters specifically for what the question asks. It IS the correct, complete
answer — do not second-guess whether each item is relevant, and do not say
you lack information unless the list below is literally empty.

Question: {question}

Matching schemes from the database:
{results}

Write a simple, clear answer a farmer can easily understand, listing the
scheme names above. If the list is empty, say you don't have that
information and suggest contacting the local Agriculture Officer.
""")


def answer_question(driver, question: str) -> dict:
    cypher_query = generate_cypher(question)

    if not is_safe_query(cypher_query):
        return {
            "question": question,
            "cypher": cypher_query,
            "results": None,
            "answer": "⚠️ Generated query was blocked for safety reasons (contained a write/destructive keyword).",
        }

    try:
        results = run_query(driver, cypher_query)
    except Exception as e:
        return {
            "question": question,
            "cypher": cypher_query,
            "results": None,
            "answer": f"⚠️ Query failed to run: {e}",
        }

    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"question": question, "results": format_results(results)})

    return {
        "question": question,
        "cypher": cypher_query,
        "results": results,
        "answer": response.content.strip(),
    }


# ---------------------------------------------------------------------------
# Quick manual test from the command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    test_questions = [
        "Which schemes help farmers in Coimbatore?",
        "What schemes support women farmers?",
        "Are there any schemes for maize farmers?",
    ]

    try:
        for q in test_questions:
            print(f"\n❓ {q}")
            result = answer_question(driver, q)
            print(f"   Cypher: {result['cypher']}")
            print(f"   Answer: {result['answer']}")
    finally:
        driver.close()