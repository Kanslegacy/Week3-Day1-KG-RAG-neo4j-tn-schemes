"""
Step 6.3: NetworkX answer pipeline.

Two-stage pipeline:
  STAGE 1 — LLM reads the question and extracts structured intent as JSON.
             No Cypher, no graph knowledge needed — just entity extraction.
  STAGE 2 — Python calls the right nx_retriever function(s) based on that
             JSON, then a second LLM call formats the results into a
             farmer-friendly answer.

Compare with text_to_cypher.py (Neo4j):
  Neo4j:    LLM writes a full Cypher query → execute against a server
  NetworkX: LLM extracts JSON keywords → Python traverses local graph

The retrieval logic is completely different, but the final answer
generation is identical — same ANSWER_PROMPT, same formatting.
"""
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

try:
    from kg_pipeline.nx_retriever import (
        schemes_by_district, schemes_by_crop, schemes_by_subgroup,
        schemes_by_benefit_type, schemes_by_sponsor, schemes_by_multi,
        list_all_nodes_of_type, G,
    )
except ModuleNotFoundError:
    from nx_retriever import (
        schemes_by_district, schemes_by_crop, schemes_by_subgroup,
        schemes_by_benefit_type, schemes_by_sponsor, schemes_by_multi,
        list_all_nodes_of_type, G,
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# STAGE 1 — Intent extraction
# ---------------------------------------------------------------------------

INTENT_PROMPT = ChatPromptTemplate.from_template("""
You extract structured search intent from a farmer's question about Tamil
Nadu Agriculture schemes. Return ONLY a JSON object — no markdown, no text.

Known values that exist in the database (use ONLY these, do not invent):
  Districts : {districts}
  Crops     : {crops}
  SubGroups : {subgroups}
  Benefits  : Subsidy, Grants
  Sponsors  : State, Centre

JSON schema to return:
{{
  "district"    : "string or null",
  "crop"        : "string or null",
  "subgroup"    : "string or null",
  "benefit_type": "string or null",
  "sponsor"     : "string or null",
  "list_all"    : false
}}

Set "list_all" to true if the farmer is asking to see all schemes with
no specific filter (e.g. "show me all schemes", "what schemes exist").
Set everything else to null if not mentioned in the question.

Question: {question}
""")


def extract_intent(question: str) -> dict:
    districts = list_all_nodes_of_type("District")
    crops     = list_all_nodes_of_type("Crop")
    subgroups = list_all_nodes_of_type("SubGroup")

    chain = INTENT_PROMPT | llm
    response = chain.invoke({
        "question" : question,
        "districts": ", ".join(districts),
        "crops"    : ", ".join(crops),
        "subgroups": ", ".join(subgroups),
    })

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ⚠️ Could not parse intent JSON: {raw[:100]}")
        return {
            "district": None, "crop": None, "subgroup": None,
            "benefit_type": None, "sponsor": None, "list_all": False,
        }


# ---------------------------------------------------------------------------
# STAGE 2a — Retrieval
# ---------------------------------------------------------------------------

def retrieve(intent: dict) -> list[str]:
    district     = intent.get("district")
    crop         = intent.get("crop")
    subgroup     = intent.get("subgroup")
    benefit_type = intent.get("benefit_type")
    sponsor      = intent.get("sponsor")
    list_all     = intent.get("list_all", False)

    multi_filters = [f for f in [district, crop, subgroup] if f]

    if list_all:
        return sorted([
            n for n, a in G.nodes(data=True)
            if a.get("node_type") == "Scheme"
        ])

    if len(multi_filters) > 1:
        return schemes_by_multi(
            district=district,
            crop=crop,
            subgroup=subgroup,
        )

    if district:     return schemes_by_district(district)
    if crop:         return schemes_by_crop(crop)
    if subgroup:     return schemes_by_subgroup(subgroup)
    if benefit_type: return schemes_by_benefit_type(benefit_type)
    if sponsor:      return schemes_by_sponsor(sponsor)

    return []


# ---------------------------------------------------------------------------
# STAGE 2b — Answer generation
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant for Tamil Nadu farmers, answering questions about
government Agriculture schemes.

The list below was already retrieved by querying a knowledge graph that
filters specifically for what the question asks. It IS the correct, complete
answer — do not second-guess relevance, and do not say you lack information
unless the list is literally empty.

Question: {question}

Matching schemes from the knowledge graph:
{results}

Write a simple, clear answer a farmer can easily understand, listing the
scheme names. If the list is empty, say you don't have that information
and suggest contacting the local Agriculture Officer.
""")


def format_results(schemes: list[str]) -> str:
    if not schemes:
        return "No matching schemes found."
    return "\n".join(f"- {s}" for s in schemes)


def answer_question_nx(question: str) -> dict:
    intent  = extract_intent(question)
    schemes = retrieve(intent)

    chain    = ANSWER_PROMPT | llm
    response = chain.invoke({
        "question": question,
        "results" : format_results(schemes),
    })

    return {
        "question": question,
        "intent"  : intent,
        "schemes" : schemes,
        "answer"  : response.content.strip(),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "Which schemes help farmers in Coimbatore?",
        "What schemes support women farmers?",
        "Are there any schemes for maize farmers?",
        "Which schemes help women farmers growing maize in Coimbatore?",
    ]

    for q in test_questions:
        print(f"\n❓ {q}")
        result = answer_question_nx(q)
        print(f"   Intent : {result['intent']}")
        print(f"   Answer : {result['answer']}")