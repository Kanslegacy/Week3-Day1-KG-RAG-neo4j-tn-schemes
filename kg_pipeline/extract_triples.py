"""
Step 2: Extract Knowledge Graph triples from the TN agriculture scheme data.

Two extraction methods combined:
1. Deterministic (no LLM) — straight from structured fields like
   "Sponsered By", "Beneficiaries", "Types of Benefits", "Concerned Department".
2. LLM-based (GPT-4o-mini) — reads the free-text "Description" field and pulls
   out crops, districts, and target sub-groups mentioned in the text.

Output: data/triples.json — a flat list of triples, each shaped like:
{
  "scheme": "Training to Farmers",
  "subject": "Training to Farmers",
  "subject_type": "Scheme",
  "relation": "TARGETS_BENEFICIARY",
  "object": "Farmers",
  "object_type": "Beneficiary"
}
"""
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

INPUT_FILE = "data/schemes_data.json"
OUTPUT_FILE = "data/triples.json"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# PART A — Deterministic triples (straight from structured fields)
# ---------------------------------------------------------------------------

def build_structured_triples(scheme: dict) -> list[dict]:
    """Builds triples directly from fields that already ARE relationships —
    no LLM needed, so this is instant and 100% consistent."""
    name = scheme["name"]
    triples = []

    field_map = [
        ("Sponsered By", "SPONSORED_BY", "Sponsor"),
        ("Beneficiaries", "TARGETS_BENEFICIARY", "Beneficiary"),
        ("Types of Benefits", "HAS_BENEFIT_TYPE", "BenefitType"),
        ("Concerned Department", "BELONGS_TO_DEPARTMENT", "Department"),
        ("Scheme Type", "HAS_SCHEME_TYPE", "SchemeType"),
    ]

    for field_name, relation, object_type in field_map:
        value = scheme.get(field_name, "").strip()
        # skip empty / placeholder values like "na" or "-"
        if value and value.lower() not in ("na", "-", "n/a"):
            triples.append({
                "scheme": name,
                "subject": name,
                "subject_type": "Scheme",
                "relation": relation,
                "object": value,
                "object_type": object_type,
            })

    return triples


# ---------------------------------------------------------------------------
# PART B — LLM-based triples (from the free-text Description)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are extracting structured facts from a Tamil Nadu government agriculture
scheme description. Only use these THREE relation types — do not invent others:

- TARGETS_CROP        (object is a crop, e.g. Paddy, Maize, Groundnut, Cotton, Sugarcane)
- TARGETS_DISTRICT     (object is a Tamil Nadu district, e.g. Coimbatore, Madurai, Salem)
- TARGETS_SUBGROUP     (object is a specific beneficiary sub-group mentioned,
                         e.g. "Small Farmers", "SC/ST Farmers", "Women Farmers")

Scheme name: {scheme_name}

Description:
{description}

Return ONLY a JSON array (no markdown, no explanation) of objects shaped like:
[{{"relation": "TARGETS_CROP", "object": "Paddy"}}, ...]

If nothing relevant is mentioned, return an empty array: []
""")

# ---------------------------------------------------------------------------
# Entity normalization — different schemes describe the same group/crop with
# different words (e.g. "Farm Women" vs "Women Farmers"). Without this step,
# each variant becomes its OWN node in the graph, fragmenting connections
# that should exist. Add to this map as you spot more variants while
# reviewing the full 54-scheme output.
# ---------------------------------------------------------------------------
CANONICAL_MAP = {
    "farm women": "Women Farmers",
    "women farmers": "Women Farmers",
    "sc / st farmers": "SC/ST Farmers",
    "sc/st farmers": "SC/ST Farmers",
    "small / marginal farmers": "Small Farmers",
    "small/marginal farmers": "Small Farmers",
    "small farmers": "Small Farmers",
    "marginal farmers": "Small Farmers",
}

def normalize_entity(value: str) -> str:
    """Maps known variant phrasings to one canonical name."""
    key = value.strip().lower()
    return CANONICAL_MAP.get(key, value.strip())

def llm_extract_triples(scheme: dict) -> list[dict]:
    name = scheme["name"]
    description = scheme.get("Description", "").strip()

    if not description or description.lower() in ("na", "-"):
        return []

    chain = EXTRACTION_PROMPT | llm
    response = chain.invoke({"scheme_name": name, "description": description})

    raw = response.content.strip()
    # GPT-4o-mini sometimes wraps JSON in ```json fences even when told not to — strip them
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ⚠️ Could not parse LLM output for '{name}', skipping. Raw: {raw[:100]}")
        return []

    type_map = {
        "TARGETS_CROP": "Crop",
        "TARGETS_DISTRICT": "District",
        "TARGETS_SUBGROUP": "SubGroup",
    }

    triples = []
    for fact in facts:
        relation = fact.get("relation")
        obj = fact.get("object", "").strip()
        obj = normalize_entity(obj)
        if relation in type_map and obj:
            triples.append({
                "scheme": name,
                "subject": name,
                "subject_type": "Scheme",
                "relation": relation,
                "object": obj,
                "object_type": type_map[relation],
            })
    return triples


# ---------------------------------------------------------------------------
# PART C — Run both, combine, save
# ---------------------------------------------------------------------------

def main(limit: int | None = None):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    if limit:
        schemes = schemes[:limit]
        print(f"⚠️ Running on first {limit} schemes only (test mode)")

    all_triples = []

    for i, scheme in enumerate(schemes, start=1):
        print(f"[{i}/{len(schemes)}] {scheme['name']}")

        structured = build_structured_triples(scheme)
        llm_based = llm_extract_triples(scheme)

        all_triples.extend(structured)
        all_triples.extend(llm_based)

        print(f"   → {len(structured)} structured + {len(llm_based)} LLM-extracted triples")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved {len(all_triples)} total triples to {OUTPUT_FILE}")


if __name__ == "__main__":
    # Start small on purpose — test on 3 schemes first to check quality/cost
    # before running on all 54. Change to main() (no limit) once you're happy.
    main()