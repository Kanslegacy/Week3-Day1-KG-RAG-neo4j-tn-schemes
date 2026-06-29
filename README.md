# 🌾 TN Agriculture Scheme Assistant — Knowledge Graph RAG (KG-RAG)

An AI chatbot that answers questions about Tamil Nadu Government Agriculture
& Farmers Welfare Department schemes — built using **Knowledge Graph
Retrieval-Augmented Generation (KG-RAG)**, with **Neo4j AuraDB** as the
graph storage layer.

> Ask a question in plain language → an LLM converts it into a graph
> query → the query traverses real relationships between schemes, crops,
> districts, and farmer groups → an AI model answers using only the
> retrieved facts.

🔗 **Official data source:** [tn.gov.in – Agriculture Schemes](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==)

This project is a follow-up to an earlier **Vector RAG** version of the
same chatbot (FAISS + embeddings). Building both side-by-side was
intentional — it's the clearest way to *feel* the difference between
"search by meaning" (vector RAG) and "search by relationship" (KG-RAG).

---

## 1. Why a Knowledge Graph instead of (or alongside) Vector Search?

Vector RAG retrieves text chunks that are *semantically similar* to a
question. That's great for "what is this scheme about," but weak for
**relationship questions** — e.g. *"Which schemes help women farmers
growing maize in Coimbatore?"* A plain similarity search has no real
concept of "this scheme is connected to this crop, this district, and
this group" — it just guesses based on word overlap.

A **Knowledge Graph** stores facts as explicit connections:

```
(Scheme) --[TARGETS_CROP]--> (Maize)
(Scheme) --[TARGETS_DISTRICT]--> (Coimbatore)
(Scheme) --[TARGETS_SUBGROUP]--> (Women Farmers)
```

Answering a multi-condition question becomes a graph traversal — exact
and complete — instead of a similarity guess.

---

## 2. Architecture Overview

```
┌─────────────────────┐
│ schemes_data.json    │   (scraped from tn.gov.in)
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ STEP 2: Triple Extraction                    │
│  • Deterministic triples from structured      │
│    fields (Sponsor, Beneficiary, Dept, etc.)  │
│  • LLM (GPT-4o-mini) extracts Crop / District/ │
│    SubGroup triples from free-text Description │
│  • Entity normalization (e.g. "Farm Women"     │
│    → "Women Farmers")                          │
└──────────┬────────────────────────────────────┘
           │  data/triples.json
           ▼
┌─────────────────────────────────────────────┐
│ STEP 3: Load into Neo4j AuraDB                │
│  • MERGE (not CREATE) → shared entities        │
│    become single connected nodes               │
│  • Uniqueness constraints per node label        │
└──────────┬────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ STEP 4: Text-to-Cypher Retrieval               │
│  • LLM writes a Cypher query from the question  │
│  • Safety guardrail blocks any write/destructive │
│    keyword before execution                      │
│  • Query runs against the graph                  │
│  • Second LLM call turns raw results into a       │
│    farmer-friendly natural language answer        │
└──────────┬────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ STEP 5: Streamlit Chat UI                      │
│  • Chat history, live question input            │
│  • Shows the generated Cypher per answer         │
│    (transparency into how the graph was queried) │
└─────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI `gpt-4o-mini` (via LangChain) |
| Graph Database | Neo4j AuraDB (Free tier) |
| Orchestration | LangChain (LCEL `\|` chains) |
| UI | Streamlit |
| Data source | Scraped from tn.gov.in (BeautifulSoup) |

---

## 4. Project Structure

```
tn-kg-rag/
├── .streamlit/
│   └── config.toml          # light agriculture-green theme
├── data/
│   ├── schemes_data.json    # raw scraped scheme data (54 schemes)
│   └── triples.json         # extracted KG triples
├── kg_pipeline/
│   ├── test_connection.py   # Step 1: Neo4j connectivity check
│   ├── extract_triples.py   # Step 2: triple extraction
│   ├── load_to_neo4j.py     # Step 3: load triples into Neo4j
│   └── text_to_cypher.py    # Step 4: retrieval + answer generation
├── scraper/                 # original scraping scripts (data source)
├── app.py                   # Step 5: Streamlit chat UI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 5. The Knowledge Graph Schema

**Node labels:**
`Scheme`, `Sponsor`, `Beneficiary`, `BenefitType`, `Department`,
`SchemeType`, `Crop`, `District`, `SubGroup`

**Relationships (always `Scheme` → something):**

| Relationship | Source | Example |
|---|---|---|
| `SPONSORED_BY` | structured field | `(Scheme)→(State)` |
| `TARGETS_BENEFICIARY` | structured field | `(Scheme)→(Farmers)` |
| `HAS_BENEFIT_TYPE` | structured field | `(Scheme)→(Subsidy)` |
| `BELONGS_TO_DEPARTMENT` | structured field | `(Scheme)→(Agriculture Dept)` |
| `HAS_SCHEME_TYPE` | structured field | `(Scheme)→(...)` |
| `TARGETS_CROP` | LLM-extracted from Description | `(Scheme)→(Maize)` |
| `TARGETS_DISTRICT` | LLM-extracted from Description | `(Scheme)→(Coimbatore)` |
| `TARGETS_SUBGROUP` | LLM-extracted from Description | `(Scheme)→(Women Farmers)` |

**Final graph size:** 54 schemes → 93 nodes, 391 relationships.

> **Design note:** non-`Scheme` node types are never connected to each
> other directly — every relationship originates from a `Scheme` node.
> This keeps the schema simple and makes text-to-Cypher generation more
> reliable, at the cost of not modeling deeper relationships (e.g.
> crop ↔ district co-occurrence) that a more advanced KG could capture.

---

## 6. Setup

```bash
git clone <this-repo>
cd tn-kg-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in real values
```

`.env` requires:
```
OPENAI_API_KEY=...
NEO4J_URI=neo4j+ssc://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

> **Note on `neo4j+ssc://`:** the standard `neo4j+s://` scheme failed in
> development with `SSLCertVerificationError` / "Unable to retrieve
> routing information" — traced to TLS interception on the local
> network (common with antivirus "HTTPS scanning" or campus/corporate
> networks). `neo4j+ssc://` connects encrypted but skips certificate
> verification, which resolved it. For production use, the better fix
> is adding a `*.neo4j.io` exception in the intercepting software and
> reverting to `neo4j+s://`.

Run the pipeline in order:
```bash
python kg_pipeline/test_connection.py    # verify Neo4j connectivity
python kg_pipeline/extract_triples.py    # build data/triples.json
python kg_pipeline/load_to_neo4j.py      # load triples into the graph
streamlit run app.py                     # launch the chatbot
```

---

## 7. Safety: Text-to-Cypher Guardrails

Since the LLM generates Cypher queries directly (rather than filling
fixed templates), every generated query is checked **before execution**
against a list of destructive keywords (`CREATE`, `MERGE`, `DELETE`,
`SET`, `REMOVE`, `DROP`, `CALL`, `DETACH`). Any match blocks execution
entirely. This is on top of explicit prompt instructions telling the
LLM to only write read-only queries — defense in depth, not reliance
on the LLM following instructions alone.

---

## 8. Challenges Faced (and how they were solved)

1. **Entity resolution** — different schemes phrased the same group
   differently (e.g. "Farm Women" vs "Women Farmers"). Without
   normalization, these become separate, disconnected graph nodes,
   silently breaking multi-hop queries. Solved with a canonical-name
   mapping applied right after LLM extraction.

2. **Network/TLS connection failures** — `neo4j+s://` repeatedly failed
   with routing/SSL errors. Root-caused to TLS interception, not
   credentials. Resolved using `neo4j+ssc://`.

3. **Hallucinated relationship paths** — the text-to-Cypher LLM
   sometimes chained two non-`Scheme` node types together (e.g.
   `Beneficiary → Crop`), which doesn't exist in the graph, silently
   returning zero results without erroring. Fixed by explicitly stating
   the schema constraint in the prompt.

4. **Retrieval vs. generation bug** — a "women farmers" query correctly
   retrieved 14 matching schemes from Neo4j, but the final-answer LLM
   call still claimed it had no data. Root cause: the LLM had no
   explicit signal that the retrieved list was *already* the filtered
   answer, so it second-guessed relevance. Fixed by restructuring the
   answer prompt to state the data is pre-filtered and should be
   trusted directly — a good example of retrieval and generation
   needing to be debugged as separate stages.

---

## 9. Example Questions

- "Which schemes help farmers in Coimbatore?"
- "What schemes support women farmers?"
- "Are there any schemes for maize farmers?"

---

## 10. Planned Next Step

A second graph storage backend using **NetworkX** (in-memory graph
library) will be added, using the same `data/triples.json`, to compare
a managed cloud graph database against a lightweight local graph
structure — same extraction pipeline, different storage/query layer.

---

## 11. Credits

Data source: Tamil Nadu Government Agriculture & Farmers Welfare
Department scheme listings (tn.gov.in). Built as a learning project to
understand Knowledge Graph RAG architecture end-to-end.
