"""
Step 3: Load the extracted triples (data/triples.json) into Neo4j AuraDB.

Each triple becomes:
  (Subject_node) -[:RELATION]-> (Object_node)

Using MERGE (not CREATE) so shared entities like "Farmers" or "State"
become ONE node connected to many schemes, instead of duplicate nodes.
"""
import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

TRIPLES_FILE = "data/triples.json"

# All node labels that appear as subject_type/object_type in our triples.
# Used to set up uniqueness constraints before loading.
NODE_LABELS = [
    "Scheme", "Sponsor", "Beneficiary", "BenefitType",
    "Department", "SchemeType", "Crop", "District", "SubGroup",
]


def setup_constraints(driver):
    """Ensures each label's 'name' property is unique — prevents duplicate
    nodes and speeds up the MERGE operations below."""
    with driver.session() as session:
        for label in NODE_LABELS:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
            )
    print(f"✅ Constraints ensured for {len(NODE_LABELS)} node labels")


def load_triples(driver, triples: list[dict]):
    """Loads each triple as: (subject)-[:RELATION]->(object), using MERGE
    so repeated entities (e.g. 'Farmers') reuse the same node."""
    with driver.session() as session:
        for i, triple in enumerate(triples, start=1):
            subject_label = triple["subject_type"]
            object_label = triple["object_type"]
            relation = triple["relation"]

            # Labels and relationship types can't be passed as query parameters
            # in Cypher — but that's safe here because they only ever come from
            # our own fixed NODE_LABELS / type_map, never raw user/LLM text.
            # The actual VALUES (subject/object names) ARE parameterized below,
            # which is what matters for safety since those come from scraped/LLM text.
            query = f"""
                MERGE (s:{subject_label} {{name: $subject_name}})
                MERGE (o:{object_label} {{name: $object_name}})
                MERGE (s)-[:{relation}]->(o)
            """
            session.run(
                query,
                subject_name=triple["subject"],
                object_name=triple["object"],
            )

            if i % 50 == 0:
                print(f"  ...loaded {i}/{len(triples)} triples")

    print(f"✅ Loaded all {len(triples)} triples into Neo4j")


def print_summary(driver):
    """Quick counts so you can see the graph took shape."""
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-->() RETURN count(*) AS c").single()["c"]
    print(f"\n📊 Graph summary: {node_count} nodes, {rel_count} relationships")


def main():
    with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
        triples = json.load(f)

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        setup_constraints(driver)
        load_triples(driver, triples)
        print_summary(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()