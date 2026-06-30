"""
Step 6.1: Load data/triples.json into a NetworkX directed graph.

NetworkX represents each triple as:
  - Two NODES: subject and object (with a 'type' attribute for the label)
  - One DIRECTED EDGE from subject → object (with a 'relation' attribute)

Unlike Neo4j (which persists the graph to a server), this graph lives
entirely in Python memory — it's gone when the Python process exits. We
rebuild it fresh from triples.json each time the app starts, which is
fast enough at this scale (391 triples loads in milliseconds).
"""
import json
import networkx as nx

TRIPLES_FILE = "data/triples.json"


def build_graph(triples_path: str = TRIPLES_FILE) -> nx.DiGraph:
    """
    Reads triples.json and builds a NetworkX DiGraph.

    DiGraph = Directed Graph — edges have a direction (A → B),
    which matters because our relationships always go Scheme → Entity,
    not the reverse.

    Each node gets a 'node_type' attribute (e.g. 'Scheme', 'Crop')
    Each edge gets a 'relation' attribute (e.g. 'TARGETS_CROP')
    """
    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    G = nx.DiGraph()

    for triple in triples:
        subject = triple["subject"]
        obj     = triple["object"]
        relation = triple["relation"]
        subject_type = triple["subject_type"]
        object_type  = triple["object_type"]

        # add_node is safe to call multiple times for the same node —
        # NetworkX silently ignores duplicates (similar to Neo4j's MERGE).
        G.add_node(subject, node_type=subject_type)
        G.add_node(obj,     node_type=object_type)

        # add_edge connects subject → object and stores the relation name
        # as an edge attribute so we can filter by relationship type later.
        G.add_edge(subject, obj, relation=relation)

    return G


def print_graph_summary(G: nx.DiGraph):
    """Prints a quick overview of the graph so we can sanity-check it."""
    print(f"\n📊 Graph Summary")
    print(f"   Nodes : {G.number_of_nodes()}")
    print(f"   Edges : {G.number_of_edges()}")

    # Count nodes by their type attribute
    type_counts: dict[str, int] = {}
    for node, attrs in G.nodes(data=True):
        t = attrs.get("node_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n   Node breakdown by type:")
    for node_type, count in sorted(type_counts.items()):
        print(f"     {node_type:<20} {count}")

    # Sample — all Scheme nodes pointing to "Farmers"
    print(f"\n   Sample: Schemes connected to 'Farmers' beneficiary:")
    if "Farmers" in G:
        predecessors = list(G.predecessors("Farmers"))
        for p in predecessors[:5]:
            print(f"     · {p}")
        if len(predecessors) > 5:
            print(f"     ... and {len(predecessors) - 5} more")
    else:
        print("     'Farmers' node not found — check triples.json")


if __name__ == "__main__":
    print("Building NetworkX graph from triples.json...")
    G = build_graph()
    print_graph_summary(G)
    print("\n✅ Graph built successfully — ready for Step 6.2 (traversal queries)")