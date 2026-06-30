"""
Step 6.2: NetworkX traversal functions.

These Python functions replace the LLM-generated Cypher queries from
text_to_cypher.py. Instead of writing Cypher and sending it to Neo4j,
we traverse the in-memory NetworkX graph directly using Python.

Pattern for every query:
  1. Find the target node by name (case-insensitive scan)
  2. Use G.predecessors(node) → all nodes with edges pointing TO it
  3. Filter by edge 'relation' attribute to get only the right rel type
  4. Filter further to confirm predecessor is a Scheme node

Why predecessors and not successors?
  Our edges go  Scheme → Entity  (e.g. Scheme → Coimbatore).
  So to find "which Schemes link to Coimbatore", we look at nodes
  that have an edge POINTING TO "Coimbatore" — those are predecessors.
"""
import networkx as nx
try:
    from kg_pipeline.nx_graph import build_graph
except ModuleNotFoundError:
    from nx_graph import build_graph

# Build the graph once when this module is imported.
# At 391 triples this takes <10ms — no need to rebuild per query.
G = build_graph()


def find_node(G: nx.DiGraph, name: str) -> str | None:
    """
    Returns the exact node name as stored in the graph, regardless of
    how the user typed it — e.g. "coimbatore" → "Coimbatore".
    Returns None if nothing matches.
    """
    name_lower = name.strip().lower()
    for node in G.nodes():
        if node.lower() == name_lower:
            return node
    return None


def schemes_via_relation(
    G: nx.DiGraph,
    target_name: str,
    relation: str
) -> list[str]:
    """
    Core traversal used by every query below.

    Finds the target node (case-insensitive), then returns all Scheme
    nodes that have an edge of the given 'relation' type pointing to it.
    """
    node = find_node(G, target_name)
    if node is None:
        return []

    results = []
    for predecessor in G.predecessors(node):
        if G.nodes[predecessor].get("node_type") != "Scheme":
            continue
        edge_data = G.edges[predecessor, node]
        if edge_data.get("relation") == relation:
            results.append(predecessor)

    return results


def schemes_by_district(district: str) -> list[str]:
    return schemes_via_relation(G, district, "TARGETS_DISTRICT")


def schemes_by_crop(crop: str) -> list[str]:
    return schemes_via_relation(G, crop, "TARGETS_CROP")


def schemes_by_subgroup(subgroup: str) -> list[str]:
    return schemes_via_relation(G, subgroup, "TARGETS_SUBGROUP")


def schemes_by_benefit_type(benefit_type: str) -> list[str]:
    return schemes_via_relation(G, benefit_type, "HAS_BENEFIT_TYPE")


def schemes_by_sponsor(sponsor: str) -> list[str]:
    return schemes_via_relation(G, sponsor, "SPONSORED_BY")


def schemes_by_multi(
    district: str | None = None,
    crop: str | None = None,
    subgroup: str | None = None,
) -> list[str]:
    """
    Multi-condition query — find schemes matching ALL provided filters.
    This is the key power of KG-RAG over vector search: we can intersect
    multiple exact graph relationships, not just rank by similarity.
    """
    result_sets = []

    if district:
        result_sets.append(set(schemes_by_district(district)))
    if crop:
        result_sets.append(set(schemes_by_crop(crop)))
    if subgroup:
        result_sets.append(set(schemes_by_subgroup(subgroup)))

    if not result_sets:
        return []

    combined = result_sets[0]
    for s in result_sets[1:]:
        combined = combined & s

    return sorted(combined)


def list_all_nodes_of_type(node_type: str) -> list[str]:
    return sorted([
        node for node, attrs in G.nodes(data=True)
        if attrs.get("node_type") == node_type
    ])


if __name__ == "__main__":
    print("=== Testing NetworkX traversal queries ===\n")

    print("1. Schemes in Coimbatore:")
    for s in schemes_by_district("Coimbatore"):
        print(f"   · {s}")

    print("\n2. Schemes for Women Farmers:")
    for s in schemes_by_subgroup("Women Farmers"):
        print(f"   · {s}")

    print("\n3. Schemes for Maize:")
    for s in schemes_by_crop("Maize"):
        print(f"   · {s}")

    print("\n4. Multi-filter — Coimbatore AND Maize:")
    for s in schemes_by_multi(district="Coimbatore", crop="Maize"):
        print(f"   · {s}")

    print("\n5. All crops in the graph:")
    print("  ", list_all_nodes_of_type("Crop"))

    print("\n6. All districts in the graph:")
    print("  ", list_all_nodes_of_type("District"))

    print("\n✅ All traversal queries complete")